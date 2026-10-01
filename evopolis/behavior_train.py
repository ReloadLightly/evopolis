"""Fixed-budget Task 03 training, benchmark and atomic continuation."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch

from .behavior_models import BehaviorModel, FAMILIES, group_nll
from .sources import sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/task03"
CONFIG = ROOT / "configs/task03.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save(value, partial)
    partial.replace(path)


def runtime_setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.version.cuda is not None:
        raise RuntimeError("Task 03 requires the pinned CPU-only build")


def tensor_data(arrays):
    return {"x": torch.tensor(arrays["x"], dtype=torch.float32),
            "n": torch.tensor(arrays["n"], dtype=torch.int64),
            "y": torch.tensor(arrays["y"], dtype=torch.int64)}


def forward_groups(model, x):
    groups, players, steps, inputs = x.shape
    raw, _ = model(x.reshape(groups * players, steps, inputs))
    return raw.reshape(groups, players, steps, 5)


def score(model, data, indices, microbatch=2):
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, len(indices), microbatch):
            ids = indices[offset:offset + microbatch]
            values.extend(group_nll(forward_groups(model, data["x"][ids]),
                                    data["n"][ids], data["y"][ids]).tolist())
    return float(np.mean(values))


def train_epoch(model, optimizer, data, indices, shuffle, microbatch=2):
    model.train()
    order = shuffle.permutation(indices)
    if len(order) % 8:
        raise ValueError("Effective optimizer batch must have eight complete groups")
    values = []
    for start in range(0, len(order), 8):
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, start + 8, microbatch):
            ids = order[offset:offset + microbatch]
            losses = group_nll(forward_groups(model, data["x"][ids]), data["n"][ids], data["y"][ids])
            values.extend(losses.detach().tolist())
            (losses.sum() / 8).backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite training gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return float(np.mean(values))


def new_fit(family, seed):
    torch.manual_seed(seed)
    model = BehaviorModel(family)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer, np.random.Generator(np.random.PCG64(seed))


def experiment_identity():
    config = json.loads(CONFIG.read_text())
    required = {"epochs": 120, "lr": 1e-3, "weight_decay": 1e-4,
                "clip_grad_norm": 1.0, "effective_batch_groups": 8,
                "microbatch_groups": 2, "threads": 1, "num_workers": 0,
                "families": list(FAMILIES), "training_seeds": [17, 29, 43]}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Configuration differs from the declared fixed experiment")
    files = ["evopolis/behavior_models.py", "evopolis/behavior_train.py", "evopolis/behavior_data.py"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    frozen = RESULTS / "frozen_training.json"
    if frozen.exists():
        # Publication changes HEAD, not the experiment. Keep the original base
        # revision as provenance; the relevant source-file hashes below still
        # require byte-identical training code for any continuation.
        revision = json.loads(frozen.read_text())["identity"]["code_revision"]
    return {"source_sha256": json.loads((ROOT / "evopolis/source_checksums.json").read_text())["sustainable_behavior.csv"],
            "split_sha256": sha256(RESULTS / "split.json"), "config_sha256": sha256(CONFIG),
            "code_revision": revision,
            "code_sha256": {name: sha256(ROOT / name) for name in files},
            "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)},
            "threads": 1, "microbatch_groups": 2, "effective_batch_groups": 8, "num_workers": 0,
            "configuration": config}


def benchmark(data, train_ids):
    rows = []
    for family in FAMILIES:
        model, optimizer, shuffle = new_fit(family, 17)
        started = time.monotonic()
        losses = [train_epoch(model, optimizer, data, train_ids, shuffle) for _ in range(2)]
        seconds = time.monotonic() - started
        rows.append({"family": family, "epochs": 2, "seconds": seconds, "train_nll": losses,
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        print(f"Benchmark {family}: two training-only epochs {seconds:.2f}s", flush=True)
        del model, optimizer
    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "fits": rows,
              "estimated_training_seconds_12_fits": sum(r["seconds"] * 60 * 3 for r in rows),
              "estimate_excludes_validation_checkpoint_io": True,
              "choice": {"threads": 1, "microbatch_groups": 2, "effective_batch_groups": 8},
              "state_discarded": True, "test_or_validation_used": False}
    atomic_json(RESULTS / "benchmark.json", result)


def fit(family, seed, data, train_ids, validation_ids, identity, remaining):
    directory = RESULTS / "weights" / f"{family}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    model, optimizer, shuffle = new_fit(family, seed)
    initial = score(model, data, validation_ids)
    logs, first_epoch, best = [], 1, float("inf")
    elapsed_before = 0.0
    if (directory / "last.pt").exists():
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if saved["metadata"] != identity or saved["family"] != family or saved["seed"] != seed:
            raise ValueError("Resume refused: source, split, configuration or training code changed")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        shuffle.bit_generator.state = saved["shuffle_state"]
        first_epoch, best, logs = saved["epoch"] + 1, saved["best_validation_score"], saved["logs"]
        initial, elapsed_before = saved["initial_validation_nll"], saved["elapsed_seconds"]
        print(f"Resume {family}/{seed} at epoch {first_epoch}", flush=True)
    started = time.monotonic()
    for epoch in range(first_epoch, 121):
        epoch_started = time.monotonic()
        train = train_epoch(model, optimizer, data, train_ids, shuffle)
        validation = score(model, data, validation_ids)
        logs.append({"epoch": epoch, "train_nll": train, "validation_nll": validation,
                     "seconds": time.monotonic() - epoch_started})
        base = {"family": family, "seed": seed, "epoch": epoch, "model_state": model.state_dict(),
                "metadata": identity, "validation_score": validation}
        if validation < best:
            best = validation
            atomic_checkpoint(directory / "best.pt", base)
        atomic_checkpoint(directory / "last.pt", dict(base, optimizer_state=optimizer.state_dict(),
                          torch_rng_state=torch.get_rng_state(), shuffle_state=shuffle.bit_generator.state,
                          best_validation_score=best, logs=logs, initial_validation_nll=initial,
                          elapsed_seconds=elapsed_before + time.monotonic() - started))
        atomic_json(directory / "training.json", logs)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{family}/{seed} epoch {epoch:3}/120 train {train:.5f} val {validation:.5f} best {best:.5f}; "
                  f"{elapsed_before + time.monotonic()-started:.1f}s; {remaining} fits after this", flush=True)
    # Recover a log write interrupted after the atomic last-checkpoint rename.
    atomic_json(directory / "training.json", logs)
    selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    atomic_json(directory / "metadata.json", dict(identity, family=family, seed=seed,
                selected_epoch=selected["epoch"], validation_score=selected["validation_score"],
                initial_validation_nll=initial, parameter_count=sum(p.numel() for p in model.parameters()),
                best_sha256=sha256(directory / "best.pt"), last_sha256=sha256(directory / "last.pt"),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                elapsed_seconds=elapsed_before + time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "train"))
    args = parser.parse_args()
    from .behavior_data import load
    runtime_setup()
    arrays, manifest = load()
    data = tensor_data(arrays)
    groups = manifest["groups"]
    train_ids = [g["index"] for g in groups if g["split"] == "train"]
    validation_ids = [g["index"] for g in groups if g["split"] == "validation"]
    if len(train_ids) != 96 or len(validation_ids) != 32:
        raise ValueError("Unexpected declared split")
    lock = ROOT / "data/cache/task03/training.lock"
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "benchmark":
            benchmark(data, train_ids)
            return
        if not (RESULTS / "benchmark.json").exists():
            raise ValueError("Run the training-only resource benchmark before main fits")
        identity = experiment_identity()
        frozen_path = RESULTS / "frozen_training.json"
        if frozen_path.exists():
            if json.loads(frozen_path.read_text())["identity"] != identity:
                raise ValueError("Frozen experiment identity differs; diagnose before continuation")
        else:
            atomic_json(frozen_path, {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "identity": identity})
        fits = [(f, s) for f in FAMILIES for s in (17, 29, 43)]
        started = time.monotonic()
        for i, (family, seed) in enumerate(fits):
            fit(family, seed, data, train_ids, validation_ids, identity, len(fits) - i - 1)
        atomic_json(RESULTS / "training_complete.json", {"fits": 12, "epochs_per_fit": 120,
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "invocation_seconds": time.monotonic() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == "__main__":
    main()
