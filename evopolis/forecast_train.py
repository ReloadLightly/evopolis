"""Task 04 continuation without modifying the frozen Task 03 contract."""

import argparse
import copy
import datetime
import fcntl
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from .behavior_data import load
from .behavior_train import (atomic_checkpoint, atomic_json, experiment_identity,
                             new_fit, runtime_setup, score, tensor_data, train_epoch)
from .sources import sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/task04"
PARENT = ROOT / "results/task03"
CONFIG = ROOT / "configs/task04.json"
FITS = [(family, seed) for family in ("feedforward", "recurrent") for seed in (17, 29, 43)]


def preserved():
    record = json.loads((RESULTS / "preservation.json").read_text())
    mismatches = [name for name, digest in record["files"].items() if sha256(ROOT / name) != digest]
    if mismatches:
        raise ValueError(f"Original artifact changed: {mismatches}")
    return len(record["files"])


def identity():
    preserved()
    parent_identity = experiment_identity()
    if parent_identity != json.loads((PARENT / "frozen_training.json").read_text())["identity"]:
        raise ValueError("Original training contract changed")
    config = json.loads(CONFIG.read_text())
    required = {"epochs": 480, "parent_epoch": 120, "additional_epochs_total": 2160,
                "training_seeds": [17, 29, 43], "continued_families": ["feedforward", "recurrent"],
                "lr": .001, "weight_decay": .0001, "clip_grad_norm": 1.,
                "effective_batch_groups": 8, "microbatch_groups": 2, "threads": 1, "num_workers": 0}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Task 04 continuation settings differ from declaration")
    value = {"parent_identity": parent_identity, "config_sha256": sha256(CONFIG),
             "code_sha256": {"evopolis/forecast_train.py": sha256(Path(__file__))},
             "parent_last_sha256": {f"{f}_{s}": sha256(PARENT / "weights" / f"{f}_{s}" / "last.pt") for f, s in FITS},
             "parent_best_sha256": {f"{f}_{s}": sha256(PARENT / "weights" / f"{f}_{s}" / "best.pt") for f, s in FITS},
             "base_revision": json.loads((RESULTS / "preservation.json").read_text())["base_revision"],
             "configuration": config}
    path = RESULTS / "frozen_training.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != value:
            raise ValueError("Task 04 frozen continuation identity changed")
    else:
        atomic_json(path, {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "identity": value})
    return value


def restore(family, seed, saved):
    """Restore every mutable training state, including Adam moments and shuffle."""
    if saved["family"] != family or saved["seed"] != seed:
        raise ValueError("Checkpoint family/seed mismatch")
    model, optimizer, shuffle = new_fit(family, seed)
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    torch.set_rng_state(saved["torch_rng_state"])
    shuffle.bit_generator.state = saved["shuffle_state"]
    return model, optimizer, shuffle


def parent_last(family, seed, contract):
    path = PARENT / "weights" / f"{family}_{seed}" / "last.pt"
    if sha256(path) != contract["parent_last_sha256"][f"{family}_{seed}"]:
        raise ValueError("Parent checkpoint hash changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["epoch"] != 120 or saved["metadata"] != contract["parent_identity"]:
        raise ValueError("Continuation requires exact original epoch-120 state")
    return saved


def mutable_snapshot(family, seed, model, optimizer, shuffle):
    return copy.deepcopy({"family": family, "seed": seed, "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
                          "shuffle_state": shuffle.bit_generator.state})


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError("Resume tensor mismatch")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("Resume keys mismatch")
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("Resume length mismatch")
        for aa, bb in zip(a, b):
            assert_exact(aa, bb)
    elif a != b:
        raise AssertionError("Resume scalar mismatch")


def benchmark(data, train_ids, contract):
    rows = []
    # Exactly two full training-only epochs total, one per neural architecture.
    for family in ("feedforward", "recurrent"):
        saved = parent_last(family, 17, contract)
        model, optimizer, shuffle = restore(family, 17, saved)
        start, cpu = time.monotonic(), time.process_time()
        loss = train_epoch(model, optimizer, data, train_ids, shuffle)
        seconds = time.monotonic() - start
        rows.append({"family": family, "epochs": 1, "parent_epoch": 120, "seconds": seconds,
                     "cpu_seconds": time.process_time() - cpu, "train_nll": loss})
        print(f"Disposable continuation benchmark {family}: {seconds:.3f}s", flush=True)
        del model, optimizer, shuffle, saved
    atomic_json(RESULTS / "training_benchmark.json", {
        "rows": rows, "total_training_epochs": 2, "test_and_validation_used": False,
        "state_discarded": True, "estimate_seconds_excluding_validation_and_io": sum(r["seconds"] * 1080 for r in rows),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def verify_continuation(data, train_ids, contract):
    """Two train-only optimizer updates, uninterrupted versus fresh restoration."""
    import tempfile
    rows = []
    started = time.monotonic()
    for family in ("feedforward", "recurrent"):
        saved = parent_last(family, 17, contract)
        model, optimizer, shuffle = restore(family, 17, saved)
        assert_exact(model.state_dict(), saved["model_state"])
        assert_exact(optimizer.state_dict(), saved["optimizer_state"])
        assert_exact(torch.get_rng_state(), saved["torch_rng_state"])
        assert_exact(shuffle.bit_generator.state, saved["shuffle_state"])
        first = train_epoch(model, optimizer, data, train_ids[:8], shuffle)
        middle = mutable_snapshot(family, 17, model, optimizer, shuffle)
        second = train_epoch(model, optimizer, data, train_ids[:8], shuffle)
        uninterrupted = mutable_snapshot(family, 17, model, optimizer, shuffle)
        with tempfile.TemporaryDirectory(dir=ROOT / "data/cache") as folder:
            path = Path(folder) / "resume.pt"
            atomic_checkpoint(path, middle)
            reloaded = torch.load(path, map_location="cpu", weights_only=True)
        model, optimizer, shuffle = restore(family, 17, reloaded)
        resumed_loss = train_epoch(model, optimizer, data, train_ids[:8], shuffle)
        assert_exact(uninterrupted, mutable_snapshot(family, 17, model, optimizer, shuffle))
        assert second == resumed_loss
        rows.append({"family": family, "parent_epoch": 120, "first_loss": first, "second_loss": second,
                     "resumed_loss": resumed_loss, "all_model_optimizer_rng_shuffle_values_exact": True,
                     "parent_adam_steps": sorted(set(int(state["step"]) for state in saved["optimizer_state"]["state"].values()))})
    atomic_json(RESULTS / "continuation_verification.json", {"checks": rows,
                "training_only_groups_per_update": 8, "updates_per_family": 3,
                "seconds": time.monotonic() - started, "state_discarded": True})
    print("Exact continuation check passed for both neural architectures", flush=True)


def fit(family, seed, data, train_ids, val_ids, contract):
    directory = RESULTS / "weights" / f"{family}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    parent = parent_last(family, seed, contract)
    path = directory / "last.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"] != contract:
            raise ValueError("Resume contract mismatch")
    else:
        saved = parent
        # Original selected epoch participates in all-480 selection, byte intact.
        shutil.copyfile(PARENT / "weights" / f"{family}_{seed}" / "best.pt", directory / "best.pt")
    model, optimizer, shuffle = restore(family, seed, saved)
    logs = list(saved["logs"])
    best = saved["best_validation_score"]
    elapsed = saved.get("continuation_seconds", 0.)
    cpu_elapsed = saved.get("continuation_cpu_seconds", 0.)
    start, cpu = time.monotonic(), time.process_time()
    for epoch in range(saved["epoch"] + 1, 481):
        epoch_start = time.monotonic()
        train = train_epoch(model, optimizer, data, train_ids, shuffle)
        validation = score(model, data, val_ids)
        logs.append({"epoch": epoch, "train_nll": train, "validation_nll": validation,
                     "seconds": time.monotonic() - epoch_start})
        base = {"family": family, "seed": seed, "epoch": epoch, "model_state": model.state_dict(),
                "metadata": contract, "validation_score": validation,
                "parent_last_sha256": contract["parent_last_sha256"][f"{family}_{seed}"]}
        if validation < best:
            best = validation
            atomic_checkpoint(directory / "best.pt", base)
        atomic_checkpoint(path, dict(base, optimizer_state=optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(), shuffle_state=shuffle.bit_generator.state,
            best_validation_score=best, logs=logs, initial_validation_nll=parent["initial_validation_nll"],
            continuation_seconds=elapsed + time.monotonic() - start,
            continuation_cpu_seconds=cpu_elapsed + time.process_time() - cpu))
        if epoch % 20 == 0 or epoch == 121:
            print(f"{family}/{seed} epoch {epoch}/480 train {train:.5f} val {validation:.5f} best {best:.5f}", flush=True)
    atomic_json(directory / "training.json", logs)
    selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    # Check selection against complete inherited and new curve, including ties.
    expected = min(logs, key=lambda row: (row["validation_nll"], row["epoch"]))
    if selected["epoch"] != expected["epoch"] or selected["validation_score"] != expected["validation_nll"]:
        raise AssertionError("Checkpoint not earliest global validation minimum")
    model.load_state_dict(selected["model_state"])
    reproduced = score(model, data, val_ids)
    if abs(reproduced - selected["validation_score"]) > 1e-10:
        raise AssertionError("Selected validation score failed reload")
    atomic_json(directory / "metadata.json", {
        "family": family, "seed": seed, "budget": "continued", "total_epochs": 480,
        "selected_epoch": selected["epoch"], "validation_score": selected["validation_score"],
        "reloaded_validation_score": reproduced, "best_sha256": sha256(directory / "best.pt"),
        "last_sha256": sha256(path), "parent_last_sha256": contract["parent_last_sha256"][f"{family}_{seed}"],
        "parent_best_sha256": contract["parent_best_sha256"][f"{family}_{seed}"],
        "identity": contract, "continuation_seconds": elapsed + time.monotonic() - start,
        "continuation_cpu_seconds": cpu_elapsed + time.process_time() - cpu,
        "parameter_count": sum(p.numel() for p in model.parameters())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark-train", "verify-continuation", "train"))
    args = parser.parse_args()
    if not (RESULTS / "resource_before_torch.json").exists():
        raise ValueError("Profile resources with stdlib before launching Torch")
    runtime_setup()
    contract = identity()
    arrays, manifest = load()
    data = tensor_data(arrays)
    train_ids = [g["index"] for g in manifest["groups"] if g["split"] == "train"]
    val_ids = [g["index"] for g in manifest["groups"] if g["split"] == "validation"]
    if (len(train_ids), len(val_ids)) != (96, 32):
        raise ValueError("Original split changed")
    with (ROOT / "data/cache/task03/training.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "benchmark-train":
            benchmark(data, train_ids, contract)
        elif args.command == "verify-continuation":
            verify_continuation(data, train_ids, contract)
        else:
            if not all((RESULTS / name).exists() for name in ("training_benchmark.json", "continuation_verification.json")):
                raise ValueError("Benchmark and verify continuation before main training")
            start, cpu = time.monotonic(), time.process_time()
            for family, seed in FITS:
                fit(family, seed, data, train_ids, val_ids, contract)
            preserved()
            atomic_json(RESULTS / "training_complete.json", {"fits": 6, "additional_epochs": 2160,
                "total_epochs_per_fit": 480, "wall_seconds": time.monotonic() - start,
                "cpu_seconds": time.process_time() - cpu, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
