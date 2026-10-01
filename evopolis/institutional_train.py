"""Task 06's nine fresh fixed-budget fits, with exact atomic continuation."""

import argparse
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from .behavior_data import ROOT, load
from .behavior_train import (atomic_checkpoint, atomic_json, runtime_setup,
                             score as neural_score, tensor_data,
                             train_epoch as neural_train_epoch)
from .conditional_train import score as conditional_score, train_epoch as conditional_train_epoch
from .institutional_models import (FAMILIES, augment_arrays, make_model,
                                   prepare_fa_group, prequential_fa)
from .sources import sha256

RESULTS = ROOT / "results/task06"
CONFIG = ROOT / "configs/task06.json"
SEEDS = (17, 29, 43)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record_runtime(phase, value):
    path = RESULTS / "runtime.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    result[phase] = value
    atomic_json(path, result)


def identity():
    config = json.loads(CONFIG.read_text())
    expected = {"epochs": 480, "optimizer": "Adam", "lr": .001,
                "weight_decay": .0001, "clip_grad_norm": 1.,
                "effective_batch_groups": 8, "gru_microbatch_groups": 2,
                "conditional_microbatch_groups": 1,
                "training_groups": 96, "validation_groups": 32}
    if any(config["training"][key] != value for key, value in expected.items()):
        raise ValueError("Task 06 declared optimizer or cohort settings changed")
    if (config["fa_families"] != list(FAMILIES) or config["training_seeds"] != list(SEEDS)
            or config["quadrature_nodes"] != 41 or config["quadrature_check_nodes"] != 81
            or config["threads"] != 1 or config["num_workers"] != 0):
        raise ValueError("Task 06 fixed fit design changed")
    paths = ("evopolis/institutional_models.py", "evopolis/institutional_train.py",
             "evopolis/behavior_models.py", "evopolis/behavior_train.py",
             "evopolis/behavior_data.py", "evopolis/conditional_models.py",
             "evopolis/conditional_train.py")
    value = {"configuration": config, "config_sha256": sha256(CONFIG),
             "split_sha256": sha256(ROOT / "results/task03/split.json"),
             "source_sha256": json.loads((ROOT / "results/task03/split.json").read_text())["source_sha256"],
             "code_sha256": {p: sha256(ROOT / p) for p in paths},
             "software": {"torch": str(torch.__version__), "numpy": np.__version__}}
    path = RESULTS / "frozen_training.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != value:
            raise ValueError("Task 06 training identity differs; refusing incompatible resume")
    else:
        atomic_json(path, {"utc": utc(), "identity": value})
    return value


def new_fit(family, seed):
    torch.manual_seed(seed)
    model = make_model(family)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=.001, weight_decay=.0001)
    return model, optimizer, np.random.Generator(np.random.PCG64(seed))


def snapshot(model, optimizer, shuffle):
    return copy.deepcopy({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                          "torch_rng_state": torch.get_rng_state(), "shuffle_state": shuffle.bit_generator.state})


def restore(family, seed, saved):
    if saved["family"] != family or saved["seed"] != seed:
        raise ValueError("Checkpoint family/seed mismatch")
    model, optimizer, shuffle = new_fit(family, seed)
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    torch.set_rng_state(saved["torch_rng_state"])
    shuffle.bit_generator.state = saved["shuffle_state"]
    return model, optimizer, shuffle


def score(model, data, indices):
    if model.fa_family == "FA-GRU":
        return neural_score(model, data, indices, microbatch=2)
    return conditional_score(model, data, indices, nodes=41)


def train_epoch(model, optimizer, data, indices, shuffle):
    if model.fa_family == "FA-GRU":
        return neural_train_epoch(model, optimizer, data, indices, shuffle, microbatch=2)
    return conditional_train_epoch(model, optimizer, data, indices, shuffle, nodes=41)


def throughput_projection():
    """Actual retained epochs; archived measured timing for unstarted families."""
    historical = {"FA-GRU": ROOT / "results/task04/weights/recurrent_17/training.json",
                  "FA-P0": ROOT / "results/task05/weights/P0_17/training.json",
                  "FA-H0": ROOT / "results/task05/weights/H0_17/training.json"}
    details = []
    for family in FAMILIES:
        measured = []
        for seed in SEEDS:
            path = RESULTS / "weights" / f"{family}_{seed}" / "training.json"
            if path.exists():
                measured.extend(row["seconds"] for row in json.loads(path.read_text()))
        current_measurement = bool(measured)
        source = "Task06 retained training epochs"
        if not measured:
            measured = [row["seconds"] for row in json.loads(historical[family].read_text())]
            source = str(historical[family].relative_to(ROOT))
        seconds_per_epoch = float(np.mean(measured))
        details.append({"family": family, "seconds_per_epoch": seconds_per_epoch,
                        "timing_source": source, "current_measurement": current_measurement,
                        "projected_seconds": seconds_per_epoch * 480 * 3})
    return {"utc": utc(), "families": details, "projected_phase_seconds": sum(v["projected_seconds"] for v in details),
            "projected_measured_components_seconds": sum(v["projected_seconds"] for v in details if v["current_measurement"]),
            "all_families_measured": all(v["current_measurement"] for v in details),
            "limit_seconds": 28800, "retained_training_not_pilot": True}


def fit(family, seed, data, train_ids, val_ids, contract):
    folder = RESULTS / "weights" / f"{family}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    model, optimizer, shuffle = new_fit(family, seed)
    logs, first, best, selected, elapsed, cpu_elapsed = [], 1, float("inf"), 0, 0., 0.
    initial = score(model, data, val_ids)
    if (folder / "last.pt").exists():
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        if saved["metadata"] != contract:
            raise ValueError("Resume identity changed")
        model, optimizer, shuffle = restore(family, seed, saved)
        logs, first, best = saved["logs"], saved["epoch"] + 1, saved["best_validation_score"]
        selected, elapsed, cpu_elapsed = saved["selected_epoch"], saved["elapsed_seconds"], saved["cpu_seconds"]
        initial = saved["initial_validation_nll"]
        print(f"Resume {family}/{seed} at epoch {first}", flush=True)
    started, cpu = time.monotonic(), time.process_time()
    for epoch in range(first, 481):
        tick = time.monotonic()
        training = train_epoch(model, optimizer, data, train_ids, shuffle)
        validation = score(model, data, val_ids)
        row = {"epoch": epoch, "train_nll": training, "validation_nll": validation,
               "seconds": time.monotonic() - tick}
        if family != "FA-GRU":
            row.update(eta=float(model.eta.detach()), sigma=float(model.sigma.detach()))
        logs.append(row)
        improved = validation < best
        if improved:
            best, selected = validation, epoch
        saved = dict(snapshot(model, optimizer, shuffle), family=family, seed=seed,
                     epoch=epoch, metadata=contract, integration_nodes=41,
                     validation_score=validation, best_validation_score=best, selected_epoch=selected,
                     logs=logs, initial_validation_nll=initial,
                     elapsed_seconds=elapsed + time.monotonic() - started,
                     cpu_seconds=cpu_elapsed + time.process_time() - cpu)
        if improved:
            atomic_checkpoint(folder / "best.pt", saved)
        atomic_checkpoint(folder / "last.pt", saved)
        atomic_json(folder / "training.json", logs)
        if epoch == 1 or epoch % 10 == 0:
            projection = throughput_projection()
            atomic_json(RESULTS / "training_throughput.json", projection)
            print(f"{family}/{seed} epoch {epoch}/480 train {training:.6f} val {validation:.6f} "
                  f"best {best:.6f}; epoch {row['seconds']:.2f}s; phase estimate "
                  f"{projection['projected_phase_seconds']/3600:.2f}h", flush=True)
            # Archived timings are planning context only. A stop requires the
            # measured Task06 components alone to exceed the phase limit.
            if projection["projected_measured_components_seconds"] > contract["configuration"]["phase_limit_seconds"]:
                atomic_json(RESULTS / "training_blocker.json", dict(projection, family=family, seed=seed,
                            saved_epoch=epoch, blocker="Measured phase projection exceeds declared eight-hour limit"))
                raise RuntimeError("Phase 2(a) projected beyond eight hours; all completed epoch state saved")
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"])
    reloaded = score(model, data, val_ids)
    if abs(reloaded - best) >= 1e-12 or selected != min(logs, key=lambda v: (v["validation_nll"], v["epoch"]))["epoch"]:
        raise AssertionError("Selected checkpoint does not reproduce earliest validation minimum")
    metadata = {"family": family, "seed": seed, "selected_epoch": selected,
                "boundary_minimum": selected == 480, "validation_score": best,
                "reloaded_validation_score": reloaded, "initial_validation_nll": initial,
                "epochs": 480, "integration_nodes": 41,
                "best_sha256": sha256(folder / "best.pt"), "last_sha256": sha256(folder / "last.pt"),
                "elapsed_seconds": elapsed + time.monotonic() - started,
                "cpu_seconds": cpu_elapsed + time.process_time() - cpu,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if family != "FA-GRU":
        metadata.update(eta=float(model.eta.detach()), sigma=float(model.sigma.detach()))
    atomic_json(folder / "metadata.json", metadata)
    return metadata


def load_model(family, seed):
    path = RESULTS / "weights" / f"{family}_{seed}" / "best.pt"
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved["family"] != family or saved["seed"] != seed:
        raise ValueError("Checkpoint identity mismatch")
    model = make_model(family)
    model.load_state_dict(saved["model_state"])
    return model.eval(), saved


def verify_integration():
    """All 128 development groups; same five numerical gates as Task 05."""
    arrays, manifest = load()
    groups = [g for g in manifest["groups"] if g["split"] in ("train", "validation")]
    config = json.loads(CONFIG.read_text())
    started, cpu = time.monotonic(), time.process_time()
    checks, rows = [], []
    for seed in SEEDS:
        model, _ = load_model("FA-H0", seed)
        worst = {key: 0. for key in config["quadrature_tolerances"]}
        for group in groups:
            index = group["index"]
            offers = arrays["offers"][index].T
            mask = offers >= 1
            low = prequential_fa(model, arrays, index, nodes=41)
            high = prequential_fa(model, arrays, index, nodes=81)
            delta = low["log_prob"] - high["log_prob"]
            fraction = np.arange(201)[None, None, :] / np.maximum(offers[..., None], 1)
            pmf_delta = low["pmfs"] - high["pmfs"]
            result = {"group_nll": float(abs(delta.sum()) / mask.sum()),
                      "resident_nll": float(np.max(abs(delta.sum(0)) / np.maximum(mask.sum(0), 1))),
                      "prediction_fraction": float(abs((pmf_delta * fraction).sum(-1))[mask].max()),
                      "tail_probability": float(abs((pmf_delta * (fraction >= .8)).sum(-1))[mask].max()),
                      "posterior_normalization": max(float(abs(r["posterior_weights"].sum(-1) - 1).max()) for r in (low, high))}
            rows.append(dict(result, family="FA-H0", seed=seed, key=group["key"], split=group["split"]))
            for key in worst:
                worst[key] = max(worst[key], result[key])
        passed = all(worst[key] < threshold for key, threshold in config["quadrature_tolerances"].items())
        checks.append({"family": "FA-H0", "seed": seed, "nodes_compared": [41, 81],
                       "checkpoint_sha256": sha256(RESULTS / "weights" / f"FA-H0_{seed}" / "best.pt"),
                       "maxima": worst, "passed": passed})
        print(f"FA-H0/{seed} 41/81 integration: {worst}; passed={passed}", flush=True)
        # Persist each completed fit so a failed numerical gate remains inspectable.
        atomic_json(RESULTS / "integration_checks.json", {"scope": "all 128 Exp1 training/validation groups",
                    "checks": checks, "group_checks": rows, "passed": len(checks) == 3 and all(c["passed"] for c in checks),
                    "complete": len(checks) == 3, "tolerances": config["quadrature_tolerances"]})
    receipt = {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "utc": utc()}
    record_runtime("phase2a_integration", receipt)
    if not all(c["passed"] for c in checks):
        raise RuntimeError("Task 06 FA-H0 41-versus-81 numerical accuracy gate failed")


def train(selected_family=None, selected_seed=None):
    contract = identity()
    arrays, manifest = load()
    ids = {split: [g["index"] for g in manifest["groups"] if g["split"] == split]
           for split in ("train", "validation")}
    if (len(ids["train"]), len(ids["validation"])) != (96, 32):
        raise ValueError("Unexpected training/validation cohort")
    # Features for fit preparation are computed only for development groups.
    development = ids["train"] + ids["validation"]
    reduced = {k: v[development] for k, v in arrays.items()}
    augmented = augment_arrays(reduced)
    remap = {original: local for local, original in enumerate(development)}
    local_ids = {s: [remap[i] for i in values] for s, values in ids.items()}
    neural_data = tensor_data(augmented)
    conditional_data = {i: prepare_fa_group(reduced, i) for i in range(len(development))}
    atomic_json(RESULTS / "feature_development_audit.json", {
        "groups": 128, "rounds": int(augmented["institution_signal"].size),
        "clipped_rounds": int(augmented["institution_clipping"].sum()),
        "undefined_rounds": int(augmented["institution_undefined"].sum()),
        "by_rule": {rule: {"rounds": 40 * sum(manifest["groups"][i]["mechanism"] == rule for i in development),
                           "clipped": sum(int(augmented["institution_clipping"][remap[i]].sum()) for i in development if manifest["groups"][i]["mechanism"] == rule)}
                    for rule in ("Equal", "Mixed", "Proportional", "M1")}})
    started, cpu = time.monotonic(), time.process_time()
    try:
        with (RESULTS / "training.lock").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for family in ([selected_family] if selected_family else FAMILIES):
                for seed in ([selected_seed] if selected_seed else SEEDS):
                    data = neural_data if family == "FA-GRU" else conditional_data
                    fit(family, seed, data, local_ids["train"], local_ids["validation"], contract)
    finally:
        record_runtime("phase2a_training", {"invocation_wall_seconds": time.monotonic() - started,
                       "invocation_cpu_seconds": time.process_time() - cpu,
                       "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "utc": utc()})
    metadata = []
    for family in FAMILIES:
        for seed in SEEDS:
            path = RESULTS / "weights" / f"{family}_{seed}" / "metadata.json"
            if path.exists():
                metadata.append(json.loads(path.read_text()))
    if len(metadata) == 9:
        means = {f: float(np.mean([r["validation_score"] for r in metadata if r["family"] == f])) for f in FAMILIES}
        best = min(FAMILIES, key=lambda f: means[f])
        receipt = {"fits": 9, "epochs_per_fit": 480, "mean_validation_nll": means,
                   "fa_best": best, "boundary_minima": [f"{r['family']}_{r['seed']}" for r in metadata if r["boundary_minimum"]],
                   "elapsed_seconds": sum(r["elapsed_seconds"] for r in metadata),
                   "cpu_seconds": sum(r["cpu_seconds"] for r in metadata),
                   "peak_rss_kib": max(r["peak_rss_kib"] for r in metadata), "utc": utc()}
        atomic_json(RESULTS / "training_complete.json", receipt)
        record_runtime("phase2a_training", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "verify-integration"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", choices=SEEDS, type=int)
    arguments = parser.parse_args()
    if not (RESULTS / "resource_current_before_torch.json").exists():
        raise RuntimeError("Task 06 resource profile must precede any Torch work")
    started, cpu = time.monotonic(), time.process_time()
    phase = "phase2a_training" if arguments.command == "train" else "phase2a_integration"
    status, error_text = "complete", None
    try:
        runtime_setup()
        if arguments.command == "train":
            train(arguments.family, arguments.seed)
        else:
            verify_integration()
    except BaseException as error:
        status, error_text = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        path = RESULTS / "runtime.json"
        previous = json.loads(path.read_text()).get(phase, {}) if path.exists() else {}
        record_runtime(phase, dict(previous, status=status, error=error_text,
                       invocation_wall_seconds=time.monotonic() - started,
                       invocation_cpu_seconds=time.process_time() - cpu,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, utc=utc()))


if __name__ == "__main__":
    main()
