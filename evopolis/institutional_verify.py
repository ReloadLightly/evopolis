"""Task 06 numerical gates and sequential fresh-process forecast verification.

No Experiment 2, 3 or 4 loader is called. Persistent calibrated likelihoods are
checked on Experiment 1 validation groups only. Existing scientific files are
read and hashed, never rewritten.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/task06"
VERIFY = RESULTS / "verification"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic receipt writes without importing the numerical data modules."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() == content:
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def feature_replay():
    """Cross-check actual implemented features against independent scalar math.

    The continuous-rule theoretical slope is a separate discrepancy report:
    existing BC1 rows differ from the published continuous rule, so this
    residual is not used as the implementation-identity gate.
    """
    from .institutional_data import load_synthetic
    from .institutional_models import signal_sequence

    cohort = load_synthetic()
    groups = [g for g in cohort.groups if g["cohort"] == "BC1" and g["mechanism"] == "Interpolating"]
    if len(groups) != 512:
        raise ValueError("Feature replay requires all 512 recorded BC1 Interpolating games")
    rows = []
    for group in groups:
        index = group["index"]
        offers = cohort.arrays["offers"][index].T
        actions = cohort.arrays["y"][index].T
        actual, flags, clips = signal_sequence(offers, actions)
        last, undefined = .5, True
        for t, current in enumerate(offers):
            clipped = False
            raw = theoretical = None
            if t:
                previous = actions[t - 1]
                total_e, total_c = math.fsum(current), math.fsum(previous)
                if total_e > 0 and total_c > 0:
                    b = [c / total_c - .25 for c in previous]
                    denominator = math.fsum(v * v for v in b)
                    if denominator >= 1e-9:
                        a = [e / total_e - .25 for e in current]
                        raw = math.fsum(aa * bb for aa, bb in zip(a, b)) / denominator
                        clipped = raw < -1 or raw > 2
                        last, undefined = max(-1., min(2., raw)), False
                        theoretical = 1 - (float(cohort.arrays["pool"][index, t]) / 200) ** 22
            rows.append({"key": group["key"], "round_id": t,
                         "feature_difference": float(actual[t]) - last,
                         "undefined_matches": bool(flags[t]) == undefined,
                         "clipping_matches": bool(clips[t]) == clipped,
                         "implemented_signal": float(actual[t]), "independent_signal": last,
                         "observed_raw_slope": raw, "continuous_rule_slope": theoretical,
                         "continuous_rule_slope_residual": None if raw is None else raw - theoretical,
                         "clipped": clipped})
    worst = max(abs(r["feature_difference"]) for r in rows)
    passed = (worst < 1e-9 and all(r["undefined_matches"] and r["clipping_matches"] for r in rows))
    residuals = [r["continuous_rule_slope_residual"] for r in rows if r["continuous_rule_slope_residual"] is not None]
    report = {"games": len(groups), "rounds": len(rows), "passed": passed,
              "implementation_tolerance": 1e-9, "max_abs_implementation_difference": worst,
              "clipped_rows": sum(r["clipped"] for r in rows),
              "max_abs_continuous_rule_slope_residual": max(map(abs, residuals)),
              "continuous_rule_discrepancy_status": "Reported as observed source/published-rule discrepancy; not repaired or used to alter the feature",
              "source_sha256": cohort.audit["source_sha256"],
              "code_sha256": sha256(ROOT / "evopolis/institutional_models.py"), "rows": rows}
    write_json(RESULTS / "feature_replay_verification.json", report)
    if not passed:
        raise AssertionError("Implemented institution feature differs from independent BC1-row reconstruction")
    return {key: value for key, value in report.items() if key != "rows"}


def checkpoint_selection():
    """Check earliest minima and declared full budgets for all 39 unique fits."""
    import torch
    from .institutional_rollout import registry

    checks = []
    for record in registry(include_fa=True, include_cl=False):
        path = ROOT / record["path"]
        folder = path.parent
        metadata = json.loads((folder / "metadata.json").read_text())
        logs = json.loads((folder / "training.json").read_text())
        expected_epochs = 120 if record["family"].startswith("T03-") else 480
        if [r["epoch"] for r in logs] != list(range(1, expected_epochs + 1)):
            raise AssertionError(f"Incomplete declared training budget: {record['family']}/{record['seed']}")
        earliest = min(logs, key=lambda r: (r["validation_nll"], r["epoch"]))
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if (earliest["epoch"] != record["epoch"] or earliest["epoch"] != saved["epoch"]
                or earliest["validation_nll"] != metadata["validation_score"]
                or abs(saved["validation_score"] - earliest["validation_nll"]) > 1e-12):
            raise AssertionError(f"Checkpoint is not its earliest validation minimum: {record}")
        if sha256(folder / "last.pt") != metadata["last_sha256"]:
            raise AssertionError(f"Last checkpoint hash mismatch: {folder}")
        checks.append({"family": record["family"], "seed": record["seed"],
                       "epochs": expected_epochs, "selected_epoch": earliest["epoch"],
                       "validation_nll": earliest["validation_nll"],
                       "boundary_minimum": earliest["epoch"] == expected_epochs,
                       "checkpoint_sha256": record["sha256"], "passed": True})
        del saved
    result = {"unique_fits": len(checks), "passed": len(checks) == 39, "checks": checks}
    write_json(RESULTS / "checkpoint_verification.json", result)
    if not result["passed"]:
        raise AssertionError("Expected all 39 unique reference/FA checkpoints")
    return result


def _regeneration_contract(record):
    key = f"{record['family']}_17_Mixed"
    summary = RESULTS / "rollouts" / f"{key}.json"
    trajectory = summary.with_suffix(".npz")
    saved = json.loads(summary.read_text())
    if (saved["contract"]["checkpoint"] != record or saved["contract"]["count"] != 512
            or saved["contract"]["rule"] != "Mixed" or saved["contract"]["namespace"] != 20261102):
        raise AssertionError("Fresh-process verification requires the full frozen Mixed cell")
    if sha256(trajectory) != saved["trajectory_sha256"]:
        raise AssertionError("Saved trajectory checksum differs")
    if sha256(ROOT / record["path"]) != record["sha256"]:
        raise AssertionError("Selected checkpoint checksum differs")
    for name, digest in saved["contract"]["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise AssertionError(f"Forecast code changed before regeneration: {name}")
    contract = {"family": record["family"], "seed": 17, "rule": "Mixed", "games": 512,
                "forecast_summary_sha256": sha256(summary), "trajectory_sha256": sha256(trajectory),
                "checkpoint_sha256": record["sha256"],
                "verification_code_sha256": sha256(Path(__file__)),
                "generation_contract_sha256": content_hash(saved["contract"])}
    return contract, saved, trajectory


def regenerate_one(family):
    import numpy as np
    from .institutional_rollout import load_checkpoint, registry, seeds_for, simulate

    record = next(r for r in registry(True, True) if r["family"] == family and r["seed"] == 17)
    contract, saved, trajectory = _regeneration_contract(record)
    seeds = seeds_for(family, 17, "Mixed", 512)
    if [str(s) for s in seeds] != saved["contract"]["seeds"]:
        raise AssertionError("Regenerated seed table differs")
    started, cpu = time.monotonic(), time.process_time()
    model = load_checkpoint(record)
    rows, paths, audit = simulate(model, "Mixed", seeds, tilt=saved["contract"]["tilt"], retain=4)
    if rows != saved["games"] or audit != saved["audit"]:
        raise AssertionError(f"Fresh-process game summaries/audit differ for {family}")
    with np.load(trajectory, allow_pickle=False) as archived:
        if set(paths) != set(archived.files):
            raise AssertionError("Trajectory field sets differ")
        for name, value in paths.items():
            np.testing.assert_array_equal(value, archived[name], err_msg=f"{family}: {name}")
    result = {"contract": contract, "passed": True, "pid": os.getpid(), "fresh_process": True,
              "exact_game_summaries": 512, "exact_trajectories": 4,
              "seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(VERIFY / f"regeneration_{family}.json", result)
    return result


def _regeneration_records():
    """Read the 19 seed-17 forecast identities without importing Torch/NumPy.

    The worker independently checks each identity against the live numerical
    registry. Keeping the orchestrator lean leaves memory for one worker.
    """
    config = json.loads((ROOT / "configs/task06.json").read_text())
    calibration = json.loads((RESULTS / "calibration.json").read_text())
    if not calibration["complete"] or len(calibration["selected"]) != 18:
        raise AssertionError("Complete six-family, three-seed calibration is required")
    records = []
    for family in config["reference_families"] + config["fa_families"]:
        if family.startswith("T0"):
            task, base = family.split("-")
            base = "recurrent" if base == "GRU" else base
            folder = ROOT / f"results/task{task[1:]}/weights/{base}_17"
        elif family.startswith("FA-"):
            base = family
            folder = RESULTS / "weights" / f"{family}_17"
        else:
            base = family
            folder = ROOT / "results/task05/weights" / f"{base}_17"
        path = folder / "best.pt"
        metadata = json.loads((folder / "metadata.json").read_text())
        digest = sha256(path)
        if metadata["best_sha256"] != digest:
            raise AssertionError(f"Selected checkpoint checksum differs: {path}")
        records.append({"family": family, "base_family": base, "seed": 17,
                        "path": str(path.relative_to(ROOT)), "sha256": digest,
                        "epoch": metadata["selected_epoch"], "tilt": [0., 0.]})
    for record in list(records):
        key = f"{record['family']}_17"
        if record["family"] in config["calibration_families"]:
            selected = calibration["selected"][key]
            if selected["checkpoint_sha256"] != record["sha256"]:
                raise AssertionError(f"Calibration checkpoint differs: {key}")
            records.append(dict(record, family="CL-" + record["family"], tilt=selected["tilt"]))
    return records


def regenerate():
    """Each family runs in a separate new CPU process, one at a time."""
    VERIFY.mkdir(parents=True, exist_ok=True)
    records = _regeneration_records()
    if len(records) != 19:
        raise AssertionError("Fresh-process regeneration requires all 19 forecast families")
    checks = []
    started = time.monotonic()
    for record in records:
        family = record["family"]
        contract, _, _ = _regeneration_contract(record)
        receipt = VERIFY / f"regeneration_{family}.json"
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if result["contract"] != contract or not result["passed"]:
                raise AssertionError(f"Regeneration receipt contract changed: {family}")
        else:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            command = [sys.executable, "-m", "evopolis.institutional_verify", "regenerate-one", "--family", family]
            process = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if process.returncode:
                failure = {"family": family, "command": command, "returncode": process.returncode,
                           "stdout_tail": process.stdout[-4000:], "stderr_tail": process.stderr[-4000:],
                           "completed_families": [r["contract"]["family"] for r in checks]}
                write_json(VERIFY / "regeneration_failure.json", failure)
                raise RuntimeError(f"Fresh-process regeneration failed for {family}: {process.stderr[-2000:]}")
            result = json.loads(receipt.read_text())
            if result["contract"] != contract or not result["passed"] or result["pid"] == os.getpid():
                raise AssertionError("Invalid fresh-process worker receipt")
        checks.append(result)
        projected = sum(r["seconds"] for r in checks) / len(checks) * len(records)
        write_json(RESULTS / "fresh_process_verification.json", {
            "checks": checks, "families": len(checks), "complete": len(checks) == 19,
            "passed": len(checks) == 19 and all(r["passed"] for r in checks),
            "seconds_this_invocation": time.monotonic() - started,
            "projected_seconds": projected,
            "peak_worker_rss_kib": max(r["peak_rss_kib"] for r in checks)})
        print(f"Fresh process {family}: 512 summaries and four trajectories exact", flush=True)
        if projected > 28800:
            raise RuntimeError("Fresh-process verification projects beyond eight hours; completed receipts saved")
    return {"families": len(checks), "passed": True}


def tilted_integration():
    """Check selected calibrated persistent fits on all 32 validation groups."""
    import numpy as np
    from .behavior_data import load
    from .institutional_predict import prequential
    from .institutional_rollout import load_checkpoint, registry

    config = json.loads((ROOT / "configs/task06.json").read_text())
    arrays, manifest = load()
    groups = [g for g in manifest["groups"] if g["split"] == "validation"]
    records = [r for r in registry(True, True) if r["family"] in ("CL-H0", "CL-FA-H0")]
    if len(groups) != 32 or len(records) != 6:
        raise AssertionError("Expected all six calibrated H0 fits and all 32 validation groups")
    VERIFY.mkdir(parents=True, exist_ok=True)
    checks = []
    for record in records:
        key = f"{record['family']}_{record['seed']}"
        contract = {"record": record, "nodes_compared": [41, 81],
                    "split_sha256": sha256(ROOT / "results/task03/split.json"),
                    "source_sha256": manifest["source_sha256"],
                    "code_sha256": {name: sha256(ROOT / name) for name in
                        ("evopolis/institutional_verify.py", "evopolis/institutional_predict.py",
                         "evopolis/institutional_models.py", "evopolis/conditional_models.py")}}
        path = VERIFY / f"tilted_integration_{key}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["contract"] != contract:
                raise AssertionError("Calibrated integration receipt contract changed")
        else:
            started = time.monotonic()
            model = load_checkpoint(record)
            maxima = {name: 0. for name in config["quadrature_tolerances"]}
            rows = []
            for group in groups:
                index = group["index"]
                offers = arrays["offers"][index].T
                mask = offers >= 1
                low = prequential(model, arrays, index, tilt=record["tilt"], nodes=41)
                high = prequential(model, arrays, index, tilt=record["tilt"], nodes=81)
                delta = low["log_prob"] - high["log_prob"]
                pmf_delta = low["pmfs"] - high["pmfs"]
                fraction = np.arange(201)[None, None, :] / np.maximum(offers[..., None], 1)
                metrics = {"group_nll": float(abs(delta.sum()) / mask.sum()),
                           "resident_nll": float(np.max(abs(delta.sum(0)) / np.maximum(mask.sum(0), 1))),
                           "prediction_fraction": float(abs((pmf_delta * fraction).sum(-1))[mask].max()),
                           "tail_probability": float(abs((pmf_delta * (fraction >= .8)).sum(-1))[mask].max()),
                           "posterior_normalization": max(float(abs(v["posterior_weights"].sum(-1) - 1).max()) for v in (low, high))}
                rows.append(dict(metrics, key=group["key"]))
                for name, value in metrics.items():
                    maxima[name] = max(maxima[name], value)
            result = {"contract": contract, "maxima": maxima, "groups": rows,
                      "passed": all(maxima[name] < bound for name, bound in config["quadrature_tolerances"].items()),
                      "seconds": time.monotonic() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            write_json(path, result)
        checks.append(result)
        complete = len(checks) == 6
        projected = sum(r["seconds"] for r in checks) / len(checks) * len(records)
        write_json(RESULTS / "tilted_integration_verification.json", {
            "scope": "All 32 Exp1 validation groups; CL-H0 and CL-FA-H0, all three seeds",
            "checks": checks, "complete": complete, "passed": complete and all(r["passed"] for r in checks),
            "projected_seconds": projected, "tolerances": config["quadrature_tolerances"]})
        print(f"Calibrated integration {key}: {result['maxima']}; passed={result['passed']}", flush=True)
        if not result["passed"]:
            raise RuntimeError(f"Calibrated persistent likelihood fails41-versus81 accuracy gate: {key}; method unchanged")
        if projected > config["phase_limit_seconds"]:
            raise RuntimeError("Calibrated integration projects beyond eight hours; completed receipts saved")
    return {"fits": 6, "groups_per_fit": 32, "passed": True}


def protected():
    preservation = json.loads((RESULTS / "preservation.json").read_text())
    # These are direct Task06-authorized edits. Older scientific outputs,
    # configurations and source modules receive no exception.
    authorized = {"README.md", "docs/github-description.txt", "docs/tasks/06-frozen-human-transfer.md"}
    hashes = preservation["files"]
    checked = {name: digest for name, digest in hashes.items() if name not in authorized}
    changed = [name for name, digest in checked.items() if not (ROOT / name).exists() or sha256(ROOT / name) != digest]
    result = {"base_revision": preservation["base_revision"], "checked_files": len(checked),
              "authorized_exclusions_present_in_baseline": sorted(set(hashes) & authorized),
              "changed": changed, "passed": not changed, "utc": _utc()}
    write_json(RESULTS / "preservation_verification.json", result)
    if changed:
        raise AssertionError(f"Protected Task01-05 artifacts changed: {changed}")
    return result


def run_all():
    """Release each numerical phase's memory before starting the next one."""
    phases = ("feature-replay", "checkpoints", "tilted-integration", "regenerate", "protected")
    completed = []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for phase in phases:
        command = [sys.executable, "-m", "evopolis.institutional_verify", phase]
        # Inherit output so long numerical phases remain observable while this
        # standard-library-only parent waits without retaining numerical state.
        process = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if process.returncode:
            write_json(VERIFY / "all_failure.json", {"failed_phase": phase, "returncode": process.returncode,
                       "completed_phases": completed, "command": command, "utc": _utc()})
            raise RuntimeError(f"Verification phase {phase} failed with exit code {process.returncode}; child receipts retained")
        completed.append(phase)
        write_json(RESULTS / "verification_complete.json", {
            "completed_phases": completed, "complete": len(completed) == len(phases),
            "passed": len(completed) == len(phases), "sequential_fresh_phase_processes": True,
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "utc": _utc()})
    return {"passed": True, "phases": len(completed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("feature-replay", "checkpoints", "regenerate", "regenerate-one",
                                             "tilted-integration", "protected", "all"))
    parser.add_argument("--family")
    args = parser.parse_args()
    if args.command == "regenerate-one" and not args.family:
        parser.error("regenerate-one requires --family")
    VERIFY.mkdir(parents=True, exist_ok=True)
    functions = {"feature-replay": feature_replay, "checkpoints": checkpoint_selection,
                 "regenerate": regenerate, "regenerate-one": lambda: regenerate_one(args.family),
                 "tilted-integration": tilted_integration, "protected": protected, "all": run_all}
    command = args.command
    started, cpu = time.monotonic(), time.process_time()
    status, error_text = "complete", None
    try:
        if command in ("feature-replay", "checkpoints", "tilted-integration", "regenerate-one"):
            from .behavior_train import runtime_setup
            runtime_setup()
        result = functions[command]()
        print(json.dumps({"command": command, "passed": result.get("passed", True)}), flush=True)
    except BaseException as error:
        status, error_text = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        # Workers report their RSS in their own exact-regeneration receipt;
        # only the parent merges the phase resource file. The all orchestrator
        # adds its own receipt without replacing any child-phase records.
        if command != "regenerate-one":
            path = RESULTS / "runtime.json"
            resources = json.loads(path.read_text()) if path.exists() else {}
            resources[f"verification_{command}"] = {
                "status": status, "error": error_text, "wall_seconds": time.monotonic() - started,
                "cpu_seconds": time.process_time() - cpu,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "utc": _utc()}
            write_json(path, resources)


if __name__ == "__main__":
    main()
