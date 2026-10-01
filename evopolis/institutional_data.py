"""Typed, evidence-gated Task 06 cohorts without importing torch.

The condition label is checked before any excluded row's numerical fields are
parsed. Experiment 2 additionally requires a remotely verified freeze receipt;
Experiments 3 and 4 have no parser in this module. Recorded numerical residuals
are retained, never repaired. Full condition/launch/episode keys identify games.
"""

import argparse
from dataclasses import dataclass
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import numpy as np

from .behavior_data import ROOT, FIELDS, content_hash, load, write_json
from .sources import sha256
from .world import CAPACITY, PLAYERS, ROUNDS, allocate, participant_observation, step


RESULTS = ROOT / "results/task06"
SOURCE = ROOT / "data/raw/sustainable_behavior.csv"
FROZEN_REQUIRED_FILES = {
    "configs/task06.json": "config_sha256",
    "docs/tasks/06-institutional-validity.md": "protocol_sha256",
    "results/task03/split.json": "split_sha256",
    "results/task06/recorded_games.json": "recorded_forecasts_sha256",
    "results/task06/institutional_source_audit.json": "known_source_discrepancies_sha256",
    "results/task06/d1_human_order_constraints.json": "validation_constraints_sha256",
}
SYNTHETIC_LABELS = {
    "Equal Baseline BC 1": ("BC1", "Equal"),
    "Mixed Baseline BC 1": ("BC1", "Mixed"),
    "Proportional Baseline BC 1": ("BC1", "Proportional"),
    "Interpolating Baseline BC 1": ("BC1", "Interpolating"),
    "RL Agent (M1) BC 1": ("BC1", "M1"),
    "Interpolating Baseline BC 2": ("BC2", "Interpolating"),
    "RL Agent (M2) BC 2": ("BC2", "M2"),
}
EXP2_LABELS = {
    "Proportional Baseline Exp 2": ("Exp2", "Proportional"),
    "Interpolating Baseline Exp 2": ("Exp2", "Interpolating"),
    "RL Agent (M1) Exp 2": ("Exp2", "M1"),
}


@dataclass(frozen=True)
class Cohort:
    name: str
    arrays: dict[str, np.ndarray]
    groups: list[dict]
    audit: dict


def load_exp1() -> Cohort:
    arrays, manifest = load()
    groups = [{**row, "cohort": "Exp1", "pool40_source": "recorded next pool"} for row in manifest["groups"]]
    return Cohort("Exp1", arrays, groups,
                  {"source_sha256": manifest["source_sha256"], "split_hash": manifest["manifest_hash"],
                   "groups": len(groups), "behavioral_rows": len(groups) * ROUNDS,
                   "opened_test_groups": True, "residual_floor": "preserved as recorded"})


def _source_check(source: Path) -> str:
    manifest = json.loads((ROOT / "results/task01/manifest.json").read_text())
    expected = next(row["sha256"] for row in manifest["sources"] if row["file"] == source.name)
    digest = sha256(source)
    if digest != expected:
        raise ValueError("Pinned institutional source checksum mismatch")
    return digest


def _parse_selected(source: Path, labels: dict, *, human: bool, name: str,
                    expected_per_condition: int | None = None) -> Cohort:
    """Allocate compact per-game arrays; never retain all CSV string rows."""
    if labels not in (SYNTHETIC_LABELS, EXP2_LABELS):
        raise ValueError("Only the declared synthetic or gated Experiment 2 cohorts are supported")
    records = {}
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {field: header.index(field) for field in FIELDS}
        label_position = positions["mech_name_by_player"]
        for csv_row, row in enumerate(reader, start=2):
            label = row[label_position]
            if label not in labels:
                continue
            # No number from an excluded label is interpreted above this line.
            key = (label, row[positions["launch_id"]], row[positions["episode_id"]])
            round_id = int(row[positions["round_id"]])
            if not 0 <= round_id < ROUNDS:
                raise ValueError(f"Round index outside 0..39: {key}, {round_id}")
            if key not in records:
                records[key] = {"pool": np.empty(ROUNDS), "next_pool": np.empty(ROUNDS),
                                "offers": np.empty((PLAYERS, ROUNDS)), "y": np.empty((PLAYERS, ROUNDS)),
                                "surplus": np.empty((PLAYERS, ROUNDS)),
                                "csv_rows": np.zeros(ROUNDS, dtype=np.int64),
                                "next_pool_recorded": np.zeros(ROUNDS, dtype=bool)}
            record = records[key]
            if record["csv_rows"][round_id]:
                raise ValueError(f"Duplicate full game/round key: {key}, {round_id}")
            pool = float(row[positions["mechanism_observation.pool"]])
            offers = tuple(float(row[positions[f"offer_{i}"]]) for i in range(PLAYERS))
            contributions = tuple(float(row[positions[f"player_action_{i}"]]) for i in range(PLAYERS))
            surplus = tuple(float(row[positions[f"player_reward_{i}"]]) for i in range(PLAYERS))
            if not all(math.isfinite(v) for v in (pool, *offers, *contributions, *surplus)):
                raise ValueError(f"Nonfinite outcome in {key}, round {round_id}")
            # Source replay tolerance is validation allowance, not rounding or repair.
            step(pool, offers, contributions, integer_contributions=human, tolerance=1e-4)
            if human and any(c != math.floor(c) or not 0 <= c <= math.floor(e)
                             for c, e in zip(contributions, offers)):
                raise ValueError(f"Illegal integer target in {key}, round {round_id}")
            recorded_next = row[positions["next_environment_state.pool"]].strip()
            if recorded_next and recorded_next.lower() not in {"nan", "none", "null"}:
                next_pool = float(recorded_next)
                if not math.isfinite(next_pool):
                    raise ValueError(f"Nonfinite next pool in {key}, round {round_id}")
                record["next_pool"][round_id] = next_pool
                record["next_pool_recorded"][round_id] = True
            elif human:
                raise ValueError(f"Human recorded next pool missing in {key}, round {round_id}")
            record["pool"][round_id] = pool
            record["offers"][:, round_id] = offers
            record["y"][:, round_id] = contributions
            record["surplus"][:, round_id] = surplus
            record["csv_rows"][round_id] = csv_row

    groups, assembled = [], []
    adjacent_residuals, inferred_final = [], []
    for index, key in enumerate(sorted(records)):
        record = records[key]
        if not np.all(record["csv_rows"]):
            raise ValueError(f"Incomplete recorded horizon for {key}")
        for t in range(ROUNDS):
            if not record["next_pool_recorded"][t]:
                if t < ROUNDS - 1:
                    record["next_pool"][t] = record["pool"][t + 1]
                else:
                    record["next_pool"][t] = step(record["pool"][t], record["offers"][:, t],
                                                   record["y"][:, t], tolerance=1e-4).next_pool
                    inferred_final.append(float(record["next_pool"][t]))
            if t < ROUNDS - 1:
                adjacent_residuals.append(float(record["next_pool"][t] - record["pool"][t + 1]))
        if human and any(v != 0 for v in adjacent_residuals[-39:]):
            raise ValueError(f"Within-row and adjacent recorded pools differ in {key}")
        record["n"] = np.floor(record["offers"]).astype(np.int64)
        if human:
            record["y"] = record["y"].astype(np.int64)
            x = np.empty((PLAYERS, ROUNDS, 9), dtype=np.float64)
            for t in range(ROUNDS):
                previous = None if t == 0 else record["y"][:, t - 1]
                for p in range(PLAYERS):
                    x[p, t] = np.asarray(participant_observation(p, record["pool"][t], record["offers"][:, t], previous)) / CAPACITY
            record["x"] = x
        cohort, mechanism = labels[key[0]]
        groups.append({"index": index, "key": list(key), "condition": key[0], "launch_id": key[1],
                       "episode_id": key[2], "mechanism": mechanism, "cohort": cohort,
                       "nonforced_choices": int((record["n"] >= 1).sum()),
                       "pool40_source": "recorded next pool" if record["next_pool_recorded"][-1]
                       else "published-equation inferred; final next pool absent from release"})
        assembled.append(record)
    if not assembled:
        raise ValueError(f"No selected records found for {name}")
    arrays = {field: np.stack([record[field] for record in assembled]) for field in assembled[0]}
    counts = {condition: sum(group["condition"] == condition for group in groups) for condition in labels}
    if expected_per_condition is not None and any(n != expected_per_condition for n in counts.values()):
        raise ValueError(f"Unexpected {name} condition counts: {counts}")
    return Cohort(name, arrays, groups,
                  {"groups": len(groups), "behavioral_rows": len(groups) * ROUNDS, "condition_counts": counts,
                   "duplicate_rounds": 0, "incomplete_games": 0, "illegal_actions": 0,
                   "integer_targets_required": human, "pool40_inferred_games": len(inferred_final),
                   "pool40_inferred_negative_count": sum(value < 0 for value in inferred_final),
                   "pool40_inference_limitation": "Synthetic terminal pool is not recorded. Published accounting is used only for pool40 and final survival; no residual floor is invented. Earlier recorded pools and all recorded rewards remain unchanged.",
                   "recorded_human_adjacent_max_residual": max(map(abs, adjacent_residuals), default=0) if human else None})


def load_synthetic(source: Path = SOURCE) -> Cohort:
    digest = _source_check(source)
    cohort = _parse_selected(source, SYNTHETIC_LABELS, human=False, name="Recorded BC1/BC2", expected_per_condition=512)
    cohort.audit["source_sha256"] = digest
    return cohort


def _verify_frozen_artifacts(root: Path, manifest: dict) -> dict:
    """Validate every frozen forecast input before permitting transfer access."""
    def require_hash(name, expected):
        path = root / name
        if not isinstance(expected, str) or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Experiment 2 is closed: frozen artifact missing or changed: {name}")

    summaries = manifest.get("forecast_summary_sha256")
    checkpoints = manifest.get("checkpoints")
    cells = manifest.get("seed_table")
    if not isinstance(summaries, dict) or not summaries or not isinstance(checkpoints, dict) or not checkpoints:
        raise RuntimeError("Experiment 2 is closed: frozen forecast/checkpoint hash maps are absent")
    if not isinstance(cells, list) or len(cells) != len(summaries):
        raise RuntimeError("Experiment 2 is closed: frozen seed table is incomplete")
    # Hash summaries before parsing them; failures cannot reach the human CSV.
    for name, expected in summaries.items():
        require_hash(name, expected)
    for name, record in checkpoints.items():
        if name != f"{record['family']}_{record['seed']}":
            raise RuntimeError("Experiment 2 is closed: inconsistent frozen checkpoint identity")
        require_hash(record["path"], record["sha256"])
    files, identities, used_checkpoints = set(), set(), set()
    for cell in cells:
        name = cell["file"]
        key = f"{cell['family']}_{cell['training_seed']}"
        identity = (cell["family"], cell["training_seed"], cell["rule"])
        if name in files or identity in identities or summaries.get(name) != cell["sha256"] or key not in checkpoints:
            raise RuntimeError("Experiment 2 is closed: duplicate or inconsistent frozen forecast cell")
        files.add(name)
        identities.add(identity)
        used_checkpoints.add(key)
        require_hash(cell["trajectory_file"], cell["trajectory_sha256"])
        saved = json.loads((root / name).read_text())
        contract = saved["contract"]
        if (contract["checkpoint"] != checkpoints[key] or contract["rule"] != cell["rule"]
                or contract["count"] != cell["games"] or contract["seeds"] != cell["seeds"]
                or saved["trajectory_sha256"] != cell["trajectory_sha256"]
                or len(saved["games"]) != cell["games"]
                or [r["rollout_seed"] for r in saved["games"]] != cell["seeds"]):
            raise RuntimeError(f"Experiment 2 is closed: frozen cell contract differs: {name}")
    if files != set(summaries) or used_checkpoints != set(checkpoints):
        raise RuntimeError("Experiment 2 is closed: frozen forecast tables have different coverage")
    for name, key in FROZEN_REQUIRED_FILES.items():
        require_hash(name, manifest.get(key))
    evidence = manifest.get("required_evidence_sha256")
    if not isinstance(evidence, dict) or not evidence:
        raise RuntimeError("Experiment 2 is closed: required prefreeze evidence hashes are absent")
    for name, expected in evidence.items():
        require_hash(name, expected)
    # Downstream scoring reads these small selection records. Their scientific
    # contents must still equal the selection already stored in the manifest.
    training = json.loads((root / "results/task06/training_complete.json").read_text())
    calibration = json.loads((root / "results/task06/calibration.json").read_text())
    decision = json.loads((root / "results/task06/headroom_decision.json").read_text())
    if (training["fa_best"] != manifest["fa_best"] or calibration["selected"] != manifest["calibrated_tilts"]
            or decision != manifest["D1"] or not calibration["complete"]):
        raise RuntimeError("Experiment 2 is closed: frozen selection or decision changed")
    return {"forecast_summaries": len(files), "checkpoint_records": len(checkpoints),
            "trajectory_files": len(cells), "recorded_forecasts_and_source_audit_verified": True}


def _frozen_public_files(manifest: dict) -> dict:
    """Deduplicate every artifact that the public preregistration promises.

    The pinned raw CSV is an external source, represented by a content hash and
    its existing public source URL. It is intentionally not a required Git blob.
    """
    files = {}

    def add(name, digest):
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or not isinstance(digest, str) or len(digest) != 64):
            raise RuntimeError('Experiment 2 is closed: invalid frozen public artifact identity')
        if name in files and files[name] != digest:
            raise RuntimeError(f'Experiment 2 is closed: conflicting frozen hashes for {name}')
        files[name] = digest

    for field in ('analysis_code_sha256', 'forecast_summary_sha256', 'required_evidence_sha256'):
        values = manifest.get(field)
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f'Experiment 2 is closed: missing public artifact map {field}')
        for name, digest in values.items():
            add(name, digest)
    for record in manifest['checkpoints'].values():
        add(record['path'], record['sha256'])
    for cell in manifest['seed_table']:
        add(cell['trajectory_file'], cell['trajectory_sha256'])
    for name, field in FROZEN_REQUIRED_FILES.items():
        add(name, manifest.get(field))
    return files


def _committed_sha256(root: Path, commit: str, name: str) -> str:
    """Hash a Git blob in bounded chunks, without materializing the full file."""
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(['git', 'cat-file', 'blob', f'{commit}:{name}'],
                                   cwd=root, stdout=subprocess.PIPE, stderr=errors)
        try:
            for block in iter(lambda: process.stdout.read(1 << 20), b''):
                digest.update(block)
            status = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        if status:
            errors.seek(0)
            explanation = errors.read(2000).decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'Experiment 2 is closed: frozen commit lacks readable artifact {name}: {explanation}')
    return digest.hexdigest()


def _verify_public_frozen_artifacts(root: Path, commit: str, manifest: dict) -> dict:
    files = _frozen_public_files(manifest)
    for name, expected in sorted(files.items()):
        if _committed_sha256(root, commit, name) != expected:
            raise RuntimeError(f'Experiment 2 is closed: committed frozen artifact differs: {name}')
    return {'committed_artifacts': len(files), 'all_committed_sha256_match': True,
            'raw_csv': 'External pinned source; not required as a Git blob'}


def verify_freeze(root: Path = ROOT) -> dict:
    """Require unchanged frozen analysis and a commit reachable on origin's ref."""
    folder = root / "results/task06"
    receipt_path, manifest_path = folder / "freeze_pushed.json", folder / "manifest.json"
    if not receipt_path.exists():
        raise RuntimeError("Experiment 2 is closed: pushed Phase 3 freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if receipt.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("Experiment 2 is closed: freeze manifest hash differs")
    analysis_hashes = manifest.get("analysis_code_sha256")
    if not isinstance(analysis_hashes, dict) or not analysis_hashes:
        raise RuntimeError("Experiment 2 is closed: analysis-code hash map is absent")
    if receipt.get("analysis_code_sha256") != analysis_hashes:
        raise RuntimeError("Experiment 2 is closed: receipt analysis-code hashes differ")
    for name, digest in analysis_hashes.items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Experiment 2 is closed: analysis code changed: {name}")
    artifact_verification = _verify_frozen_artifacts(root, manifest)
    commit, remote, ref = receipt.get("commit"), receipt.get("remote"), receipt.get("ref")
    if remote != "origin" or not isinstance(ref, str) or not ref.startswith("refs/heads/"):
        raise RuntimeError("Experiment 2 is closed: invalid freeze remote/ref")
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
    if origin.removesuffix(".git") not in {"git@github.com:ReloadLightly/evopolis", "https://github.com/ReloadLightly/evopolis", "ssh://git@github.com/ReloadLightly/evopolis"}:
        raise RuntimeError("Experiment 2 is closed: origin is not ReloadLightly/evopolis")
    remote_line = subprocess.check_output(["git", "ls-remote", "--exit-code", remote, ref], cwd=root, text=True, timeout=60).strip().split()
    if len(remote_line) != 2 or remote_line[1] != ref:
        raise RuntimeError("Experiment 2 is closed: remote freeze commit is unverified")
    remote_commit = remote_line[0]
    # Exact remote tip at first opening; descendant tips are valid when resuming.
    relation = subprocess.run(["git", "merge-base", "--is-ancestor", commit, remote_commit], cwd=root, check=False)
    if relation.returncode:
        raise RuntimeError("Experiment 2 is closed: freeze commit is not on the remote branch")
    committed_manifest = subprocess.check_output(["git", "show", f"{commit}:results/task06/manifest.json"], cwd=root)
    if hashlib.sha256(committed_manifest).hexdigest() != receipt["manifest_sha256"]:
        raise RuntimeError("Experiment 2 is closed: committed freeze manifest differs")
    public_verification = _verify_public_frozen_artifacts(root, commit, manifest)
    return {**receipt, "verified_remote_commit": remote_commit,
            "artifact_verification": artifact_verification,
            "public_artifact_verification": public_verification}


def load_exp2(source: Path = SOURCE, *, root: Path = ROOT) -> Cohort:
    freeze = verify_freeze(root)
    digest = _source_check(source)
    opening_path = root / 'results/task06/experiment2_opening.json'
    if opening_path.exists():
        opening = json.loads(opening_path.read_text())
        if opening['freeze_commit'] != freeze['commit'] or opening['source_sha256'] != digest:
            raise RuntimeError('Experiment 2 was already opened under a different freeze/source')
    else:
        opening = {'first_access_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'freeze_commit': freeze['commit'], 'source_sha256': digest,
                   'status': 'parsing started; behavioral evidence may have been read',
                   'public_artifact_verification': freeze['public_artifact_verification']}
        # Persist before parsing: a later correctness failure must never restore
        # the appearance of an unopened transfer cohort.
        write_json(opening_path, opening)
    cohort = _parse_selected(source, EXP2_LABELS, human=True, name="Exp2", expected_per_condition=40)
    earlier = json.loads((root / "results/task03/split.json").read_text())["groups"]
    overlap = sorted({g["launch_id"] for g in earlier} & {g["launch_id"] for g in cohort.groups})
    if overlap:
        raise ValueError(f"Experiment 1/2 group overlap: {overlap}")
    opening.update(status='parsing and disjointness checks complete', groups=len(cohort.groups))
    write_json(opening_path, opening)
    cohort.audit.update({"source_sha256": digest, "exp1_launch_overlap": overlap, "freeze_commit": freeze["commit"]})
    return cohort


def load_prefreeze(source: Path = SOURCE) -> tuple[Cohort, Cohort]:
    return load_exp1(), load_synthetic(source)


def game_summaries(cohort: Cohort) -> list[dict]:
    result = []
    for group in cohort.groups:
        i = group["index"]
        means = [math.fsum(cohort.arrays["surplus"][i, p]) / ROUNDS for p in range(PLAYERS)]
        total = math.fsum(means)
        inequality = None if total == 0 else math.fsum(abs(a - b) for a in means for b in means) / (8 * total)
        after = cohort.arrays["next_pool"][i]
        first = next((t + 1 for t, pool in enumerate(after) if pool < 1), None)
        result.append({**group, "surplus": total / PLAYERS, "gini": inequality,
                       "survival": bool(after[-1] > 1), "pool20": float(after[19]), "pool40": float(after[-1]),
                       "first_round_below_one": first, "first_depletion_after_round": first,
                       "player_mean_surplus": means})
    return result


def replay_allocations(cohort: Cohort, *, rules=("Equal", "Mixed", "Proportional", "Interpolating"),
                       include_rows=True) -> dict:
    """Replay public baseline offers and rule-response slopes on every row."""
    results = []
    for group in cohort.groups:
        mechanism = group["mechanism"]
        if mechanism not in rules:
            continue
        i = group["index"]
        for t in range(ROUNDS):
            pool = float(cohort.arrays["pool"][i, t])
            offers = cohort.arrays["offers"][i, :, t]
            previous = None if t == 0 else cohort.arrays["y"][i, :, t - 1]
            expected = allocate(pool, previous, mechanism=mechanism.lower())
            residual = np.asarray(expected) - offers
            slope = expected_slope = slope_residual = None
            if previous is not None and math.fsum(previous) > 0 and math.fsum(offers) > 0:
                b = previous / math.fsum(previous) - .25
                denominator = float(np.dot(b, b))
                if denominator >= 1e-9:
                    slope = float(np.dot(b, offers / math.fsum(offers) - .25) / denominator)
                    expected_slope = {"Equal": 0., "Mixed": .5, "Proportional": 1.,
                                      "Interpolating": 1 - (pool / CAPACITY) ** 22}[mechanism]
                    slope_residual = slope - expected_slope
            record = {"key": group["key"], "cohort": group["cohort"], "mechanism": mechanism,
                      "round_id": t, "pool": pool, "max_abs_offer_residual": float(np.max(np.abs(residual))),
                      "offer_residuals": residual.tolist(), "slope": slope, "expected_slope": expected_slope,
                      "slope_residual": slope_residual,
                      "slope_clipped": slope is not None and (slope < -1 or slope > 2)}
            if "csv_rows" in cohort.arrays:
                record["csv_row"] = int(cohort.arrays["csv_rows"][i, t])
            results.append(record)
    strata = sorted({(r["cohort"], r["mechanism"]) for r in results})
    summaries = []
    for name, rule in strata:
        rows = [r for r in results if (r["cohort"], r["mechanism"]) == (name, rule)]
        slopes = [abs(r["slope_residual"]) for r in rows if r["slope_residual"] is not None]
        summaries.append({"cohort": name, "mechanism": rule, "rows": len(rows),
                          "max_abs_offer_residual": max(r["max_abs_offer_residual"] for r in rows),
                          "offer_rows_above_1e_4": sum(r["max_abs_offer_residual"] > 1e-4 for r in rows),
                          "offer_rows_above_1e_9": sum(r["max_abs_offer_residual"] > 1e-9 for r in rows),
                          "slope_defined_rows": len(slopes), "slope_undefined_rows": len(rows) - len(slopes),
                          "max_abs_slope_residual": max(slopes, default=None),
                          "slope_rows_above_1e_4": sum(s > 1e-4 for s in slopes),
                          "slope_clipped_rows": sum(r["slope_clipped"] for r in rows)})
    return {"allocation_code_sha256": sha256(ROOT / "evopolis/world.py"),
            "residual_sign": "reconstructed offer minus recorded; observed slope minus theoretical slope",
            "summaries": summaries, **({"rows": results} if include_rows else {})}


def prepare_prefreeze(source: Path = SOURCE, results_dir: Path = RESULTS) -> dict:
    started = time.perf_counter()
    human, synthetic = load_prefreeze(source)
    summaries = game_summaries(human) + game_summaries(synthetic)
    write_json(results_dir / "recorded_games.json", {"schema_version": 1, "cohorts": [human.audit, synthetic.audit], "games": summaries})
    replay = replay_allocations(synthetic, rules=("Interpolating",))
    write_json(results_dir / "recorded_interpolating_replay.json", replay)
    result = {"phase": "prefreeze_data", "elapsed_seconds": time.perf_counter() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "human_games": len(human.groups), "synthetic_games": len(synthetic.groups),
              "experiment_2_opened": False, "replay": replay["summaries"]}
    write_json(results_dir / "data_preparation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--open-exp2", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_prefreeze(), indent=2))
    elif args.open_exp2:
        cohort = load_exp2()
        write_json(RESULTS / "experiment2_games.json", {"audit": cohort.audit, "games": game_summaries(cohort)})
        write_json(RESULTS / "experiment2_replay.json", replay_allocations(cohort, rules=("Proportional", "Interpolating")))
        print(json.dumps(cohort.audit, indent=2))
    else:
        parser.error("Choose --prepare or --open-exp2")


if __name__ == "__main__":
    main()
