"""Compact, provenance-checked human-only arrays for the frozen Task 03 fit.

Only the four exact Experiment 1 conditions enter this module. No torch import
is needed to prepare or audit data. Mechanism/identity/future outcomes are saved
as audit metadata, never concatenated into the nine participant observations.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .sources import REVISION, SOURCES, acquire, sha256
from .world import CAPACITY, PLAYERS, ROUNDS, participant_observation


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/task03.json"
RESULTS_DIR = ROOT / "results/task03"
CACHE_DIR = ROOT / "data/cache/task03"
SOURCE_NAME = "sustainable_behavior.csv"
CONDITIONS = {
    "Equal Baseline Exp 1": "Equal",
    "Mixed Baseline Exp 1": "Mixed",
    "Proportional Baseline Exp 1": "Proportional",
    "RL Agent (M1) Exp 1": "M1",
}
EXPECTED_NONFORCED = {"Equal": 1832, "Mixed": 3891, "Proportional": 2893, "M1": 4198}
FIELDS = (
    "mech_name_by_player", "launch_id", "episode_id", "round_id",
    "mechanism_observation.pool", "next_environment_state.pool",
    *(f"offer_{i}" for i in range(PLAYERS)),
    *(f"player_action_{i}" for i in range(PLAYERS)),
    *(f"player_reward_{i}" for i in range(PLAYERS)),
)


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    """Atomic readable JSON; deterministic content stays byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() == content:
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as sink:
        sink.write(content)
        temporary = Path(sink.name)
    temporary.replace(path)


def read_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text())


def stream_groups(source: Path) -> dict[tuple[str, str, str], list[dict]]:
    """Parse the CSV once, retaining only needed columns for human Exp 1.

    Even numeric fields of excluded cohorts are never parsed or summarized.
    """
    groups = defaultdict(list)
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: header.index(name) for name in FIELDS}
        condition_index = positions["mech_name_by_player"]
        for row in reader:
            condition = row[condition_index]
            if condition not in CONDITIONS:
                continue
            selected = {name: row[index] for name, index in positions.items()}
            key = (condition, selected["launch_id"], selected["episode_id"])
            groups[key].append(selected)
    return dict(groups)


def build_arrays(groups: dict) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Audit every legal target before conversion; retain complete histories."""
    shape = (len(groups), PLAYERS, ROUNDS)
    arrays = {
        "x": np.empty((*shape, 9), dtype=np.float64),
        "y": np.empty(shape, dtype=np.int64),
        "n": np.empty(shape, dtype=np.int64),
        "offers": np.empty(shape, dtype=np.float64),
        "surplus": np.empty(shape, dtype=np.float64),
        "pool": np.empty((len(groups), ROUNDS), dtype=np.float64),
        "next_pool": np.empty((len(groups), ROUNDS), dtype=np.float64),
    }
    metadata = []
    for index, key in enumerate(sorted(groups)):
        if key[0] not in CONDITIONS:
            raise ValueError(f"Non-human-Experiment-1 provenance: {key}")
        rows = sorted(groups[key], key=lambda row: int(row["round_id"]))
        if [int(row["round_id"]) for row in rows] != list(range(ROUNDS)):
            raise ValueError(f"Expected each recorded round 0..39 exactly once: {key}")
        previous = None
        for time, row in enumerate(rows):
            if (row["mech_name_by_player"], row["launch_id"], row["episode_id"]) != key:
                raise ValueError("Rows from different full episode keys cannot be combined")
            pool = float(row["mechanism_observation.pool"])
            next_pool = float(row["next_environment_state.pool"])
            offers = tuple(float(row[f"offer_{i}"]) for i in range(PLAYERS))
            returns = tuple(float(row[f"player_action_{i}"]) for i in range(PLAYERS))
            surplus = tuple(float(row[f"player_reward_{i}"]) for i in range(PLAYERS))
            if not all(math.isfinite(value) for value in (pool, next_pool, *offers, *returns, *surplus)):
                raise ValueError(f"Nonfinite source value at {key}, round {time}")
            for player, (offer, contribution) in enumerate(zip(offers, returns)):
                # No rounding, epsilon, clipping, or dropping discrepant rows.
                n = math.floor(offer)
                if offer < 0 or not contribution.is_integer() or not 0 <= contribution <= n:
                    raise ValueError(f"Illegal integer target: {key}, round {time}, player {player}, offer={offer!r}, contribution={contribution!r}")
                arrays["x"][index, player, time] = np.asarray(participant_observation(player, pool, offers, previous)) / CAPACITY
                arrays["y"][index, player, time] = int(contribution)
                arrays["n"][index, player, time] = n
                arrays["offers"][index, player, time] = offer
                arrays["surplus"][index, player, time] = surplus[player]
            arrays["pool"][index, time] = pool
            arrays["next_pool"][index, time] = next_pool
            previous = returns
        count = int(np.count_nonzero(arrays["n"][index] >= 1))
        if count == 0:
            raise ValueError(f"Group has no nonforced choices: {key}")
        metadata.append({"index": index, "key": list(key), "condition": key[0], "launch_id": key[1], "episode_id": key[2], "mechanism": CONDITIONS[key[0]], "nonforced_choices": count, "forced_choices": PLAYERS * ROUNDS - count})
    return arrays, metadata


def assign_splits(groups: list[dict], *, seed: int = 20261001, sizes: tuple[int, int, int] = (24, 8, 8)) -> list[dict]:
    """One PCG64 stream, sorted condition order and sorted full keys per stratum."""
    result = [dict(group) for group in groups]
    rng = np.random.Generator(np.random.PCG64(seed))
    for condition in sorted(CONDITIONS):
        indices = sorted((i for i, group in enumerate(result) if group["condition"] == condition), key=lambda i: tuple(result[i]["key"]))
        if len(indices) != sum(sizes):
            raise ValueError(f"Unexpected number of groups for {condition}: {len(indices)}")
        shuffled = rng.permutation(indices)
        start = 0
        for split, size in zip(("train", "validation", "test"), sizes):
            for index in shuffled[start:start + size]:
                result[int(index)]["split"] = split
            start += size
    launch_splits = defaultdict(set)
    for group in result:
        launch_splits[group["launch_id"]].add(group["split"])
    if any(len(splits) > 1 for splits in launch_splits.values()):
        raise ValueError("launch_id crosses split boundaries")
    return result


def split_counts(groups: list[dict]) -> dict:
    counts = {}
    for split in ("train", "validation", "test", "all"):
        counts[split] = {}
        for mechanism in CONDITIONS.values():
            selected = [g for g in groups if g["mechanism"] == mechanism and (split == "all" or g["split"] == split)]
            total = len(selected) * PLAYERS * ROUNDS
            nonforced = sum(g["nonforced_choices"] for g in selected)
            counts[split][mechanism] = {"groups": len(selected), "group_rounds": len(selected) * ROUNDS, "choices": total, "nonforced_choices": nonforced, "forced_choices": total - nonforced, "forced_fraction": (total - nonforced) / total if total else None}
    return counts


def rollout_seed_table(config: dict) -> dict:
    rows = []
    for family in config["families"]:
        for training_seed in config["training_seeds"]:
            for mechanism in config["rollout_baselines"]:
                for rollout_index in range(config["rollout_games_per_cell"]):
                    index = len(rows)
                    seed = int(np.random.SeedSequence([config["rollout_seed_namespace"], index]).generate_state(1, dtype=np.uint64)[0])
                    rows.append({"index": index, "family": family, "training_seed": training_seed, "mechanism": mechanism, "rollout_index": rollout_index, "rollout_seed": seed})
    seeds = {row["rollout_seed"] for row in rows}
    if len(seeds) != len(rows) or seeds.intersection({config["split_seed"], config["bootstrap_seed"], *config["training_seeds"]}):
        raise ValueError("Rollout seeds must be unique and distinct from fitting/evaluation seeds")
    return {"namespace": config["rollout_seed_namespace"], "algorithm": "numpy.random.SeedSequence([namespace, global_row_index]).generate_state(1, dtype=uint64); row order family, training seed, allocation, rollout index", "numpy_version": np.__version__, "games": rows, "table_hash": content_hash(rows)}


def prepare(*, raw_dir: Path = ROOT / "data/raw", cache_dir: Path = CACHE_DIR, results_dir: Path = RESULTS_DIR, config_path: Path = CONFIG_PATH) -> dict:
    config = read_config(config_path)
    pinned = json.loads((ROOT / "results/task01/manifest.json").read_text())
    expected = {entry["file"]: entry["sha256"] for entry in pinned["sources"]}
    source = raw_dir / SOURCE_NAME
    if not source.exists():
        acquire(raw_dir, expected)
    if sha256(source) != expected[SOURCE_NAME]:
        raise ValueError("Pinned human source checksum mismatch")
    print("Preparing only the four exact human Experiment 1 conditions", flush=True)
    arrays, groups = build_arrays(stream_groups(source))
    if len(groups) != 160:
        raise ValueError(f"Expected 160 human Experiment 1 groups, got {len(groups)}")
    groups = assign_splits(groups, seed=config["split_seed"])
    counts = split_counts(groups)
    if {mechanism: values["nonforced_choices"] for mechanism, values in counts["all"].items()} != EXPECTED_NONFORCED:
        raise ValueError("Nonforced audit differs from the declared 12,814-choice planning audit")
    manifest = {
        "schema_version": 1, "source_file": SOURCE_NAME, "source_url": SOURCES[SOURCE_NAME], "source_sha256": expected[SOURCE_NAME], "upstream_revision": REVISION,
        "split_seed": config["split_seed"], "random_generator": "numpy.random.Generator(numpy.random.PCG64(seed)); single stream, sorted condition strings, then sorted full keys, 24/8/8 permutation slices", "numpy_version": np.__version__,
        "groups": groups, "counts": counts,
        "participant_limit": "launch_id isolation verified; cross-group repeat participants cannot be checked without released participant identifiers",
        "test_status": "Held out from Task 03 fitting and selection; all Experiment 1 groups were described in Task 01",
    }
    manifest["manifest_hash"] = content_hash(manifest)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as sink:
        temporary = Path(sink.name)
        np.savez_compressed(sink, **arrays)
    target = cache_dir / "human_exp1.npz"
    temporary.replace(target)
    write_json(results_dir / "split.json", manifest)
    write_json(results_dir / "rollout_seeds.json", rollout_seed_table(config))
    preparation = {
        "schema_version": 1, "source_sha256": expected[SOURCE_NAME], "split_hash": manifest["manifest_hash"], "config_hash_at_preparation": content_hash(config),
        "array_sha256": sha256(target), "array_file": "data/cache/task03/human_exp1.npz", "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "array_uncompressed_bytes": sum(value.nbytes for value in arrays.values()), "array_compressed_bytes": target.stat().st_size,
        "feature_order": config["feature_order"], "normalization": CAPACITY,
        "audit": {"groups": len(groups), "group_rounds": 6400, "player_decisions": 25600, "nonforced_choices": 12814, "forced_choices": 12786, "noninteger_targets": 0, "support_violations": 0, "zero_scoreable_groups": 0, "launch_id_split_overlap": 0},
        "provenance_filter": list(CONDITIONS), "numeric_input_columns": ["offer_0..3", "same-episode previous player_action_0..3", "mechanism_observation.pool"],
        "excluded_cohorts": "All BC1/BC2 records and human Experiments 2-4 ignored before parsing numeric fields",
        "axes": "x: group, resident, round, feature; y/n/offers/surplus: group,resident,round; pool/next_pool: group,round",
    }
    write_json(results_dir / "preparation.json", preparation)
    print(f"Audited 160 groups / 25,600 decisions / 12,814 nonforced choices; compact arrays {target.stat().st_size:,} bytes", flush=True)
    return manifest


def load(*, cache_dir: Path = CACHE_DIR, results_dir: Path = RESULTS_DIR) -> tuple[dict[str, np.ndarray], dict]:
    """Load verified compact arrays, without importing torch or source outcomes."""
    manifest = json.loads((results_dir / "split.json").read_text())
    recorded_hash = manifest["manifest_hash"]
    if content_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}) != recorded_hash:
        raise ValueError("Split manifest integrity check failed")
    preparation = json.loads((results_dir / "preparation.json").read_text())
    target = cache_dir / "human_exp1.npz"
    if preparation["split_hash"] != recorded_hash or sha256(target) != preparation["array_sha256"]:
        raise ValueError("Prepared array or split hash mismatch")
    with np.load(target, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="stream the pinned CSV and regenerate deterministic compact arrays")
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare()
    else:
        _, manifest = load()
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
