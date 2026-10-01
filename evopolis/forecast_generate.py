"""Task 04 forecasts from observed boundaries, with paired reproducible streams.

No Torch import occurs until inference is explicitly requested. The original
Task 03 model and environment remain byte-identical. Only human Experiment 1
arrays enter this module; no withheld transfer records are read.
"""

import argparse
from contextlib import closing
import datetime
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sqlite3
import time
import zlib

import numpy as np

from .behavior_data import ROOT, content_hash, load, write_json
from .sources import sha256
from .world import CAPACITY, MULTIPLIER, PLAYERS, ROUNDS, WorldState, allocate, participant_observation, step

RESULTS = ROOT / "results/task04"
ARCHIVE = RESULTS / "forecasts.sqlite3"
CONFIG = ROOT / "configs/task04.json"
FAMILIES = ("constant", "linear", "feedforward", "recurrent")
NEURAL = ("feedforward", "recurrent")
SEEDS = (17, 29, 43)
MECHANISMS = ("Equal", "Mixed", "Proportional")
ORIGINS = (0, 5, 10, 20)
HORIZONS = (1, 5, 10, 20)
NAMESPACE = 20261015


def packed(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def validate_configuration():
    configuration = json.loads(CONFIG.read_text())
    required = {"families": list(FAMILIES), "continued_families": list(NEURAL), "training_seeds": list(SEEDS),
                "mechanisms": list(MECHANISMS), "origins": list(ORIGINS), "horizons": list(HORIZONS),
                "primary_origin": 5, "primary_horizon": 10, "branches_per_cell": 64,
                "seed_namespace": NAMESPACE, "total_branches": 150912, "primary_bank": 0}
    if any(configuration.get(key) != value for key, value in required.items()):
        raise ValueError("Configuration differs from the prescribed forecast budget or populations")


def baseline_groups(manifest, split="test"):
    return [g for g in manifest["groups"] if g["split"] == split and g["mechanism"] in MECHANISMS]


def eligible(arrays, group, origin):
    return bool(np.any(arrays["offers"][group["index"], :, origin] >= 1))


def checkpoint_registry(require_continued=True):
    """Read published selection metadata without importing a tensor framework."""
    rows = []
    for budget, families in (("original", FAMILIES), ("continued", NEURAL)):
        directory = ROOT / ("results/task03" if budget == "original" else "results/task04")
        for family in families:
            for seed in SEEDS:
                folder = directory / "weights" / f"{family}_{seed}"
                if not (folder / "metadata.json").exists():
                    if budget == "continued" and not require_continued:
                        continue
                    raise FileNotFoundError(f"Missing selected checkpoint metadata: {folder}")
                metadata = json.loads((folder / "metadata.json").read_text())
                path = folder / "best.pt"
                digest = sha256(path)
                if metadata.get("best_sha256", digest) != digest:
                    raise ValueError("Selected checkpoint differs from its metadata")
                rows.append({"budget": budget, "budget_epochs": 120 if budget == "original" else 480,
                             "family": family, "seed": seed, "training_seed": seed,
                             "path": str(path.relative_to(ROOT)), "sha256": digest,
                             "checkpoint_sha256": digest, "selected_epoch": metadata["selected_epoch"]})
    return rows


def load_checkpoint(record):
    import torch
    from .behavior_models import BehaviorModel

    path = ROOT / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Checkpoint changed after registry freezing")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["family"], checkpoint["seed"], checkpoint["epoch"]) != (record["family"], record["seed"], record["selected_epoch"]):
        raise ValueError("Checkpoint identity differs from selected registry")
    model = BehaviorModel(record["family"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def numerical_audit(arrays, manifest):
    """Report source values, support differences and residuals without repairs."""
    groups = baseline_groups(manifest)
    counts, differences, origins, residuals, allocation_residuals = {}, {}, [], [], []
    for origin in ORIGINS:
        counts[str(origin)] = {m: sum(eligible(arrays, g, origin) for g in groups if g["mechanism"] == m) for m in MECHANISMS}
    for mechanism in MECHANISMS:
        differences[mechanism] = {"canonical_one_higher": 0, "other": 0}
    for group in groups:
        index = group["index"]
        for t in range(ROUNDS):
            pool = float(arrays["pool"][index, t])
            offers = arrays["offers"][index, :, t]
            contributions = arrays["y"][index, :, t]
            previous = None if t == 0 else arrays["y"][index, :, t-1]
            canonical = allocate(pool, previous, mechanism=group["mechanism"].lower())
            delta = np.floor(canonical).astype(np.int64) - np.floor(offers).astype(np.int64)
            differences[group["mechanism"]]["canonical_one_higher"] += int(np.count_nonzero(delta == 1))
            differences[group["mechanism"]]["other"] += int(np.count_nonzero((delta != 0) & (delta != 1)))
            allocated = math.fsum(offers)
            residual = pool - allocated
            allocation_residuals.append(residual)
            if np.any(offers < 0) or allocated > pool or np.any(contributions < 0) or np.any(contributions > np.floor(offers)):
                raise ValueError(f"Infeasible recorded boundary: {group['key']} round {t}")
            equation = step(pool, offers, contributions, integer_contributions=True).next_pool
            residuals.append(float(arrays["next_pool"][index, t]) - equation)
            if t in ORIGINS:
                origins.append({"group_index": index, "key": group["key"], "mechanism": group["mechanism"], "origin": t,
                                "pool": pool, "offers": offers.tolist(), "legal_max": np.floor(offers).astype(int).tolist(),
                                "unallocated": residual, "eligible": eligible(arrays, group, t),
                                "canonical_offers": list(canonical), "canonical_legal_max": np.floor(canonical).astype(int).tolist(),
                                "support_differences": delta.tolist()})
    expected = {"0": {"Equal": 8, "Mixed": 8, "Proportional": 8}, "5": {"Equal": 5, "Mixed": 8, "Proportional": 8},
                "10": {"Equal": 2, "Mixed": 7, "Proportional": 7}, "20": {"Equal": 0, "Mixed": 6, "Proportional": 7}}
    if counts != expected or len(origins) != 96:
        raise ValueError(f"Origin eligibility differs from declared planning audit: {counts}")
    actual_differences = {m: differences[m]["canonical_one_higher"] for m in MECHANISMS}
    if actual_differences != {"Equal": 32, "Mixed": 32, "Proportional": 102} or any(d["other"] for d in differences.values()):
        raise ValueError(f"Recorded support audit differs from planning audit: {differences}")
    primary = [row for row in origins if row["origin"] == 5 and row["eligible"]]
    primary_support = {m: {"groups": sum(row["mechanism"] == m for row in primary),
                           "changed_player_maxima": sum(sum(v != 0 for v in row["support_differences"]) for row in primary if row["mechanism"] == m),
                           "groups_with_any_support_change": sum(any(row["support_differences"]) for row in primary if row["mechanism"] == m)} for m in MECHANISMS}
    return {"group_count": len(groups), "recorded_rounds": len(residuals), "origin_count": len(origins), "origins": origins,
            "eligible_counts": counts, "support_differences": differences, "support_differences_total": sum(actual_differences.values()),
            "primary_eligible_first_allocation_support": primary_support,
            "origin_unallocated_min": min(row["unallocated"] for row in origins), "origin_unallocated_max": max(row["unallocated"] for row in origins),
            "recorded_transition_residual": {"definition": "recorded next pool minus equation from recorded pool/offers/returns",
                                            "min": min(residuals), "max": max(residuals), "mean_abs": float(np.mean(np.abs(residuals))),
                                            "max_abs": max(abs(v) for v in residuals)},
            "source_sha256": manifest["source_sha256"], "split_sha256": sha256(ROOT / "results/task03/split.json")}


def seed_table(arrays, manifest):
    rows = []
    for group in baseline_groups(manifest):
        for origin in ORIGINS:
            for family_index, family in enumerate(FAMILIES):
                for seed in SEEDS:
                    for bank_index, bank in enumerate(("main", "second")):
                        if bank == "second" and (origin != 5 or not eligible(arrays, group, origin)):
                            continue
                        seeds = [int(np.random.SeedSequence([NAMESPACE, group["index"], origin, family_index, seed, bank_index, branch]).generate_state(1, dtype=np.uint64)[0]) for branch in range(64)]
                        rows.append({"group_index": group["index"], "key": group["key"], "origin": origin, "family": family,
                                     "training_seed": seed, "bank": bank, "rollout_seeds": [str(s) for s in seeds]})
    flattened = [s for row in rows for s in row["rollout_seeds"]]
    if len(flattened) != 89856 or len(set(flattened)) != len(flattened):
        raise ValueError("Incomplete or colliding seed table")
    return {"namespace": NAMESPACE, "rows": rows, "table_hash": content_hash(rows), "distinct_branch_streams": len(flattened),
            "algorithm": "SeedSequence([20261015, group_index, origin, family_index, training_seed, bank_index, branch_index]).generate_state(1,uint64); families constant/linear/feedforward/recurrent, banks main/second",
            "resident_streams": "PCG64(SeedSequence([branch_seed, resident_index])); four independent streams per branch",
            "paired_budgets": "original and continued share seeds within family; canonical first allocation shares main-bank seeds",
            "numpy_version": np.__version__}


def prepare():
    validate_configuration()
    arrays, manifest = load()
    audit = numerical_audit(arrays, manifest)
    seeds = seed_table(arrays, manifest)
    for filename, value in (("numerical_audit.json", audit), ("forecast_seeds.json", seeds)):
        path = RESULTS / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to replace changed frozen {filename}")
        write_json(path, value)
    print(f"Audited 96 feasible origins; primary eligible 21; frozen {seeds['distinct_branch_streams']:,} distinct branch streams", flush=True)
    return audit


def observed_rounds(arrays, group):
    index = group["index"]
    rows = []
    for t in range(ROUNDS):
        pool = float(arrays["pool"][index, t])
        offers = arrays["offers"][index, :, t].tolist()
        returns = arrays["y"][index, :, t].tolist()
        equation = step(pool, offers, returns, integer_contributions=True).next_pool
        after = float(arrays["next_pool"][index, t])
        cumulative = [math.fsum(arrays["surplus"][index, p, :t+1]) for p in range(PLAYERS)]
        rows.append({"round_id": t, "pool_before": pool, "pool_after": after, "offers": offers, "contributions": returns,
                     "surplus": arrays["surplus"][index, :, t].tolist(), "cumulative_surplus": cumulative,
                     "group_cumulative_surplus": math.fsum(cumulative), "equation_after": equation, "pool_residual": after-equation,
                     "equation_input_valid": True, "after_source": "recorded human Experiment 1 outcome", "active_count": sum(v >= 1 for v in offers),
                     "padded": False, "observed": True})
    return rows


def simulate_branches(model, family, arrays, group, origin, seeds, *, horizon=20, canonical_first=False):
    """Free-run futures after recorded prefix, without accessing future actions.

    Prefix observations 0..k-1 are read only for GRU warming. The source-round-k
    observation is constructed afresh from pool, chosen allocation and k-1
    returns and is processed exactly once. All four emissions precede any step.
    """
    import torch
    from .behavior_models import emission_parts, emission_stats
    from .behavior_generate import sample_integer

    if origin < 0 or horizon < 1 or origin+horizon > ROUNDS or not seeds:
        raise ValueError("Forecast must fit inside the recorded 40-round horizon")
    index, count = group["index"], len(seeds)
    pool = float(arrays["pool"][index, origin])
    previous = None if origin == 0 else tuple(arrays["y"][index, :, origin-1].tolist())
    first_offers = (allocate(pool, previous, mechanism=group["mechanism"].lower()) if canonical_first
                    else tuple(arrays["offers"][index, :, origin].tolist()))
    # Validate strict source feasibility and legal support, without epsilon repair.
    if min(first_offers) < 0 or math.fsum(first_offers) > pool:
        raise ValueError("Infeasible forecast origin")
    states = [WorldState(pool, origin, previous) for _ in seeds]
    streams = [[np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), p]))) for p in range(PLAYERS)] for seed in seeds]
    hidden = None
    if family == "recurrent":
        if origin:
            prefix = torch.tensor(arrays["x"][index, :, :origin], dtype=torch.float32)
            with torch.no_grad():
                _, prefix_hidden = model(prefix)
            hidden = prefix_hidden.repeat(1, count, 1)
        else:
            hidden = torch.zeros(1, count*PLAYERS, 32)
    paths = {"pool_before": np.zeros((count, horizon)), "pool_after": np.zeros((count, horizon)),
             "window_surplus": np.zeros((count, horizon)), "participation": np.zeros((count, horizon), dtype=np.int8),
             "offers": np.zeros((count, horizon, PLAYERS)), "contributions": np.zeros((count, horizon, PLAYERS), dtype=np.int16),
             "surplus": np.zeros((count, horizon, PLAYERS)), "expected_contribution": np.zeros((count, horizon, PLAYERS)),
             "p_zero": np.ones((count, horizon, PLAYERS)), "p_max": np.ones((count, horizon, PLAYERS)),
             "padded": np.ones((count, horizon), dtype=bool), "actual_rounds": np.zeros(count, dtype=np.int16)}
    for offset in range(horizon):
        live = [i for i, state in enumerate(states) if not state.done]
        if offset:
            paths["window_surplus"][:, offset] = paths["window_surplus"][:, offset-1]
        if not live:
            continue
        offers = {i: first_offers if offset == 0 else allocate(states[i].pool, states[i].previous_contributions, mechanism=group["mechanism"].lower()) for i in live}
        observations = [participant_observation(p, states[i].pool, offers[i], states[i].previous_contributions) for i in live for p in range(PLAYERS)]
        x = torch.tensor(np.asarray(observations)/CAPACITY, dtype=torch.float32).unsqueeze(1)
        resident_indices = [i*PLAYERS+p for i in live for p in range(PLAYERS)]
        with torch.no_grad():
            raw, next_hidden = model(x, None if hidden is None else hidden[:, resident_indices])
            if hidden is not None:
                hidden[:, resident_indices] = next_hidden
            raw = raw[:, 0]
            n = torch.tensor([math.floor(v) for i in live for v in offers[i]], dtype=torch.int64)
            logs, alpha, beta = emission_parts(raw)
            stats = emission_stats(raw, n)
        weights = logs.exp().detach().numpy()
        a, b = alpha.detach().numpy(), beta.detach().numpy()
        statistics = {name: value.detach().numpy() for name, value in stats.items()}
        for position, i in enumerate(live):
            begin = position*PLAYERS
            contributions = [sample_integer(streams[i][p], math.floor(offers[i][p]), weights[begin+p], a[begin+p], b[begin+p]) for p in range(PLAYERS)]
            prior = states[i]
            states[i], result = prior.advance(offers[i], contributions, integer_contributions=True)
            paths["pool_before"][i, offset] = prior.pool
            paths["pool_after"][i, offset] = result.next_pool
            paths["offers"][i, offset] = offers[i]
            paths["contributions"][i, offset] = contributions
            paths["surplus"][i, offset] = result.surplus
            paths["window_surplus"][i, offset] += math.fsum(result.surplus)
            paths["participation"][i, offset] = sum(v >= 1 for v in offers[i])
            paths["padded"][i, offset] = False
            paths["actual_rounds"][i] += 1
            for destination, source in (("expected_contribution", "mean"), ("p_zero", "p_zero"), ("p_max", "p_max")):
                paths[destination][i, offset] = statistics[source][begin:begin+PLAYERS]
    if not all(np.isfinite(value).all() for value in paths.values()):
        raise FloatingPointError("Nonfinite forecast")
    return paths


def branch_episode(paths, branch_index, arrays, group, origin, seeds):
    prefix = [math.fsum(arrays["surplus"][group["index"], p, :origin]) for p in range(PLAYERS)]
    rows = []
    horizon = paths["pool_after"].shape[1]
    for offset in range(horizon):
        offers = paths["offers"][branch_index, offset].tolist()
        returns = paths["contributions"][branch_index, offset].tolist()
        cumulative = [prefix[p] + math.fsum(paths["surplus"][branch_index, :offset+1, p]) for p in range(PLAYERS)]
        padded = bool(paths["padded"][branch_index, offset])
        after = float(paths["pool_after"][branch_index, offset])
        predictions = [{"expected_contribution": float(paths["expected_contribution"][branch_index, offset, p]),
                        "p_zero": float(paths["p_zero"][branch_index, offset, p]), "p_max": float(paths["p_max"][branch_index, offset, p]),
                        "legal_max": math.floor(offers[p]), "evaluated": not padded} for p in range(PLAYERS)]
        rows.append({"round_id": origin+offset, "pool_before": float(paths["pool_before"][branch_index, offset]), "pool_after": after,
                     "offers": offers, "contributions": returns, "surplus": paths["surplus"][branch_index, offset].tolist(),
                     "cumulative_surplus": cumulative, "group_cumulative_surplus": math.fsum(cumulative),
                     "window_cumulative_surplus": float(paths["window_surplus"][branch_index, offset]),
                     "equation_after": after, "pool_residual": 0.0, "equation_input_valid": True,
                     "active_count": int(paths["participation"][branch_index, offset]), "padded": padded, "observed": False,
                     "after_source": "post-termination zero padding" if padded else "model forecast through WorldState", "predictions": predictions})
    return {"branch_index": branch_index, "rollout_seed": str(seeds[branch_index]), "rounds": rows,
            "actual_rounds": int(paths["actual_rounds"][branch_index]),
            "exact_zero_termination": bool(paths["pool_after"][branch_index, -1] == 0)}


def generate_cell(model, checkpoint, arrays, group, origin, seeds, *, horizon=20, canonical_first=False):
    paths = simulate_branches(model, checkpoint["family"], arrays, group, origin, seeds, horizon=horizon, canonical_first=canonical_first)
    executed = np.argwhere(~paths["padded"])
    maximum_transition_residual = 0.0
    maximum_allocation_overshoot = 0.0
    for branch_index, offset in executed:
        before = float(paths["pool_before"][branch_index, offset])
        allocated = math.fsum(paths["offers"][branch_index, offset])
        returned = math.fsum(paths["contributions"][branch_index, offset])
        equation = min(CAPACITY, before-allocated+MULTIPLIER*returned)
        maximum_transition_residual = max(maximum_transition_residual, abs(float(paths["pool_after"][branch_index, offset])-equation))
        maximum_allocation_overshoot = max(maximum_allocation_overshoot, allocated-before)
    maximum_legal_violation = float(max(0, np.max(-paths["contributions"]),
                                        np.max(paths["contributions"]-np.floor(paths["offers"]))))
    if maximum_transition_residual != 0 or maximum_legal_violation != 0 or maximum_allocation_overshoot > 0:
        raise ValueError("Generated numerical transition, allocation or legal-support audit failed")
    simulation_audit = {"executed_rounds": len(executed), "padded_rounds": int(paths["padded"].sum()),
                        "max_abs_transition_residual": maximum_transition_residual,
                        "max_legal_support_violation": maximum_legal_violation,
                        "max_allocation_over_pool": maximum_allocation_overshoot}
    final_surplus = paths["window_surplus"][:, -1]
    median = float(np.median(final_surplus))
    chosen = min(range(len(seeds)), key=lambda i: (abs(float(final_surplus[i])-median), i))
    branch = branch_episode(paths, chosen, arrays, group, origin, seeds)
    zero = branch if chosen == 0 else branch_episode(paths, 0, arrays, group, origin, seeds)
    observed = observed_rounds(arrays, group)
    summaries = {name: {"mean": value.mean(axis=0).tolist(), "p10": np.quantile(value, .1, axis=0).tolist(),
                        "p50": np.quantile(value, .5, axis=0).tolist(), "p90": np.quantile(value, .9, axis=0).tolist()}
                 for name, value in paths.items() if name in ("pool_after", "participation", "window_surplus")}
    bands = [{"round_id": origin+t, "pool_lower": summaries["pool_after"]["p10"][t], "pool_median": summaries["pool_after"]["p50"][t],
              "pool_upper": summaries["pool_after"]["p90"][t], "participation_lower": summaries["participation"]["p10"][t],
              "participation_median": summaries["participation"]["p50"][t], "participation_upper": summaries["participation"]["p90"][t]} for t in range(horizon)]
    endpoints = {str(h): np.stack((paths["pool_after"][:, h-1]/CAPACITY, paths["window_surplus"][:, h-1]/(CAPACITY*h)), axis=-1).tolist() for h in HORIZONS if h <= horizon}
    observed_endpoints = {str(h): [float(arrays["next_pool"][group["index"], origin+h-1])/CAPACITY,
                                  math.fsum(arrays["surplus"][group["index"], :, origin:origin+h].flat)/(CAPACITY*h)] for h in HORIZONS if h <= horizon}
    body = {"rounds": observed[:origin]+branch["rounds"], "observed_rounds": observed, "simulation_audit": simulation_audit,
            "selected_branch": branch, "branch_zero": zero, "bands": bands, "summary_bands": summaries,
            "endpoints": endpoints, "observed_endpoints": observed_endpoints,
            "branch_selection": "nearest median total retained resources over full forecast window; stable branch-index tie break; illustrative only",
            "provenance": {"source": "observed human Experiment 1 prefix, followed by frozen learned-agent forecast",
                           "boundary": "canonical" if canonical_first else "recorded", "online_parameter_adaptation": False,
                           "recurrent_state": "per resident, warmed only on recorded observations before origin"}}
    # Keep numerical trajectories needed by proper scores, without redundant per-step neural emissions.
    retained = {name: paths[name] for name in ("pool_after", "window_surplus", "participation", "actual_rounds")}
    return body, retained


def cell_id(record, group, origin, bank, convention):
    return f"forecast-{record['budget']}-{record['family']}-{record['seed']}-g{group['index']}-k{origin}-{bank}-{convention}"


def recipes(arrays, manifest, registry, table):
    index = {(r["group_index"], r["origin"], r["family"], r["training_seed"], r["bank"]): r["rollout_seeds"] for r in table["rows"]}
    for record in registry:
        for group in baseline_groups(manifest):
            for origin in ORIGINS:
                choices = [("main", "recorded", 20)]
                if origin == 5 and eligible(arrays, group, origin):
                    choices.append(("second", "recorded", 10))
                    if record["family"] in NEURAL:
                        choices.append(("main", "canonical", 10))
                for bank, convention, horizon in choices:
                    seeds = index[(group["index"], origin, record["family"], record["seed"], bank)]
                    yield record, group, origin, bank, convention, horizon, seeds


def archive_identity(registry, table):
    return {"configuration_sha256": sha256(CONFIG), "split_sha256": sha256(ROOT / "results/task03/split.json"),
            "source_sha256": json.loads((ROOT / "results/task03/split.json").read_text())["source_sha256"],
            "seed_table_hash": table["table_hash"], "checkpoint_registry": registry,
            "code_sha256": {name: sha256(ROOT / name) for name in ("evopolis/forecast_generate.py", "evopolis/world.py", "evopolis/behavior_models.py", "evopolis/behavior_data.py", "evopolis/behavior_generate.py")}}


def pack_paths(paths):
    output = io.BytesIO()
    np.savez_compressed(output, **paths)
    return output.getvalue()


def unpack_paths(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def metadata_row(record, group, origin, bank, convention, horizon, body, source_hash):
    branch = body["selected_branch"]
    return {**record, "id": cell_id(record, group, origin, bank, convention), "cohort": "forecast", "human_id": hashlib.sha256(packed(group["key"]).encode()).hexdigest(),
            "group_index": group["index"], "key": group["key"], "condition": group["condition"], "launch_id": group["launch_id"],
            "episode_id": group["episode_id"], "mechanism": group["mechanism"], "origin": origin, "horizon": horizon,
            "bank": bank, "boundary": convention, "convention": convention, "selected_branch_index": branch["branch_index"],
            "rollout_seed": branch["rollout_seed"], "source_sha256": source_hash, "branches": 64, "round_count": origin+horizon,
            "actual_rounds": branch["actual_rounds"], "source_label": "Forecast from observed human history", "provenance_label": "Observed prefix / model-generated continuation"}


def generate():
    from .behavior_train import runtime_setup
    runtime_setup()
    validate_configuration()
    arrays, manifest = load()
    table = json.loads((RESULTS / "forecast_seeds.json").read_text())
    if table["table_hash"] != content_hash(table["rows"]):
        raise ValueError("Frozen forecast seed table changed")
    registry = checkpoint_registry()
    identity = archive_identity(registry, table)
    completed = json.loads((RESULTS / "training_complete.json").read_text())
    if completed["fits"] != 6 or completed["additional_epochs"] != 2160:
        raise ValueError("All six declared continuations must finish before forecasting")
    benchmark_receipt = json.loads((RESULTS / "forecast_benchmark.json").read_text())
    if not benchmark_receipt["validation_only"] or benchmark_receipt["selected_chunk_size"] != 64:
        raise ValueError("Forecast inference requires the declared validation-only chunk benchmark")
    frozen_path = RESULTS / "forecast_frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen["identity"] != identity:
            raise ValueError("Forecast source, selected weights or configuration changed after freezing")
    else:
        write_json(frozen_path, {"frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "identity": identity, "benchmark": benchmark_receipt,
                                 "branches": 150912, "primary_bank": "main / first64", "display_branch": "branch0"})
    partial = ARCHIVE.with_suffix(".sqlite3.partial")
    target = ARCHIVE if ARCHIVE.exists() else partial
    started, cpu_start = time.monotonic(), time.process_time()
    database = sqlite3.connect(target)
    generated, skipped, model_key, model = 0, 0, None, None
    all_recipes = list(recipes(arrays, manifest, registry, table))
    if len(all_recipes) != 2358 or len(all_recipes)*64 != 150912:
        raise ValueError("Forecast budget differs from declared 150,912 branches")
    try:
        database.execute("CREATE TABLE IF NOT EXISTS cells (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, body BLOB NOT NULL, trajectories BLOB NOT NULL)")
        database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        existing = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing and json.loads(existing[0]) != identity:
            raise ValueError("Resume rejected: generation contract changed")
        database.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (packed(identity),))
        database.commit()
        existing_ids = {row[0] for row in database.execute("SELECT id FROM cells")}
        for number, (record, group, origin, bank, convention, horizon, seeds) in enumerate(all_recipes, 1):
            identifier = cell_id(record, group, origin, bank, convention)
            if identifier in existing_ids:
                skipped += 1
                continue
            key = (record["budget"], record["family"], record["seed"])
            if key != model_key:
                model, checkpoint = load_checkpoint(record)
                model_key = key
            body, paths = generate_cell(model, checkpoint, arrays, group, origin, seeds, horizon=horizon, canonical_first=convention == "canonical")
            body["provenance"]["source_sha256"] = manifest["source_sha256"]
            metadata = metadata_row(record, group, origin, bank, convention, horizon, body, manifest["source_sha256"])
            metadata["simulation_audit"] = body["simulation_audit"]
            with database:
                database.execute("INSERT INTO cells VALUES (?,?,?,?)", (identifier, packed(metadata), zlib.compress(packed(body).encode(), 6), pack_paths(paths)))
            generated += 1
            if number % 24 == 0 or number == len(all_recipes):
                print(f"Forecast cells {number}/{len(all_recipes)} ({number*64:,} branches): {record['budget']} {record['family']} seed {record['seed']}; {time.monotonic()-started:.1f}s", flush=True)
        count = database.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
        if count != 2358:
            raise ValueError("Missing forecast cells")
        simulation_totals = {"executed_rounds": 0, "padded_rounds": 0, "max_abs_transition_residual": 0.0,
                             "max_legal_support_violation": 0.0, "max_allocation_over_pool": 0.0}
        for encoded, in database.execute("SELECT metadata FROM cells"):
            audit = json.loads(encoded)["simulation_audit"]
            for name in simulation_totals:
                simulation_totals[name] = simulation_totals[name]+audit[name] if name.endswith("_rounds") else max(simulation_totals[name],audit[name])
        if simulation_totals["executed_rounds"]+simulation_totals["padded_rounds"] != 2615040:
            raise ValueError("Total forecast rounds differ from the fixed principal/secondary/sensitivity budget")
        default_group = next(g for g in baseline_groups(manifest) if g["mechanism"] == "Mixed")
        default_record = next(r for r in registry if r["budget"] == "continued" and r["family"] == "recurrent" and r["seed"] == 17)
        provenance = {"identity": identity, "total_branches": 150912, "principal_branches": 110592, "second_bank_branches": 24192,
                      "canonical_branches": 16128, "weights_frozen": True, "display_default": "first sorted Mixed human test group / continued GRU seed17 / origin5 / branch0",
                      "uncertainty_band": "10th to 90th Monte Carlo percentiles; not human-sample or fitted-seed uncertainty",
                      "selected_branch": "nearest median 20-round window surplus per cell, stable branch index tie break; branch0 also saved"}
        with database:
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('default_id',?)", (packed(cell_id(default_record, default_group, 5, "main", "recorded")),))
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('provenance',?)", (packed(provenance),))
        database.execute("VACUUM")
    finally:
        database.close()
    if target == partial:
        partial.replace(ARCHIVE)
    receipt = {"cells": 2358, "branches": 150912, "generated_cells_this_invocation": generated, "resumed_cells": skipped,
               "wall_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu_start,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "archive_sha256": sha256(ARCHIVE),
               "archive_bytes": ARCHIVE.stat().st_size, "simulation_audit": simulation_totals, "identity": identity}
    write_json(RESULTS / "forecast_generation.json", receipt)
    print(packed({k: v for k, v in receipt.items() if k != "identity"}), flush=True)


def iter_cells(include_paths=True, path=ARCHIVE):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)) as database:
        columns = "metadata,body,trajectories" if include_paths else "metadata,body"
        for row in database.execute(f"SELECT {columns} FROM cells ORDER BY id"):
            yield json.loads(row[0]), json.loads(zlib.decompress(row[1])), unpack_paths(row[2]) if include_paths else None


def get_cell(identifier, include_paths=False, path=ARCHIVE):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)) as database:
        columns = "metadata,body,trajectories" if include_paths else "metadata,body"
        row = database.execute(f"SELECT {columns} FROM cells WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise KeyError(identifier)
    return json.loads(row[0]), json.loads(zlib.decompress(row[1])), unpack_paths(row[2]) if include_paths else None


def benchmark():
    from .behavior_train import runtime_setup
    runtime_setup()
    arrays, manifest = load()
    group = next(g for g in baseline_groups(manifest, "validation") if g["mechanism"] == "Mixed")
    record = next(r for r in checkpoint_registry(False) if r["family"] == "recurrent" and r["seed"] == 17)
    model, checkpoint = load_checkpoint(record)
    started, cpu_start = time.monotonic(), time.process_time()
    seeds = [int(np.random.SeedSequence([NAMESPACE, 999, i]).generate_state(1, dtype=np.uint64)[0]) for i in range(64)]
    body, paths = generate_cell(model, checkpoint, arrays, group, 5, seeds)
    elapsed = time.monotonic()-started
    receipt = {"validation_only": True, "group_index": group["index"], "family": "recurrent", "seed": 17,
               "branches": 64, "horizon": 20, "selected_chunk_size": 64, "resident_batch": 256,
               "wall_seconds": elapsed, "cpu_seconds": time.process_time()-cpu_start,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "projected_full_run_seconds": elapsed*2358, "projection_excludes": "archive IO, mixtures of families/origins and shorter secondary/sensitivity banks",
               "scientific_settings_changed": False}
    write_json(RESULTS / "forecast_benchmark.json", receipt)
    print(packed(receipt), flush=True)


def verify():
    """Fresh-process default-cell regeneration and archived numerical invariants."""
    from .behavior_train import runtime_setup
    runtime_setup()
    arrays, manifest = load()
    table = json.loads((RESULTS / "forecast_seeds.json").read_text())
    registry = checkpoint_registry()
    with closing(sqlite3.connect(ARCHIVE.resolve().as_uri()+"?mode=ro", uri=True)) as database:
        stored = {k: json.loads(v) for k, v in database.execute("SELECT key,value FROM metadata")}
    if stored["identity"] != archive_identity(registry, table):
        raise ValueError("Forecast generation identity changed")
    metadata, body, paths = get_cell(stored["default_id"], True)
    recipe = next(row for row in recipes(arrays, manifest, registry, table) if cell_id(row[0], row[1], row[2], row[3], row[4]) == metadata["id"])
    record, group, origin, bank, convention, horizon, seeds = recipe
    model, checkpoint = load_checkpoint(record)
    rerun, regenerated = generate_cell(model, checkpoint, arrays, group, origin, seeds, horizon=horizon)
    rerun["provenance"]["source_sha256"] = manifest["source_sha256"]
    if rerun != body:
        raise ValueError("Fresh-process selected forecast does not reproduce exactly")
    for name in paths:
        np.testing.assert_array_equal(paths[name], regenerated[name])
    checked, selected_rounds, support_changes = 0, 0, 0
    for meta, saved, trajectories in iter_cells():
        h = meta["horizon"]
        for horizon_key, draws in saved["endpoints"].items():
            endpoint = int(horizon_key)
            wanted = np.stack((trajectories["pool_after"][:, endpoint-1]/200, trajectories["window_surplus"][:, endpoint-1]/(200*endpoint)), axis=-1)
            np.testing.assert_array_equal(draws, wanted)
        if trajectories["pool_after"].shape != (64, h) or np.any(trajectories["pool_after"] < 0) or np.any(trajectories["pool_after"] > 200):
            raise ValueError("Forecast pool bounds or branch count differ")
        if np.any(np.diff(trajectories["window_surplus"], axis=1) < -1e-10):
            raise ValueError("Cumulative retained resources decreased")
        for i in range(64):
            zero = np.flatnonzero(trajectories["pool_after"][i] == 0)
            if len(zero):
                first = int(zero[0])
                if np.any(trajectories["pool_after"][i, first:] != 0) or np.any(trajectories["window_surplus"][i, first:] != trajectories["window_surplus"][i, first]):
                    raise ValueError("Exact-zero absorbing padding violated")
        for branch in (saved["selected_branch"], saved["branch_zero"]):
            for row in branch["rounds"]:
                result = step(row["pool_before"], row["offers"], row["contributions"], integer_contributions=True)
                if result.next_pool != row["pool_after"] or list(result.surplus) != row["surplus"]:
                    raise ValueError("Stored illustrated branch violates accounting")
                if any(c > math.floor(o) or c < 0 for c, o in zip(row["contributions"], row["offers"])):
                    raise ValueError("Stored illustrated branch violates integer support")
                selected_rounds += 1
        checked += 1
    if checked != 2358:
        raise ValueError("Incorrect total cells")
    receipt = {"cells_checked": checked, "branches_checked": checked*64, "illustrated_rounds_checked": selected_rounds,
               "fresh_process_regenerated_id": metadata["id"], "fresh_process_branches_exact": 64,
               "archive_sha256": sha256(ARCHIVE), "selected_branch_seed": metadata["rollout_seed"],
               "selected_branch_index": metadata["selected_branch_index"], "source_key": metadata["key"],
               "displayed_branch_index": 0, "displayed_branch_seed": body["branch_zero"]["rollout_seed"],
               "recorded_first_pool": body["branch_zero"]["rounds"][0]["pool_before"],
               "first_forecast_offers": body["branch_zero"]["rounds"][0]["offers"],
               "first_forecast_returns": body["branch_zero"]["rounds"][0]["contributions"]}
    write_json(RESULTS / "forecast_verification.json", receipt)
    print(packed(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "benchmark", "generate", "verify"))
    command = parser.parse_args().command
    action = {"prepare": prepare, "audit": prepare, "benchmark": benchmark, "generate": generate, "verify": verify}[command]
    # Shared with the fixed Task 03 and Task 04 trainers. Keep the complete
    # memory-heavy invocation serial, including lazy Torch import and loading.
    lock_path = ROOT / "data/cache/task03/training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        action()


if __name__ == "__main__":
    main()
