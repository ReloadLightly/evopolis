"""Task 04 fixed-budget prediction and collective-forecast diagnostics.

Every score is computed per fitted seed and human group. Seeds are averaged
within human groups before mechanism weighting or paired group uncertainty.
This follow-up reuses opened Experiment 1 evidence, never Experiments 2–3.
"""

import argparse
from collections import defaultdict
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np

from .behavior_data import ROOT, load, write_json
from .behavior_evaluate import write_csv
from .sources import sha256

RESULTS = ROOT / "results/task04"
FIGURES = ROOT / "docs/assets"
MECHANISMS = ("Equal", "Mixed", "Proportional")
FAMILIES = ("constant", "linear", "feedforward", "recurrent")
SEEDS = (17, 29, 43)
ORIGINS = (0, 5, 10, 20)
HORIZONS = (1, 5, 10, 20)
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPLICATES = 2000


def energy_score(draws, observed):
    """Unbiased finite-ensemble Euclidean energy score (off-diagonal pairs)."""
    draws = np.asarray(draws, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if draws.ndim != 2 or len(draws) < 2 or draws.shape[1:] != observed.shape:
        raise ValueError("Expected at least two multivariate draws and one endpoint")
    if not np.isfinite(draws).all() or not np.isfinite(observed).all():
        raise ValueError("Nonfinite endpoint")
    first = np.linalg.norm(draws - observed, axis=1).mean()
    pairwise = np.linalg.norm(draws[:, None] - draws[None, :], axis=-1).sum()
    return float(first - pairwise / (2 * len(draws) * (len(draws) - 1)))


def crps(draws, observed):
    """Off-diagonal univariate energy / CRPS estimate; do not clip negatives."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 1 or len(draws) < 2:
        raise ValueError("CRPS requires at least two scalar draws")
    return energy_score(draws[:, None], np.array([observed]))


def interval_coverage(draws, observed):
    lower, upper = np.quantile(np.asarray(draws), (0.1, 0.9))
    return float(lower <= observed <= upper), float(lower), float(upper)


def convolve_returns(probabilities, maxima):
    """Exact independent sum PMF on legal supports, without FFT truncation."""
    if len(probabilities) != 4 or len(maxima) != 4:
        raise ValueError("Exactly four residents are required")
    total = np.array([1.0])
    for probability, maximum in zip(probabilities, maxima):
        maximum = int(maximum)
        probability = np.asarray(probability, dtype=np.float64)
        if maximum < 0 or maximum >= len(probability):
            raise ValueError("Invalid legal support")
        if np.any(probability < 0) or np.any(probability[maximum + 1:] != 0):
            raise ValueError("Illegal probability mass")
        if not np.isclose(probability[:maximum + 1].sum(), 1, rtol=1e-10, atol=1e-10):
            raise ValueError("Resident PMF is not normalized")
        total = np.convolve(total, probability[:maximum + 1])
    if not np.isclose(total.sum(), 1, rtol=1e-9, atol=1e-9):
        raise ValueError("Convolved PMF is not normalized")
    return total


def renewal_score(probabilities, maxima, offers, returns):
    """Recorded allocation is replaced exactly when 1.4 * sum(c) >= sum(e)."""
    distribution = convolve_returns(probabilities, maxima)
    actual = int(np.asarray(returns).sum())
    if actual < 0 or actual >= len(distribution) or distribution[actual] <= 0:
        raise ValueError("Observed total has no legal probability mass")
    allocated = math.fsum(float(value) for value in offers)
    event_probability = float(distribution[1.4 * np.arange(len(distribution)) >= allocated].sum())
    event_actual = float(1.4 * actual >= allocated)
    return {"aggregate_nll": float(-np.log(distribution[actual])),
            "renewal_brier": (event_probability - event_actual) ** 2,
            "renewal_probability": event_probability, "renewal_observed": event_actual}


def procedure(row):
    return f"{row['family']}_{row['budget']}"


def paired_group_summary(rows, metrics, *, mechanisms=MECHANISMS,
                         replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """Equal strata with unequal group counts; absent strata stay unavailable.

    One record is expected per procedure/seed/group. Exactly the same group
    resampling is applied across procedures. No pooling of model predictions.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(procedure(row), row["mechanism"], int(row["group_index"]))].append(row)
    procedures = sorted({procedure(row) for row in rows})
    if not procedures:
        return {"summary": [], "comparisons": [], "group_counts": {m: 0 for m in mechanisms},
                "unavailable_mechanisms": list(mechanisms), "all_declared_mechanisms_available": False}
    group_ids = {m: sorted({int(row["group_index"]) for row in rows if row["mechanism"] == m}) for m in mechanisms}
    available = [m for m in mechanisms if group_ids[m]]
    values = {}
    for m in available:
        values[m] = np.empty((len(procedures), len(group_ids[m]), len(metrics)))
        for p, name in enumerate(procedures):
            for g, group_id in enumerate(group_ids[m]):
                selected = grouped[(name, m, group_id)]
                if len(selected) != 3 or {int(row["seed"]) for row in selected} != set(SEEDS):
                    raise ValueError(f"Missing/duplicate fitted seeds for {name}/{m}/{group_id}")
                values[m][p, g] = [math.fsum(float(row[metric]) for row in selected) / 3 for metric in metrics]
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.zeros((replicates, len(procedures), len(metrics)))
    for m in available:
        indices = rng.integers(len(group_ids[m]), size=(replicates, len(group_ids[m])))
        bootstrap += values[m][:, indices, :].mean(axis=2).transpose(1, 0, 2) / len(available)
    overall = np.mean([values[m].mean(axis=1) for m in available], axis=0)
    summary = []
    for p, name in enumerate(procedures):
        for m in mechanisms:
            means = values[m][p].mean(axis=0) if m in available else None
            summary.append({"procedure": name, "mechanism": m, "groups": len(group_ids[m]),
                            **{metric: float(means[i]) if means is not None else None for i, metric in enumerate(metrics)}})
        entry = {"procedure": name, "mechanism": "all" if len(available) == len(mechanisms) else "available_mechanisms_only",
                 "groups": sum(map(len, group_ids.values())),
                 **{metric: float(overall[p, i]) for i, metric in enumerate(metrics)},
                 "ci95": {metric: np.quantile(bootstrap[:, p, i], (0.025, 0.975)).tolist() for i, metric in enumerate(metrics)}}
        per_seed = []
        for fitted_seed in SEEDS:
            chosen = [row for row in rows if procedure(row) == name and int(row["seed"]) == fitted_seed]
            per_seed.append({"seed": fitted_seed, **{metric: float(np.mean([
                np.mean([float(row[metric]) for row in chosen if row["mechanism"] == m]) for m in available])) for metric in metrics}})
        entry["per_seed"] = per_seed
        summary.append(entry)
    comparisons = []
    for family in ("feedforward", "recurrent"):
        names = [f"{family}_{budget}" for budget in ("original", "continued")]
        if all(name in procedures for name in names):
            old, new = (procedures.index(name) for name in names)
            comparison = {"family": family, "comparison": "continued minus original", "groups": sum(map(len, group_ids.values())),
                          "estimand": "equal mechanisms" if len(available) == len(mechanisms) else "available mechanisms only",
                          "differences": {metric: float(overall[new, i] - overall[old, i]) for i, metric in enumerate(metrics)},
                          "ci95": {metric: np.quantile(bootstrap[:, new, i] - bootstrap[:, old, i], (0.025, 0.975)).tolist() for i, metric in enumerate(metrics)},
                          "by_mechanism": {m: {metric: float((values[m][new, :, i] - values[m][old, :, i]).mean()) for i, metric in enumerate(metrics)} if m in available else None for m in mechanisms}}
            comparisons.append(comparison)
    return {"aggregation": "Three fitted-seed scores within group, groups within mechanism, mechanisms equally; no mixing forecast draws across fitted seeds",
            "summary": summary, "comparisons": comparisons, "group_counts": {m: len(group_ids[m]) for m in mechanisms},
            "unavailable_mechanisms": [m for m in mechanisms if m not in available],
            "all_declared_mechanisms_available": len(available) == len(mechanisms),
            "bootstrap": {"replicates": replicates, "seed": seed, "generator": "PCG64", "unit": "human interacting group", "paired": True, "group_ids": group_ids,
                          "scope": "Human sample uncertainty conditional on the fitted seeds and finite forecast banks"}}


def freeze_evaluation(results=RESULTS):
    """Freeze the declared diagnostic scoring before opening new comparisons."""
    from .forecast_generate import checkpoint_registry
    manifest = json.loads((ROOT / "results/task03/split.json").read_text())
    contract = {"config_sha256": sha256(ROOT / "configs/task04.json"),
                "scoring_code_sha256": sha256(Path(__file__)),
                "model_code_sha256": sha256(ROOT / "evopolis/behavior_models.py"),
                "checkpoints": checkpoint_registry(), "split_hash": manifest["manifest_hash"],
                "source_sha256": manifest["source_sha256"],
                "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    path = results / "evaluation_opening.json"
    if path.exists():
        if json.loads(path.read_text())["contract"] != contract:
            raise ValueError("Frozen Task 04 evaluation contract changed; document any correctness repair before rescoring")
    else:
        write_json(path, {"opened_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "status": "Prospective diagnostic follow-up on previously opened Experiment 1 evidence; not a new untouched test",
                         "contract": contract})
    return contract


def evaluate_observations(results=RESULTS):
    """All 18 checkpoints: original 32-group individual and 24-group joint scores."""
    import torch
    from .behavior_models import BehaviorModel, emission_log_prob, emission_probs
    from .forecast_generate import checkpoint_registry

    from .behavior_train import runtime_setup
    runtime_setup()
    freeze_evaluation(results)
    started, cpu = time.monotonic(), time.process_time()
    arrays, manifest = load()
    groups = [g for g in manifest["groups"] if g["split"] == "test"]
    registry = checkpoint_registry()
    individual, collective, calibration_rounds = [], [], []
    for spec in registry:
        path = ROOT / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError("Frozen checkpoint SHA mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = BehaviorModel(spec["family"]).eval()
        model.load_state_dict(checkpoint["model_state"])
        for offset in range(0, len(groups), 2):
            batch = groups[offset:offset + 2]
            ids = [g["index"] for g in batch]
            x = torch.tensor(arrays["x"][ids], dtype=torch.float32)
            legal = arrays["n"][ids]
            targets = arrays["y"][ids]
            with torch.no_grad():
                raw, _ = model(x.reshape(-1, 40, 9))
                raw = raw.reshape(len(batch), 4, 40, 5)
                losses = -emission_log_prob(raw, torch.as_tensor(legal), torch.as_tensor(targets)).detach().numpy()
                probabilities = emission_probs(raw, torch.as_tensor(legal)).detach().numpy()
            for index, group in enumerate(batch):
                base = {"family": spec["family"], "budget": spec["budget"], "seed": spec["seed"],
                        "selected_epoch": int(checkpoint["epoch"]), "checkpoint_sha256": spec["sha256"],
                        "group_index": group["index"], "group_key": json.dumps(group["key"], separators=(",", ":")), "mechanism": group["mechanism"]}
                mask = legal[index] >= 1
                individual.append({**base, "nonforced_choices": int(mask.sum()), "nll": float(losses[index][mask].mean())})
                if group["mechanism"] not in MECHANISMS:
                    continue
                round_scores = []
                live_rounds = np.flatnonzero(mask.any(axis=0))
                for round_id in live_rounds:
                    score = renewal_score(probabilities[index, :, round_id], legal[index, :, round_id],
                                          arrays["offers"][group["index"], :, round_id], targets[index, :, round_id])
                    round_scores.append(score)
                    calibration_rounds.append({**base, "round_id": int(round_id), "round_weight": 1 / (3 * 8 * len(live_rounds)), **score})
                collective.append({**base, "nonforced_rounds": len(live_rounds), **{
                    metric: float(np.mean([row[metric] for row in round_scores])) for metric in ("aggregate_nll", "renewal_brier", "renewal_probability", "renewal_observed")}})
        print(f"One-step evaluation: {spec['family']}/{spec['seed']}/{spec['budget']}, epoch {checkpoint['epoch']}", flush=True)
    old_rows = list(csv.DictReader((ROOT / "results/task03/prediction_groups.csv").open()))
    old_lookup = {(r["family"], int(r["seed"]), int(r["group_index"])): float(r["nll"]) for r in old_rows if r["split"] == "test" and r["family"] != "uniform"}
    discrepancies = [abs(r["nll"] - old_lookup[(r["family"], r["seed"], r["group_index"])]) for r in individual if r["budget"] == "original"]
    if max(discrepancies) > 1e-10:
        raise ValueError("Original Task 03 score changed")
    bins = []
    for name in sorted({procedure(row) for row in calibration_rounds}):
        selected = [row for row in calibration_rounds if procedure(row) == name]
        for mechanism in (*MECHANISMS, "all"):
            selected_m = [row for row in selected if mechanism == "all" or row["mechanism"] == mechanism]
            for b in range(10):
                lower, upper = b / 10, (b + 1) / 10
                records = [row for row in selected_m if lower <= row["renewal_probability"] and (row["renewal_probability"] <= upper if b == 9 else row["renewal_probability"] < upper)]
                weight = math.fsum(r["round_weight"] for r in records)
                bins.append({"procedure": name, "mechanism": mechanism, "bin": b, "lower": lower, "upper": upper,
                             "weighted_mass": weight / 3 * (3 if mechanism != "all" else 1),
                             "predicted": math.fsum(r["renewal_probability"] * r["round_weight"] for r in records) / weight if weight else None,
                             "observed": math.fsum(r["renewal_observed"] * r["round_weight"] for r in records) / weight if weight else None,
                             "group_count": len({r["group_index"] for r in records}), "round_seed_count": len(records)})
    for name in sorted({procedure(row) for row in calibration_rounds}):
        for mechanism in (*MECHANISMS, "all"):
            total_bin_mass = math.fsum(row["weighted_mass"] for row in bins if row["procedure"] == name and row["mechanism"] == mechanism)
            if not math.isclose(total_bin_mass, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Renewal calibration bin weights do not sum to one: {name}/{mechanism}")
    summary = {"status": "Diagnostic reuse of opened Experiment 1; Experiments 2–3 untouched",
               "individual_32": paired_group_summary(individual, ("nll",), mechanisms=(*MECHANISMS, "M1")),
               "individual_24": paired_group_summary([r for r in individual if r["mechanism"] in MECHANISMS], ("nll",)),
               "collective_one_step": paired_group_summary(collective, ("aggregate_nll", "renewal_brier", "renewal_probability", "renewal_observed")),
               "original_task03_max_nll_disagreement": max(discrepancies),
               "renewal_event": "1.4 * total_return >= sum(recorded current offers)",
               "probability_bin_edges": np.linspace(0, 1, 11).tolist(),
               "resource": {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "individual_groups.csv", individual)
    write_csv(results / "collective_one_step_groups.csv", collective)
    write_csv(results / "collective_calibration_rounds.csv", calibration_rounds)
    write_csv(results / "collective_calibration_bins.csv", bins)
    write_json(results / "one_step_summary.json", summary)
    return summary


def observed_endpoint(arrays, group_index, origin, horizon):
    if origin < 0 or horizon < 1 or origin + horizon > 40:
        raise ValueError("Forecast horizon exceeds the recorded experiment")
    return np.array([arrays["next_pool"][group_index, origin + horizon - 1] / 200,
                     arrays["surplus"][group_index, :, origin:origin + horizon].sum() / (200 * horizon)])


def score_cell(metadata, body, paths, arrays):
    """Independently reconstruct observed endpoints and audit all saved horizons."""
    group_index, origin = int(metadata["group_index"]), int(metadata["origin"])
    pool, surplus = np.asarray(paths["pool_after"]), np.asarray(paths["window_surplus"])
    if pool.shape != surplus.shape or pool.ndim != 2 or pool.shape[0] != 64:
        raise ValueError("Expected exactly 64 complete paths per checkpoint/group/origin")
    if np.any(pool < 0) or np.any(pool > 200) or np.any(np.diff(surplus, axis=1) < -1e-10):
        raise ValueError("Forecast physical bound or cumulative reward violation")
    if origin + pool.shape[1] > 40:
        raise ValueError("Forecast extends past the recorded episode")
    actual_pool = arrays["next_pool"][group_index, origin:origin + pool.shape[1]] / 200
    trajectory = np.array([crps(pool[:, t] / 200, actual_pool[t]) for t in range(pool.shape[1])])
    eligible = bool((arrays["offers"][group_index, :, origin] >= 1).any())
    rows = []
    for horizon in HORIZONS:
        if horizon > pool.shape[1]:
            continue
        endpoint = np.stack((pool[:, horizon - 1] / 200, surplus[:, horizon - 1] / (200 * horizon)), axis=1)
        actual = observed_endpoint(arrays, group_index, origin, horizon)
        if np.any(endpoint < 0) or np.any(endpoint > 1 + 1e-12):
            raise ValueError("Normalized endpoint exceeds the fixed physical scale")
        if not np.allclose(endpoint, body["endpoints"][str(horizon)], rtol=0, atol=1e-12):
            raise ValueError("Saved endpoint disagrees with saved path")
        if not np.allclose(actual, body["observed_endpoints"][str(horizon)], rtol=0, atol=1e-12):
            raise ValueError("Saved observed endpoint disagrees with untouched source rewards")
        row = {"cell_id": metadata["id"], "family": metadata["family"], "budget": metadata["budget"],
               "seed": int(metadata.get("seed", metadata.get("training_seed"))), "checkpoint_sha256": metadata["checkpoint_sha256"],
               "group_index": group_index, "mechanism": metadata["mechanism"], "origin": origin,
               "horizon": horizon, "bank": metadata["bank"], "convention": metadata["convention"],
               "eligible_at_origin": int(eligible), "branches": len(pool),
               "energy": energy_score(endpoint, actual),
               "pool_crps": crps(endpoint[:, 0], actual[0]), "surplus_crps": crps(endpoint[:, 1], actual[1]),
               "trajectory_pool_crps": float(trajectory[:horizon].mean()),
               "observed_pool_scaled": float(actual[0]), "observed_surplus_scaled": float(actual[1])}
        for d, label in enumerate(("pool", "surplus")):
            coverage, lower, upper = interval_coverage(endpoint[:, d], actual[d])
            row.update({f"{label}_coverage80": coverage, f"{label}_lower80": lower, f"{label}_upper80": upper,
                        f"{label}_mean": float(endpoint[:, d].mean())})
        rows.append(row)
    return rows


FORECAST_METRICS = ("energy", "pool_crps", "surplus_crps", "trajectory_pool_crps", "pool_coverage80", "surplus_coverage80")


def sensitivity_summary(rows):
    """Canonical minus recorded boundary scores, preserving paired seed banks."""
    paired = defaultdict(dict)
    for row in rows:
        if row["origin"] == 5 and row["horizon"] == 10 and row["eligible_at_origin"] and row["bank"] == "main" and row["family"] in ("feedforward", "recurrent"):
            paired[(row["family"], row["budget"], row["seed"], row["group_index"])][row["convention"]] = row
    differences = []
    for key, arms in paired.items():
        if set(arms) != {"recorded", "canonical"}:
            raise ValueError(f"Missing boundary-sensitivity arm for {key}")
        differences.append({**arms["canonical"], **{metric: arms["canonical"][metric] - arms["recorded"][metric] for metric in FORECAST_METRICS}})
    return paired_group_summary(differences, FORECAST_METRICS)


def evaluate_forecasts(results=RESULTS):
    from .forecast_generate import iter_cells

    freeze_evaluation(results)
    started, cpu = time.monotonic(), time.process_time()
    arrays, manifest = load()
    rows = []
    cell_count = 0
    branch_count = 0
    for metadata, body, paths in iter_cells(include_paths=True):
        rows.extend(score_cell(metadata, body, paths, arrays))
        cell_count += 1
        branch_count += len(paths["pool_after"])
        if cell_count % 100 == 0:
            print(f"Forecast scores: {cell_count} cells / {branch_count:,} branches", flush=True)
    if cell_count != 2358 or branch_count != 150912:
        raise ValueError(f"Incomplete forecast plan: {cell_count} cells / {branch_count} branches")
    primary_rows = [r for r in rows if r["origin"] == 5 and r["horizon"] == 10 and r["eligible_at_origin"] and r["bank"] == "main" and r["convention"] == "recorded"]
    second_rows = [r for r in rows if r["origin"] == 5 and r["horizon"] == 10 and r["eligible_at_origin"] and r["bank"] == "second" and r["convention"] == "recorded"]
    primary = paired_group_summary(primary_rows, FORECAST_METRICS)
    if primary["group_counts"] != {"Equal": 5, "Mixed": 8, "Proportional": 8}:
        raise ValueError("Unexpected primary eligibility counts")
    second = paired_group_summary(second_rows, FORECAST_METRICS)
    secondary = []
    for origin in ORIGINS:
        for horizon in HORIZONS:
            selected = [r for r in rows if r["origin"] == origin and r["horizon"] == horizon and r["bank"] == "main" and r["convention"] == "recorded"]
            for population in ("all_groups", "origin_eligible"):
                subset = selected if population == "all_groups" else [r for r in selected if r["eligible_at_origin"]]
                summary = paired_group_summary(subset, FORECAST_METRICS)
                secondary.append({"origin": origin, "horizon": horizon, "population": population, **summary})
    summary = {"primary": {"origin": 5, "horizon": 10, "population": "origin eligible", "bank": "first 64 draws", **primary},
               "second_bank": {"origin": 5, "horizon": 10, "bank": "independent second 64 draws; not pooled into primary", **second},
               "sensitivity": {"contrast": "canonical first allocation minus recorded first allocation; same branch seeds", **sensitivity_summary(rows)},
               "secondary": secondary,
               "cell_count": cell_count, "branch_count": branch_count, "score_rows": len(rows),
               "energy_estimator": "mean distance to observed minus off-diagonal pair distances / (2 M (M - 1)); no clipping",
               "marginal_crps_estimator": "One-dimensional off-diagonal energy; same unbiased pair denominator",
               "coverage": "Observed endpoint inside [0.1,0.9] empirical quantiles, NumPy linear interpolation; includes endpoints",
               "trajectory_metric": "Mean per-round marginal pool CRPS over forecast window; not a joint path score",
               "resource": {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    write_csv(results / "forecast_scores.csv", rows)
    write_json(results / "forecast_summary.json", summary)
    return summary


def _overall(summary, name, metric):
    return next(row[metric] for row in summary["summary"] if row["procedure"] == name and row["mechanism"] in ("all", "available_mechanisms_only"))


def plot_results(results=RESULTS, figures=FIGURES):
    from .forecast_plot import plot_results as draw
    return draw(results, figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("observations", "forecasts", "plot", "all"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    lock_path = ROOT / "data/cache/task03/training.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action in ("observations", "all"):
            evaluate_observations(args.results)
        if args.action in ("forecasts", "all"):
            evaluate_forecasts(args.results)
        if args.action in ("plot", "all"):
            plot_results(args.results, args.figures)



if __name__ == "__main__":
    main()
