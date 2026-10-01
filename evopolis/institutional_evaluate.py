"""Task 06 institution benchmarks and preregistered transfer scoring.

This module never imports torch. All simulated outcomes come from frozen cell
JSON files. The Experiment 2 command opens human rows only through the pushed
freeze gate in institutional_data. Group bootstrap uncertainty holds simulation
means fixed; independent rollout Monte Carlo uncertainty is reported separately.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import norm

from .behavior_data import ROOT, content_hash, write_json
from .institutional_data import game_summaries, load_exp2, replay_allocations
from .institutional_metrics import (energy_distance, institution_error, interval,
                                    outcome_vectors, paired_error_comparison,
                                    stratified_bootstrap_indices)
from .sources import sha256


RESULTS = ROOT / "results/task06"
RULES = ("Equal", "Mixed", "Proportional")
ALL_RULES = (*RULES, "Interpolating")
SEEDS = (17, 29, 43)
METRICS = ("surplus", "survival", "gini", "pool20", "pool40")


def read_recorded(results_dir: Path = RESULTS) -> dict[str, dict[str, list[dict]]]:
    saved = json.loads((results_dir / "recorded_games.json").read_text())
    grouped = defaultdict(lambda: defaultdict(list))
    for row in saved["games"]:
        grouped[row["cohort"]][row["mechanism"]].append(row)
    return {cohort: dict(rules) for cohort, rules in grouped.items()}


def load_rollouts(results_dir: Path = RESULTS, *, references_only=False) -> tuple[dict, dict, dict]:
    """Validate complete declared production cells without loading any weights."""
    config = json.loads((ROOT / "configs/task06.json").read_text())
    families = list(config["reference_families"])
    if not references_only:
        families += config["fa_families"] + ["CL-" + f for f in config["calibration_families"]]
    pooled, seeded, hashes = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(dict)), {}
    count = config["rollout"]["games_per_checkpoint_rule"]
    for family in families:
        for seed in SEEDS:
            for rule in ALL_RULES:
                path = results_dir / "rollouts" / f"{family}_{seed}_{rule}.json"
                saved = json.loads(path.read_text())
                contract, games = saved["contract"], saved["games"]
                checkpoint = contract["checkpoint"]
                if (checkpoint["family"], checkpoint["seed"], contract["rule"], contract["count"], contract["namespace"]) != (family, seed, rule, count, config["rollout_namespace"]):
                    raise ValueError(f"Rollout contract differs from frozen design: {path}")
                if len(games) != count or [g["game"] for g in games] != list(range(count)):
                    raise ValueError(f"Incomplete/duplicate rollout game indices: {path}")
                if [g["rollout_seed"] for g in games] != contract["seeds"] or len(set(contract["seeds"])) != count:
                    raise ValueError(f"Rollout seed table differs from game summaries: {path}")
                if sha256(path.with_suffix(".npz")) != saved["trajectory_sha256"]:
                    raise ValueError(f"Rollout retained trajectories changed: {path}")
                augmented = [dict(row, family=family, training_seed=seed, mechanism=rule) for row in games]
                seeded[family][seed][rule] = augmented
                pooled[family][rule].extend(augmented)
                hashes[str(path.relative_to(ROOT))] = sha256(path)
    return ({f: dict(v) for f, v in pooled.items()},
            {f: dict(v) for f, v in seeded.items()}, hashes)


def _values(rows, metric):
    return np.asarray([float(r[metric]) for r in rows if r[metric] is not None and math.isfinite(float(r[metric]))])


def mc_se(rows, metric) -> float | None:
    """MC error of an equal-size seed mixture; weights are fixed, not refitted."""
    strata = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            strata[row.get("training_seed", "recorded")].append(float(row[metric]))
    total = sum(len(values) for values in strata.values())
    if total < 2:
        return None
    variance = 0.
    for values in strata.values():
        if len(values) > 1:
            variance += (len(values) / total) ** 2 * np.var(values, ddof=1) / len(values)
    return float(np.sqrt(variance))


def metric_summary(rows, *, simulated=False, bootstrap_indices=None) -> dict:
    result = {"groups" if not simulated else "games": len(rows)}
    for metric in METRICS:
        values = _values(rows, metric)
        value = float(values.mean()) if len(values) else None
        item = {"mean": value, "defined_games": len(values)}
        if simulated:
            item["mc_se"] = mc_se(rows, metric)
        elif bootstrap_indices is not None and len(values) == len(rows):
            item["human_group_ci95"] = interval(values[bootstrap_indices].mean(axis=1))
        result[metric] = item
    return result


def _means(by_rule, metric):
    return {rule: float(_values(rows, metric).mean()) for rule, rows in by_rule.items()}


def _rank_comparison(human, simulation, metric):
    human_means, simulated_means = _means(human, metric), _means(simulation, metric)
    ordering = sorted(RULES, key=lambda rule: (human_means[rule], rule))
    human_strict = all(human_means[a] < human_means[b] for a, b in zip(ordering, ordering[1:]))
    matched = human_strict and all(simulated_means[a] < simulated_means[b] for a, b in zip(ordering, ordering[1:]))
    return {"human_order_low_to_high": ordering, "human_strict": human_strict,
            "matches_strict_human_order": matched,
            "human_means": {r: human_means[r] for r in RULES},
            "simulated_means": {r: simulated_means[r] for r in RULES}}


def _human_population(human, population):
    include = {"all": {"train", "validation", "test"},
               "train_validation": {"train", "validation"}, "validation": {"validation"}}[population]
    return {rule: [row for row in human[rule] if row["split"] in include] for rule in RULES}


def _score_rule(human, simulated, indices):
    observation = metric_summary(human, bootstrap_indices=indices)
    prediction = metric_summary(simulated, simulated=True)
    return {"observed": observation, "predicted": prediction,
            "errors": {metric: {"signed": prediction[metric]["mean"] - observation[metric]["mean"],
                                "absolute": abs(prediction[metric]["mean"] - observation[metric]["mean"])}
                       for metric in METRICS if observation[metric]["mean"] is not None and prediction[metric]["mean"] is not None},
            "energy_u": energy_distance(outcome_vectors(simulated), outcome_vectors(human))}


def headroom_decision(human, simulators, config) -> dict:
    candidates = config["fa_families"] + ["CL-" + name for name in config["calibration_families"]]
    missing = sorted(set(candidates) - set(simulators))
    banks, rows, constraints = {}, [], []
    for population in ("train_validation", "validation"):
        target = _human_population(human, population)
        banks[population] = (target, stratified_bootstrap_indices(target))
        metrics = {}
        for metric in ("surplus", "survival"):
            rank = _rank_comparison(target, target, metric)
            means = rank["human_means"]
            tied_pairs = [[a, b] for i, a in enumerate(RULES) for b in RULES[i + 1:] if means[a] == means[b]]
            metrics[metric] = {"human_means": means, "human_order_low_to_high": rank["human_order_low_to_high"],
                               "human_strict": rank["human_strict"], "tied_rule_pairs": tied_pairs}
        feasible = all(value["human_strict"] for value in metrics.values())
        constraints.append({"population": population, "human_groups_per_rule": len(target["Equal"]),
                            "metrics": metrics, "strict_order_condition_feasible": feasible,
                            "affected_candidates": [family for family in candidates if family.startswith("CL-") == (population == "validation")],
                            "interpretation": "Both observed human rankings are strict, so a candidate can satisfy the ranking condition."
                            if feasible else "An observed human outcome ranking contains a tie. Under the stipulated strict-order decision rule, no candidate assessed on this population can qualify, regardless of forecast accuracy or IE. This constraint is not evidence that calibrated behavior is inaccurate."})
    if missing:
        return {"status": "pending", "missing_candidates": missing, "human_population_constraints": constraints}
    for family in candidates:
        population = "validation" if family.startswith("CL-") else "train_validation"
        target, indices = banks[population]
        comparisons = {}
        for metric in ("surplus", "survival"):
            obs = {rule: _values(target[rule], metric) for rule in RULES}
            sim = {rule: _values(simulators[family][rule], metric) for rule in RULES}
            bc1 = {rule: _values(simulators["BC1"][rule], metric) for rule in RULES}
            candidate_ie = institution_error(obs, sim, indices=indices)
            incumbent_ie = institution_error(obs, bc1, indices=indices)
            human_boot = {rule: obs[rule][indices[rule]].mean(axis=1) for rule in RULES}
            delta = np.stack([np.abs(sim[rule].mean() - human_boot[rule]) - np.abs(bc1[rule].mean() - human_boot[rule])
                              for rule in RULES]).mean(axis=0)
            comparisons[metric] = {"candidate_ie": candidate_ie, "bc1_ie": incumbent_ie,
                                   "paired_ie_difference": candidate_ie["estimate"] - incumbent_ie["estimate"],
                                   "paired_ie_difference_ci95": interval(delta),
                                   "ordering": _rank_comparison(target, simulators[family], metric)}
        qualifies = (comparisons["surplus"]["ordering"]["matches_strict_human_order"]
                     and comparisons["survival"]["ordering"]["matches_strict_human_order"]
                     and comparisons["surplus"]["candidate_ie"]["estimate"] <= comparisons["surplus"]["bc1_ie"]["estimate"])
        rows.append({"family": family, "population": population, "human_groups_per_rule": len(target["Equal"]),
                     "comparisons": comparisons, "qualifies_no_headroom": qualifies,
                     "strict_order_condition_feasible": next(c["strict_order_condition_feasible"] for c in constraints if c["population"] == population)})
    winners = [row["family"] for row in rows if row["qualifies_no_headroom"]]
    return {"status": "complete", "outcome": "no headroom" if winners else "headroom", "qualifying_candidates": winners,
            "candidates": rows, "rule": config["D1"], "bootstrap_seed": config["bootstrap_seed"],
            "bootstrap_replicates": config["bootstrap_replicates"],
            "human_population_constraints": constraints,
            "ordering_note": "Each comparison follows the actual human mean ordering; any human tie fails the strict-order condition."}


def benchmark(*, results_dir=RESULTS, references_only=False):
    start = time.perf_counter()
    config = json.loads((ROOT / "configs/task06.json").read_text())
    recorded = read_recorded(results_dir)
    simulators, by_seed, hashes = load_rollouts(results_dir, references_only=references_only)
    simulators["BC1"] = recorded["BC1"]
    simulators["BC2"] = recorded["BC2"]
    populations = {}
    for population in ("all", "train_validation"):
        human = _human_population(recorded["Exp1"], population)
        indices = stratified_bootstrap_indices(human)
        rows = []
        for family, cells in simulators.items():
            for rule in RULES:
                if rule in cells:
                    rows.append({"family": family, "rule": rule, **_score_rule(human[rule], cells[rule], indices[rule])})
        rankings = {family: {metric: _rank_comparison(human, cells, metric) for metric in ("surplus", "survival")}
                    for family, cells in simulators.items() if all(rule in cells for rule in RULES)}
        populations[population] = {"description": "All 40 groups per rule, including opened test groups" if population == "all"
                                   else "32 training+validation groups per rule; opened test groups excluded",
                                   "rows": rows, "ordering": rankings}
    output = {"schema_version": 1, "populations": populations,
              "recorded_descriptive": {cohort: {rule: metric_summary(rows, simulated=True) for rule, rows in cells.items()}
                                       for cohort, cells in recorded.items() if cohort != "Exp1"},
              "per_seed": {family: {str(seed): {rule: metric_summary(rows, simulated=True) for rule, rows in cells.items()}
                                     for seed, cells in seeds.items()} for family, seeds in by_seed.items()},
              "forecast_summary_sha256": hashes,
              "mc_definition": "Fixed seed weights; within-checkpoint variance/n. Training-start differences reported separately.",
              "energy_definition": "Off-diagonal U statistic; negative finite-sample values retained",
              "recorded_terminal_pool_limit": "BC1/BC2 pool40 and survival use published-equation inference; final next-pool field is absent."}
    decision = headroom_decision(recorded["Exp1"], simulators, config)
    write_json(results_dir / "benchmark.json", output)
    write_json(results_dir / "headroom_decision.json", decision)
    runtime = {"phase": "institution_benchmark", "elapsed_seconds": time.perf_counter() - start,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "references_only": references_only}
    write_json(results_dir / "benchmark_runtime.json", runtime)
    return {**runtime, "D1": decision.get("outcome", decision["status"]), "families": len(simulators)}


def archived_gru_agreement(*, results_dir=RESULTS):
    """Compare independent Task 06 draws with the archived Task 03 GRU sample."""
    archived = defaultdict(lambda: defaultdict(list))
    with (ROOT / "results/task03/generated_groups.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["family"] != "recurrent":
                continue
            row["training_seed"] = int(row["training_seed"])
            archived[row["training_seed"]][row["mechanism"]].append({
                "training_seed": row["training_seed"], "surplus": float(row["surplus"]),
                "gini": float(row["gini"]) if row["gini"] else None,
                "survival": int(row["final_sustainment"] == "True"), "pool40": float(row["final_pool"])})
    comparisons = []
    for rule in ALL_RULES:
        current, past = [], []
        for seed in SEEDS:
            path = results_dir / "rollouts" / f"T03-GRU_{seed}_{rule}.json"
            cell = json.loads(path.read_text())
            current.extend(dict(r, training_seed=seed) for r in cell["games"])
            past.extend(archived[seed][rule])
        for metric in ("surplus", "survival", "gini", "pool40"):
            difference = float(_values(current, metric).mean() - _values(past, metric).mean())
            se = math.hypot(mc_se(current, metric), mc_se(past, metric))
            comparisons.append({"rule": rule, "metric": metric, "archived_games": len(past), "new_games": len(current),
                                "archived_mean": float(_values(past, metric).mean()),
                                "new_mean": float(_values(current, metric).mean()), "difference": difference,
                                "independent_difference_mc_se": se,
                                "z": difference / se if se > 0 else (0. if difference == 0 else None)})
    cutoff = float(norm.ppf(1 - .01 / (2 * len(comparisons))))
    for row in comparisons:
        row["within_familywise_99pct_mc_envelope"] = abs(row["difference"]) <= cutoff * row["independent_difference_mc_se"]
    output = {"comparisons": comparisons, "normal_cutoff": cutoff,
              "passed": all(row["within_familywise_99pct_mc_envelope"] for row in comparisons),
              "interpretation": "Independent-sample mean agreement; conservative normal approximation with Bonferroni 99% familywise MC envelope over16 summaries. This is a Monte Carlo diagnostic, not a behavioral-model selection criterion.",
              "archived_sha256": sha256(ROOT / "results/task03/generated_groups.csv")}
    write_json(results_dir / "archived_gru_agreement.json", output)
    return output


def _transfer_scores(human, simulated, incumbent, indices):
    scored = {}
    for metric in ("surplus", "survival"):
        observations = {rule: _values(human[rule], metric) for rule in ("Interpolating", "Proportional")}
        forecasts = {rule: _values(simulated[rule], metric) for rule in observations}
        references = {rule: _values(incumbent[rule], metric) for rule in observations}
        scored[metric] = paired_error_comparison(observations, forecasts, references, indices=indices)
        for estimand, rules in (("E1", ("Interpolating", "Proportional")), ("E2", ("Interpolating",))):
            scored[metric][estimand]["prediction_mc_se"] = math.sqrt(sum(mc_se(simulated[r], metric) ** 2 for r in rules))
            scored[metric][estimand]["incumbent_mc_se"] = math.sqrt(sum(mc_se(incumbent[r], metric) ** 2 for r in rules))
    scored["per_rule"] = {rule: _score_rule(human[rule], simulated[rule], indices[rule])
                          for rule in ("Interpolating", "Proportional")}
    return scored


def transfer(*, results_dir=RESULTS):
    """Open and score Exp2 only after remote verification of the freeze commit."""
    start = time.perf_counter()
    # This call validates pushed commit and analysis-code hashes before parsing.
    cohort = load_exp2()
    human = defaultdict(list)
    for row in game_summaries(cohort):
        human[row["mechanism"]].append(row)
    write_json(results_dir / "experiment2_games.json", {"audit": cohort.audit, "games": game_summaries(cohort)})
    write_json(results_dir / "experiment2_replay.json", replay_allocations(cohort, rules=("Proportional", "Interpolating")))
    recorded = read_recorded(results_dir)
    simulators, by_seed, hashes = load_rollouts(results_dir)
    simulators["BC1"] = recorded["BC1"]
    training = json.loads((results_dir / "training_complete.json").read_text())
    fa_best = training["fa_best"]
    primary = ("BC1", fa_best, "CL-" + fa_best)
    if any(family not in simulators for family in primary):
        raise ValueError("Selected primary simulator has no complete frozen forecasts")
    target = {rule: human[rule] for rule in ("Interpolating", "Proportional")}
    indices = stratified_bootstrap_indices(target)
    scores = {family: {"role": "primary" if family in primary else "named secondary" if family in ("T03-GRU", "T03-constant") else "secondary",
                       **_transfer_scores(target, cells, simulators["BC1"], indices)} for family, cells in simulators.items()}
    per_seed = {family: {str(seed): _transfer_scores(target, cells, simulators["BC1"], indices)
                         for seed, cells in seeds.items()} for family, seeds in by_seed.items()}
    shift = {}
    for rule in ("Proportional", "M1"):
        old, new = recorded["Exp1"][rule], human[rule]
        shift[rule] = {"Exp1": metric_summary(old), "Exp2": metric_summary(new),
                       "Exp2_minus_Exp1": {metric: float(_values(new, metric).mean() - _values(old, metric).mean()) for metric in METRICS},
                       "label": "Descriptive cohort plus instruction shift; different groups"}
    d1 = json.loads((results_dir / "headroom_decision.json").read_text())["outcome"]
    d2 = {family: {estimand: scores[family]["surplus"][estimand]["classification"] for estimand in ("E1", "E2")} for family in primary}
    worse = any(value == "worse" for family, values in d2.items() if family != "BC1" for value in values.values())
    if d1 == "headroom":
        recommendation = "Specify Task07 behavioral-program search with matched-budget random control and a frozen Experiment3 test; do not run search."
    elif worse:
        recommendation = "Transfer failure: investigate instruction and cohort shift; no evolutionary search."
    else:
        recommendation = "Simple ingredient contribution: no evolutionary search; sketch the empirical paper."
    output = {"schema_version": 1, "freeze_commit": cohort.audit["freeze_commit"], "primary_simulators": list(primary),
              "fa_best": fa_best, "human": {rule: metric_summary(rows, bootstrap_indices=indices.get(rule)) for rule, rows in human.items()},
              "scores": scores, "per_seed": per_seed, "cohort_shift": shift, "D1": d1, "D2": d2,
              "recommendation": recommendation, "forecast_summary_sha256": hashes,
              "human_bootstrap_index_sha256": content_hash({r: i.tolist() for r, i in indices.items()}),
              "comparison": "Candidate minus BC1 absolute error on the identical2000 human-group resamples, stratified by rule; forecast means fixed.",
              "mc_definition": "Separate Monte Carlo SE from within-checkpoint variance and fixed equal seed weights; no seed resampling.",
              "BC2_descriptive": {rule: metric_summary(rows, simulated=True) for rule, rows in recorded["BC2"].items()},
              "limitations": {"instructions": "Exp2 participants received rule instructions; Exp1 participants did not; predictors have no instruction input.",
                              "effect": "Within-Exp2 Interpolating-minus-Proportional difference cancels shared shifts only to first order.",
                              "BC2": "Descriptive only: its training includes earlier experimental evidence.",
                              "recorded_BC_pool40": "Equation-inferred; absent terminal next-pool fields.",
                              "M1": "Recorded teacher-forced offers only; no executable policy."}}
    write_json(results_dir / "transfer.json", output)
    runtime = {"phase": "transfer_scoring", "elapsed_seconds": time.perf_counter() - start,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    write_json(results_dir / "transfer_runtime.json", runtime)
    return {**runtime, "D1": d1, "D2": d2, "recommendation": recommendation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "archived-check", "transfer"))
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    if args.command == "benchmark":
        output = benchmark(references_only=args.references_only)
    elif args.command == "archived-check":
        output = archived_gru_agreement()
    else:
        output = transfer()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
