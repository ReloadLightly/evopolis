#!/usr/bin/env python3
"""Independent, standard-library check of the notebook's Figure 2A estimands.

This streams the source CSV; it deliberately does not import EvoPolis analysis.
The formulas follow the released notebook at revision
4f1a99a9d150f9fa6bad1a0f70f11c0673d46763, whose implementation is licensed
Apache-2.0, Copyright 2024 DeepMind Technologies Limited. This checker is an
independent implementation of the described statistical definitions.
"""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MECHANISMS = (
    "Equal Baseline",
    "Mixed Baseline",
    "Proportional Baseline",
    "RL Agent (M1)",
)
COHORTS = ("BC 1", "Exp 1")


def ranksums(a, b):
    """SciPy ranksums convention: average tie ranks, no tie correction."""
    pooled = sorted((value, sample) for sample, values in enumerate((a, b)) for value in values)
    rank_sum = 0.0
    start = 0
    while start < len(pooled):
        stop = start + 1
        while stop < len(pooled) and pooled[stop][0] == pooled[start][0]:
            stop += 1
        average_rank = (start + 1 + stop) / 2
        rank_sum += average_rank * sum(sample == 0 for _, sample in pooled[start:stop])
        start = stop
    n1, n2 = len(a), len(b)
    z = (rank_sum - n1 * (n1 + n2 + 1) / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return {"z": z, "p_two_sided": math.erfc(abs(z) / math.sqrt(2))}


def compare_outputs(result, game_values, directory):
    """Check independent values against the reproduction's machine-readable files."""
    def rows(filename):
        with (directory / filename).open(newline="") as source:
            return list(csv.DictReader(source))

    largest_difference = 0.0
    checked_numbers = 0

    def equal(actual, expected, context):
        nonlocal largest_difference, checked_numbers
        difference = abs(float(actual) - expected)
        if not math.isfinite(difference) or difference > 1e-12:
            raise ValueError(f"Numerical disagreement for {context}: {actual} vs {expected}")
        largest_difference = max(largest_difference, difference)
        checked_numbers += 1

    expected_summary = {(r["cohort"], r["mechanism"]): r for r in result["conditions"]}
    actual_summary = rows("figure2a_summary.csv")
    assert len(actual_summary) == len(expected_summary)
    assert {(r["cohort"], r["mechanism"]) for r in actual_summary} == set(expected_summary)
    field_map = {
        "groups": "groups",
        "surplus_mean": "mean_surplus",
        "gini_mean": "mean_gini",
        "surplus_ci95_halfwidth": "surplus_95ci_halfwidth_notebook_ddof0",
        "gini_ci95_halfwidth": "gini_95ci_halfwidth_notebook_ddof0",
    }
    for actual in actual_summary:
        key = (actual["cohort"], actual["mechanism"])
        expected = expected_summary[key]
        for actual_field, expected_field in field_map.items():
            equal(actual[actual_field], expected[expected_field], (key, actual_field))

    expected_tests = {(r["cohort"], r["first"], r["second"], r["metric"]): r for r in result["rank_sum_tests"]}
    actual_tests = rows("rank_tests.csv")
    assert len(actual_tests) == len(expected_tests)
    assert {(r["cohort"], r["a"], r["b"], r["metric"]) for r in actual_tests} == set(expected_tests)
    for actual in actual_tests:
        key = (actual["cohort"], actual["a"], actual["b"], actual["metric"])
        for metric in ("z", "p_two_sided"):
            equal(actual[metric], expected_tests[key][metric], (key, metric))

    actual_groups = rows("figure2a_groups.csv")
    assert len(actual_groups) == len(game_values)
    assert {(r["cohort"], r["mechanism"], r["launch_id"]) for r in actual_groups} == set(game_values)
    for actual in actual_groups:
        key = (actual["cohort"], actual["mechanism"], actual["launch_id"])
        for index, metric in enumerate(("surplus", "gini")):
            equal(actual[metric], game_values[key][index], (key, metric))
    return {
        "directory": str(directory), "groups_compared": len(actual_groups),
        "conditions_compared": len(actual_summary), "rank_tests_compared": len(actual_tests),
        "numbers_compared": checked_numbers, "absolute_tolerance": 1e-12,
        "maximum_absolute_difference": largest_difference,
    }


def check(path, compare_dir=None):
    selected = {f"{m} {c}": (c, m) for c in COHORTS for m in MECHANISMS}
    games = defaultdict(lambda: [[], [], [], [], set()])
    total_rows = 0
    missing_rewards = 0
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            total_rows += 1
            label = row["mech_name_by_player"]
            if label not in selected:
                continue
            game = games[(label, row["launch_id"])]
            for player in range(4):
                value = row[f"player_reward_{player}"]
                if value == "":
                    missing_rewards += 1
                else:
                    game[player].append(float(value))
            game[4].add(int(row["round_id"]))
    grouped = defaultdict(list)
    game_values = {}
    for (label, launch), game in games.items():
        assert game[4] == set(range(40)), (label, launch, sorted(game[4]))
        assert all(len(rewards) == 40 for rewards in game[:4]), (label, launch)
        means = [statistics.mean(rewards) for rewards in game[:4]]
        surplus = statistics.mean(means)
        gini = math.fsum(abs(a - b) for a in means for b in means) / (32 * surplus) if surplus else math.nan
        grouped[selected[label]].append((surplus, gini))
        game_values[(*selected[label], launch)] = (surplus, gini)
    result = {"source_rows": total_rows, "missing_selected_rewards": missing_rewards, "conditions": [], "rank_sum_tests": []}
    for cohort in COHORTS:
        for mechanism in MECHANISMS:
            values = grouped[(cohort, mechanism)]
            surplus, gini = map(list, zip(*values))
            result["conditions"].append({
                "cohort": cohort, "mechanism": mechanism,
                "groups": len(values), "rows": len(values) * 40,
                "mean_surplus": statistics.mean(surplus),
                "mean_gini": statistics.mean(gini),
                "zero_surplus_games": sum(s == 0 for s in surplus),
                "min_surplus": min(surplus),
                "surplus_95ci_halfwidth_notebook_ddof0": 1.96 * statistics.pstdev(surplus) / math.sqrt(len(values)),
                "gini_95ci_halfwidth_notebook_ddof0": 1.96 * statistics.pstdev(gini) / math.sqrt(len(values)),
            })
    for cohort in COHORTS:
        for first, second in itertools.combinations(MECHANISMS, 2):
            for index, metric in enumerate(("surplus", "gini")):
                a = [v[index] for v in grouped[(cohort, first)]]
                b = [v[index] for v in grouped[(cohort, second)]]
                result["rank_sum_tests"].append({"cohort": cohort, "first": first, "second": second, "metric": metric, **ranksums(a, b)})
    if compare_dir is not None:
        result["verification"] = compare_outputs(result, game_values, compare_dir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="data/raw/sustainable_behavior.csv")
    parser.add_argument("--compare-dir", type=Path, help="Compare the reproduction's three numerical CSV files to 1e-12 absolute tolerance")
    args = parser.parse_args()
    print(json.dumps(check(args.csv, args.compare_dir), indent=2, allow_nan=False))
