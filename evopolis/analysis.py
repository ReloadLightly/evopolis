"""Streaming Figure 2A reproduction; all outcomes are recorded upstream results.

Aggregation and Gini adapt the DeepMind notebook (Copyright 2024 DeepMind
Technologies Limited), Apache-2.0; see NOTICE and LICENSES/Apache-2.0.txt.
"""

from collections import defaultdict
import csv
from itertools import combinations
import math
from pathlib import Path

import numpy as np
from scipy.stats import ranksums

MECHANISMS = ("Equal Baseline", "Mixed Baseline", "Proportional Baseline", "RL Agent (M1)")
COHORTS = ("BC 1", "Exp 1")


def gini(values) -> float:
    """Notebook's uncorrected Gini; zero total has undefined (NaN) inequality."""
    x = np.asarray(values, dtype=float)
    if x.sum() == 0:
        return math.nan
    return float(np.abs(np.subtract.outer(x, x)).mean() / (2 * x.mean()))


def mean_conf(values) -> tuple[float, float]:
    """Notebook helper's normal 95% half-width, applied across games (ddof=0)."""
    x = np.asarray(values, dtype=float)
    return float(x.mean()), float(1.96 * x.std(ddof=0) / math.sqrt(len(x)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure_groups(path: Path) -> list[dict]:
    selected = {f"{m} {c}": (c, m) for c in COHORTS for m in MECHANISMS}
    # Only four rewards per round for the selected cohorts, about 3 MiB of floats.
    rewards = defaultdict(lambda: [[], [], [], []])
    rounds = defaultdict(set)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["mech_name_by_player"]
            if label not in selected:
                continue
            cohort, mechanism = selected[label]
            key = (cohort, mechanism, row["launch_id"])
            round_id = int(row["round_id"])
            if round_id in rounds[key]:
                raise ValueError(f"Duplicate group-round: {key} {round_id}")
            rounds[key].add(round_id)
            for i in range(4):
                value = float(row[f"player_reward_{i}"])
                if not math.isfinite(value):
                    raise ValueError(f"Missing/nonfinite reward: {key} {round_id} {i}")
                rewards[key][i].append(value)
    output = []
    for (cohort, mechanism, launch), player_rewards in sorted(rewards.items()):
        key = (cohort, mechanism, launch)
        if rounds[key] != set(range(40)):
            raise ValueError(f"Figure 2A requires the full 0–39 horizon: {key}")
        means = [math.fsum(x) / len(x) for x in player_rewards]
        output.append({
            "cohort": cohort, "mechanism": mechanism, "launch_id": launch,
            "rounds": 40, "surplus": math.fsum(means) / 4, "gini": gini(means),
            **{f"player_mean_{i}": value for i, value in enumerate(means)},
        })
    for cohort in COHORTS:
        for mechanism in MECHANISMS:
            n = sum(r["cohort"] == cohort and r["mechanism"] == mechanism for r in output)
            expected = 512 if cohort == "BC 1" else 40
            if n != expected:
                raise ValueError(f"Unexpected group count {cohort} {mechanism}: {n}")
    return output


def summarize(groups: list[dict]) -> list[dict]:
    summary = []
    for cohort in COHORTS:
        for mechanism in MECHANISMS:
            rows = [r for r in groups if r["cohort"] == cohort and r["mechanism"] == mechanism]
            surplus, surplus_ci = mean_conf([r["surplus"] for r in rows])
            inequality, gini_ci = mean_conf([r["gini"] for r in rows])
            summary.append({
                "cohort": cohort, "mechanism": mechanism, "groups": len(rows),
                "surplus_mean": surplus, "surplus_ci95_halfwidth": surplus_ci,
                "gini_mean": inequality, "gini_ci95_halfwidth": gini_ci,
                "undefined_ginis": sum(math.isnan(r["gini"]) for r in rows),
            })
    return summary


def rank_tests(groups: list[dict]) -> list[dict]:
    output = []
    for cohort in COHORTS:
        for a, b in combinations(MECHANISMS, 2):
            for metric in ("surplus", "gini"):
                x, y = ([r[metric] for r in groups if r["cohort"] == cohort and r["mechanism"] == m] for m in (a, b))
                result = ranksums(x, y)
                output.append({"cohort": cohort, "a": a, "b": b, "metric": metric,
                               "z": float(result.statistic), "p_two_sided": float(result.pvalue)})
    return output


def table_markdown(summary: list[dict]) -> str:
    lines = ["| Evidence | Mechanism | Groups | Mean surplus (95% CI) | Mean Gini (95% CI) |",
             "| :--- | :--- | ---: | ---: | ---: |"]
    for r in summary:
        s, e = r["surplus_mean"], r["surplus_ci95_halfwidth"]
        g, h = r["gini_mean"], r["gini_ci95_halfwidth"]
        lines.append(f"| {r['cohort']} | {r['mechanism']} | {r['groups']} | {s:.3f} [{s-e:.3f}, {s+e:.3f}] | {g:.3f} [{g-h:.3f}, {g+h:.3f}] |")
    return "\n".join(lines) + "\n"
