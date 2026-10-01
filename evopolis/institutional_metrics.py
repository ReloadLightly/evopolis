"""Task 06 group-level fidelity scores and paired human-group uncertainty.

Energy distance uses the off-diagonal two-sample U statistic. It can be
negative at finite sample sizes and is deliberately not truncated at zero.
"""

from collections.abc import Mapping, Sequence
import math

import numpy as np
from scipy.spatial.distance import cdist, pdist


RULES = ("Equal", "Mixed", "Proportional")
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_REPLICATES = 2000


def _points(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("Energy distance requires at least two finite observations per sample")
    return result


def energy_distance(x, y) -> float:
    """2 E||X-Y|| - E||X-X'|| - E||Y-Y'||, excluding same-sample diagonals."""
    x, y = _points(x), _points(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Energy-distance outcome dimensions must match")
    # Blocks bound memory even for pooled (1,536-game) forecast families.
    cross_sum = math.fsum(float(cdist(x[start:start + 256], y).sum())
                          for start in range(0, len(x), 256))
    return float(2 * cross_sum / (len(x) * len(y)) - pdist(x).mean() - pdist(y).mean())


def outcome_vectors(rows: Sequence[Mapping]) -> np.ndarray:
    return np.asarray([[r["surplus"] / 10, r["pool20"] / 200, r["pool40"] / 200]
                       for r in rows], dtype=np.float64)


def summarize(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty game sample")
    result = {"games": len(rows)}
    for metric in ("surplus", "survival", "gini", "pool20", "pool40"):
        values = np.asarray([r[metric] for r in rows if r[metric] is not None], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[metric] = float(values.mean()) if len(values) else None
        result[f"{metric}_mc_se"] = (float(values.std(ddof=1) / math.sqrt(len(values)))
                                     if len(values) > 1 else None)
        result[f"{metric}_defined_games"] = len(values)
    return result


def stratified_bootstrap_indices(strata: Mapping[str, Sequence], *, replicates=BOOTSTRAP_REPLICATES,
                                 seed=BOOTSTRAP_SEED) -> dict[str, np.ndarray]:
    """One set of local group indices reused for every simulator/comparison.

    Sorted rule order makes dictionary insertion order irrelevant. Each input
    sequence must contain complete independent groups, never player choices.
    """
    if replicates < 1:
        raise ValueError("Need at least one bootstrap replicate")
    rng = np.random.Generator(np.random.PCG64(seed))
    result = {}
    for rule in sorted(strata):
        count = len(strata[rule])
        if count < 1:
            raise ValueError(f"Empty human-group stratum: {rule}")
        result[rule] = rng.integers(0, count, size=(replicates, count))
    return result


def interval(values) -> list[float]:
    return np.quantile(np.asarray(values, dtype=np.float64), [.025, .975]).tolist()


def strict_order(means: Mapping[str, float], *, rules=RULES) -> bool:
    return all(means[a] < means[b] for a, b in zip(rules, rules[1:]))


def institution_error(human: Mapping[str, Sequence[float]], simulated: Mapping[str, Sequence[float]],
                      *, indices=None, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED) -> dict:
    """Equal-rule average absolute level error; interval resamples human groups."""
    rules = tuple(sorted(human))
    if set(rules) != set(simulated):
        raise ValueError("Human and simulator rule sets must match")
    if indices is None:
        indices = stratified_bootstrap_indices(human, replicates=replicates, seed=seed)
    forecast = {rule: float(np.mean(simulated[rule])) for rule in rules}
    observed = {rule: float(np.mean(human[rule])) for rule in rules}
    boot = np.stack([np.abs(forecast[rule] - np.asarray(human[rule])[indices[rule]].mean(axis=1))
                     for rule in rules]).mean(axis=0)
    return {"estimate": float(np.mean([abs(forecast[r] - observed[r]) for r in rules])),
            "ci95": interval(boot), "human_means": observed, "simulated_means": forecast,
            "bootstrap_replicates": len(boot), "interval_resampling": "human groups, stratified by rule; forecast means fixed"}


def paired_error_comparison(human: Mapping[str, Sequence[float]], simulated: Mapping[str, Sequence[float]],
                            incumbent: Mapping[str, Sequence[float]], *, indices=None,
                            rule_a="Interpolating", rule_b="Proportional",
                            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED) -> dict:
    """Task 06 E1/E2, paired absolute-error differences against the incumbent.

    The same observed bootstrap effect/level is subtracted from both fixed
    simulation estimates in every replicate. Monte Carlo error is separate.
    """
    required = (rule_a, rule_b)
    if any(rule not in source for source in (human, simulated, incumbent) for rule in required):
        raise ValueError("Both effect rules are required in every sample")
    selected = {rule: human[rule] for rule in required}
    if indices is None:
        indices = stratified_bootstrap_indices(selected, replicates=replicates, seed=seed)
    observed = {rule: float(np.mean(human[rule])) for rule in required}
    prediction = {rule: float(np.mean(simulated[rule])) for rule in required}
    reference = {rule: float(np.mean(incumbent[rule])) for rule in required}
    boot = {rule: np.asarray(human[rule], dtype=float)[indices[rule]].mean(axis=1) for rule in required}
    if len({len(value) for value in boot.values()}) != 1:
        raise ValueError("Rule bootstrap banks must have the same replicate count")

    def mc_variance(sample):
        values = np.asarray(sample, dtype=float)
        return float(values.var(ddof=1) / len(values)) if len(values) > 1 else 0.0

    output = {}
    for name, observed_value, predicted_value, incumbent_value, observed_boot, rules in (
        ("E1", observed[rule_a] - observed[rule_b], prediction[rule_a] - prediction[rule_b],
         reference[rule_a] - reference[rule_b], boot[rule_a] - boot[rule_b], required),
        ("E2", observed[rule_a], prediction[rule_a], reference[rule_a], boot[rule_a], (rule_a,)),
    ):
        signed = predicted_value - observed_boot
        paired = np.abs(signed) - np.abs(incumbent_value - observed_boot)
        bounds = interval(paired)
        classification = "better" if bounds[1] < 0 else "worse" if bounds[0] > 0 else "not distinguishable"
        output[name] = {"observed": observed_value, "predicted": predicted_value,
                        "incumbent_predicted": incumbent_value,
                        "signed_error": predicted_value - observed_value,
                        "absolute_error": abs(predicted_value - observed_value),
                        "incumbent_absolute_error": abs(incumbent_value - observed_value),
                        "signed_error_ci95": interval(signed),
                        "absolute_error_ci95": interval(np.abs(signed)),
                        "paired_absolute_error_difference": abs(predicted_value - observed_value) - abs(incumbent_value - observed_value),
                        "paired_difference_ci95": bounds, "classification": classification,
                        "prediction_mc_se": math.sqrt(sum(mc_variance(simulated[r]) for r in rules)),
                        "incumbent_mc_se": math.sqrt(sum(mc_variance(incumbent[r]) for r in rules)),
                        "observed_ci95": interval(observed_boot), "bootstrap_replicates": len(observed_boot)}
    return output
