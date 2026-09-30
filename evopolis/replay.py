"""Stream the release to audit accounting without silently repairing records."""

from collections import defaultdict
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path

from .world import CAPACITY, MULTIPLIER, step


TIGHT_TOLERANCE = 1e-9
REPLAY_TOLERANCE = 1e-4


@dataclass
class Residuals:
    count: int = 0
    above_1e_9: int = 0
    above_1e_4: int = 0
    max_abs: float = 0.0
    sum_abs: float = 0.0
    sum_squared: float = 0.0

    def add(self, residual: float) -> None:
        absolute = abs(residual)
        self.count += 1
        self.above_1e_9 += absolute > TIGHT_TOLERANCE
        self.above_1e_4 += absolute > REPLAY_TOLERANCE
        self.max_abs = max(self.max_abs, absolute)
        self.sum_abs += absolute
        self.sum_squared += residual * residual

    def report(self) -> dict:
        return {
            "count": self.count,
            "above_1e_9": self.above_1e_9,
            "above_1e_4": self.above_1e_4,
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / self.count if self.count else None,
            "rmse": math.sqrt(self.sum_squared / self.count) if self.count else None,
        }


@dataclass
class Condition:
    rows: int = 0
    step_rejected_at_1e_4: int = 0
    first_rounds: int = 0
    nonconsecutive_rows: int = 0
    floor_pattern: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    checks: dict[str, Residuals] = field(default_factory=lambda: defaultdict(Residuals))


def replay(path: Path, output_dir: Path) -> dict:
    """Audit every row and adjacent round, writing summary and affected cases.

    A direct next-state field is available for human rows. For every origin we
    also compare the reconstruction with the next row's pre-allocation pool,
    within a (condition, launch_id, episode_id) game. Error is observed minus
    reconstructed. Rejected inputs remain in the audit: a raw equation is then
    evaluated explicitly for diagnosis, without clipping or treating them as a
    successful simulator replay. Each player's reward is checked independently.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions: dict[str, Condition] = defaultdict(Condition)
    previous = {}
    mismatch_path = output_dir / "replay_mismatches.csv"
    fields = [
        "csv_row", "condition", "launch_id", "episode_id", "round_id", "check",
        "prediction_round_id", "player_index", "observed", "reconstructed",
        "residual", "input_valid_at_1e_4",
    ]
    with path.open(newline="") as source, mismatch_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row_number, row in enumerate(csv.DictReader(source), start=2):
            label = row["mech_name_by_player"]
            group = conditions[label]
            group.rows += 1
            human = " Exp " in label
            if not human and " BC " not in label:
                raise ValueError(f"unrecognized origin at CSV row {row_number}: {label}")
            pool = float(row["mechanism_observation.pool"])
            offers = tuple(float(row[f"offer_{i}"]) for i in range(4))
            contributions = tuple(float(row[f"player_action_{i}"]) for i in range(4))
            rewards = tuple(float(row[f"player_reward_{i}"]) for i in range(4))
            numbers = (pool,) + offers + contributions + rewards
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError(f"nonfinite required replay input at CSV row {row_number}")
            valid = True
            try:
                result = step(pool, offers, contributions, tolerance=REPLAY_TOLERANCE)
                predicted = result.next_pool
                surplus = result.surplus
            except ValueError:
                group.step_rejected_at_1e_4 += 1
                valid = False
                predicted = min(CAPACITY, pool - math.fsum(offers) + MULTIPLIER * math.fsum(contributions))
                surplus = tuple(offer - contribution for offer, contribution in zip(offers, contributions))

            def check(
                name: str, observed: float, expected: float, *,
                check_valid: bool = valid, prediction_round: int | None = None,
                player_index: int | None = None,
            ) -> None:
                residual = observed - expected
                group.checks[name].add(residual)
                if abs(residual) > REPLAY_TOLERANCE:
                    if name in {"direct_next_pool", "adjacent_next_pool"} and abs(observed - 0.01) < 1e-7 and expected < 0.01:
                        group.floor_pattern[name] += 1
                    writer.writerow({
                        "csv_row": row_number, "condition": label,
                        "launch_id": row["launch_id"], "episode_id": row["episode_id"],
                        "round_id": row["round_id"], "check": name,
                        "prediction_round_id": row["round_id"] if prediction_round is None else prediction_round,
                        "player_index": player_index,
                        "observed": observed, "reconstructed": expected,
                        "residual": residual, "input_valid_at_1e_4": check_valid,
                    })

            for name, amount in {
                "negative_offer": max(0.0, -min(offers)),
                "allocation_excess": max(0.0, math.fsum(offers) - pool),
                "negative_contribution": max(0.0, -min(contributions)),
                "contribution_excess": max(0.0, max(c - e for c, e in zip(contributions, offers))),
                "pool_bound_violation": max(0.0, -pool, pool - CAPACITY),
            }.items():
                check(name, amount, 0)
            if human:
                check("human_integer_grid", max(abs(c - round(c)) for c in contributions), 0)
            for player_index, (observed, expected) in enumerate(zip(rewards, surplus)):
                check("player_surplus", observed, expected, player_index=player_index)
            if row["mechanism_reward"]:
                check("mechanism_surplus", float(row["mechanism_reward"]), math.fsum(surplus))
            if row["prev_environment_state.pool"]:
                check("pre_pool_field", float(row["prev_environment_state.pool"]), pool)
            if row["next_environment_state.pool"]:
                check("direct_next_pool", float(row["next_environment_state.pool"]), predicted)

            key = (label, row["launch_id"], row["episode_id"])
            round_id = int(float(row["round_id"]))
            if round_id == 0:
                group.first_rounds += 1
                check("initial_pool", pool, CAPACITY)
            if key in previous:
                prior_round, prior_prediction, prior_valid, prior_logged = previous[key]
                if round_id == prior_round + 1:
                    check("adjacent_next_pool", pool, prior_prediction, check_valid=prior_valid, prediction_round=prior_round)
                    if prior_logged is not None:
                        check("next_field_to_adjacent_pool", pool, prior_logged, check_valid=prior_valid, prediction_round=prior_round)
                else:
                    group.nonconsecutive_rows += 1
            previous[key] = (round_id, predicted, valid, float(row["next_environment_state.pool"]) if row["next_environment_state.pool"] else None)

    report = {
        "method": "Published equation; double precision; no data clipping or rounding.",
        "residual_sign": "observed minus reconstructed",
        "tolerances": {"tight": TIGHT_TOLERANCE, "replay": REPLAY_TOLERANCE},
        "rows": sum(group.rows for group in conditions.values()),
        "games": len(previous),
        "mismatches_file": mismatch_path.name,
        "mismatch_row_convention": "csv_row and round_id locate the observation; prediction_round_id locates the allocation/contribution round. For adjacent_next_pool the observation is the following round's pool.",
        "floor_pattern_definition": "Among residuals > 1e-4, observed pool within 1e-7 of 0.01 and reconstructed pool below 0.01. Diagnostic evidence, not a correction to the published equation.",
        "conditions": {
            label: {
                "origin": "human" if " Exp " in label else "recorded_behavioral_clone",
                "rows": group.rows,
                "first_rounds": group.first_rounds,
                "step_rejected_at_1e_4": group.step_rejected_at_1e_4,
                "nonconsecutive_rows": group.nonconsecutive_rows,
                "pool_floor_pattern_mismatches": dict(group.floor_pattern),
                "checks": {name: residual.report() for name, residual in sorted(group.checks.items())},
            }
            for label, group in sorted(conditions.items())
        },
    }
    with (output_dir / "replay_by_condition.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["condition", "origin", "check", "count", "above_1e_9", "above_1e_4", "max_abs", "mean_abs", "rmse"], lineterminator="\n")
        writer.writeheader()
        for label, group in report["conditions"].items():
            for name, residual in group["checks"].items():
                writer.writerow({"condition": label, "origin": group["origin"], "check": name, **residual})
    return report
