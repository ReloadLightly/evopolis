"""Check that replay exposes mismatches and respects independent game resets."""

import csv
from pathlib import Path
import tempfile
import unittest

from evopolis.replay import replay


class ReplayTests(unittest.TestCase):
    def test_source_disagreement_is_retained_with_adjacent_round_identity(self):
        def record(episode, round_id, pool, offers, contributions, next_pool):
            r = {
                "mech_name_by_player": "Equal Baseline Exp 1", "launch_id": "group-1",
                "episode_id": str(episode), "round_id": str(round_id),
                "mechanism_observation.pool": pool, "prev_environment_state.pool": pool,
                "next_environment_state.pool": next_pool,
                "mechanism_reward": sum(offers) - sum(contributions),
            }
            for i in range(4):
                r[f"offer_{i}"] = offers[i]
                r[f"player_action_{i}"] = contributions[i]
                r[f"player_reward_{i}"] = offers[i] - contributions[i]
            return r

        rows = [
            # Logged next state disagrees with the equation and following pool.
            record(0, 0, 200, [50] * 4, [25] * 4, 140.1),
            record(0, 1, 140, [35] * 4, [0] * 4, 0),
            record(0, 2, 0, [0] * 4, [0] * 4, 0),
            # Same group, new episode: must not compare its reset with zero.
            record(1, 0, 200, [50] * 4, [0] * 4, 0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "data.csv"
            with data.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = replay(data, folder)
            condition = report["conditions"]["Equal Baseline Exp 1"]
            self.assertEqual(report["games"], 2)
            self.assertEqual(condition["checks"]["direct_next_pool"]["above_1e_4"], 1)
            self.assertEqual(condition["checks"]["adjacent_next_pool"]["count"], 2)
            self.assertEqual(condition["checks"]["adjacent_next_pool"]["above_1e_4"], 0)
            self.assertEqual(condition["checks"]["next_field_to_adjacent_pool"]["above_1e_4"], 1)
            with (folder / "replay_mismatches.csv").open() as f:
                mismatches = list(csv.DictReader(f))
            self.assertEqual(len(mismatches), 2)
            adjacent = mismatches[1]
            self.assertEqual(adjacent["csv_row"], "3")
            self.assertEqual(adjacent["round_id"], "1")
            self.assertEqual(adjacent["prediction_round_id"], "0")
            self.assertAlmostEqual(float(adjacent["residual"]), -0.1)


if __name__ == "__main__":
    unittest.main()
