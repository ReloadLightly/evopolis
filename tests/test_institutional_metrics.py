"""Consequential Task 06 scoring, cohort and paired-resampling checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evopolis.behavior_data import FIELDS
from evopolis.institutional_data import (Cohort, SYNTHETIC_LABELS, _parse_selected,
                                        game_summaries, load_exp2, verify_freeze)
from evopolis.institutional_metrics import (energy_distance, institution_error,
                                           paired_error_comparison, stratified_bootstrap_indices)
from evopolis.institutional_evaluate import _rank_comparison, mc_se
from evopolis.sources import sha256


class InstitutionalMetricTests(unittest.TestCase):
    def test_energy_distance_matches_offdiagonal_brute_force_and_can_be_negative(self):
        x = np.array([[0., 1.], [1., 4.], [-1., 2.]])
        y = np.array([[1., 0.], [2., 3.], [1., 1.], [4., -2.]])
        cross = sum(np.linalg.norm(a - b) for a in x for b in y) / (len(x) * len(y))
        within_x = sum(np.linalg.norm(a - b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j) / (len(x) * (len(x) - 1))
        within_y = sum(np.linalg.norm(a - b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j) / (len(y) * (len(y) - 1))
        self.assertAlmostEqual(energy_distance(x, y), 2 * cross - within_x - within_y, places=13)
        self.assertLess(energy_distance(x, x), 0)
        with self.assertRaises(ValueError):
            energy_distance(x[:1], y)

    def test_paired_comparison_uses_identical_human_resamples(self):
        human = {"Interpolating": np.array([1., 4., 5., 8.]), "Proportional": np.array([1., 2., 3.])}
        incumbent = {"Interpolating": [4., 6.], "Proportional": [1., 3.]}
        indices = stratified_bootstrap_indices(human, replicates=2000, seed=20261101)
        same = paired_error_comparison(human, incumbent, incumbent, indices=indices)
        for metric in ("E1", "E2"):
            self.assertEqual(same[metric]["paired_difference_ci95"], [0., 0.])
            self.assertEqual(same[metric]["classification"], "not distinguishable")
        simulated = {"Interpolating": [2., 4.], "Proportional": [1., 2.]}
        comparison = paired_error_comparison(human, simulated, incumbent, indices=indices)
        effects = human["Interpolating"][indices["Interpolating"]].mean(axis=1) - human["Proportional"][indices["Proportional"]].mean(axis=1)
        expected = np.abs(1.5 - effects) - np.abs(3. - effects)
        np.testing.assert_array_equal(comparison["E1"]["paired_difference_ci95"], np.quantile(expected, [.025, .975]))
        reversed_strata = stratified_bootstrap_indices(dict(reversed(list(human.items()))), replicates=2000, seed=20261101)
        for rule in human:
            np.testing.assert_array_equal(indices[rule], reversed_strata[rule])

    def test_institution_error_weights_rules_equally(self):
        human = {"Equal": [1., 1.], "Mixed": [4.] * 9, "Proportional": [6.] * 3}
        simulated = {"Equal": [3., 3.], "Mixed": [4.] * 4, "Proportional": [5., 5.]}
        result = institution_error(human, simulated)
        self.assertEqual(result["estimate"], 1.)
        self.assertEqual(result["ci95"], [1., 1.])

    def test_strict_human_order_is_observed_not_assumed_and_ties_fail(self):
        human = {"Equal": [{"surplus": 2.}], "Mixed": [{"surplus": 1.}], "Proportional": [{"surplus": 3.}]}
        simulation = {"Equal": [{"surplus": 1.}], "Mixed": [{"surplus": 2.}], "Proportional": [{"surplus": 3.}]}
        result = _rank_comparison(human, simulation, "surplus")
        self.assertEqual(result["human_order_low_to_high"], ["Mixed", "Equal", "Proportional"])
        self.assertFalse(result["matches_strict_human_order"])
        human["Equal"][0]["surplus"] = 1.
        self.assertFalse(_rank_comparison(human, human, "surplus")["human_strict"])
        self.assertFalse(_rank_comparison(human, human, "surplus")["matches_strict_human_order"])

    def test_mc_error_holds_fitted_seed_mixture_fixed(self):
        rows = [{"training_seed": seed, "surplus": value} for seed, value in ((17, 1.), (29, 100.)) for _ in range(4)]
        self.assertEqual(mc_se(rows, "surplus"), 0.)
        rows[0]["surplus"] = 2.
        # Variance of the equally weighted mean = (1/2)^2 Var(seed17)/4.
        self.assertAlmostEqual(mc_se(rows, "surplus"), .125)


class InstitutionalCohortTests(unittest.TestCase):
    def test_closed_cohorts_are_filtered_before_numeric_parsing_and_synthetic_terminal_is_labelled(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            with source.open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=FIELDS)
                writer.writeheader()
                for t in range(40):
                    row = {field: "" for field in FIELDS}
                    row.update({"mech_name_by_player": "Equal Baseline BC 1", "launch_id": "shared",
                                "episode_id": "0", "round_id": str(t), "mechanism_observation.pool": "200"})
                    for p in range(4):
                        row.update({f"offer_{p}": "50", f"player_action_{p}": "20.5", f"player_reward_{p}": "29.5"})
                    writer.writerow(row)
                for label in ("Proportional Baseline Exp 2", "Interpolating Baseline Exp 3", "RL Agent (M2) Exp 4"):
                    row = {field: "MUST NOT PARSE" for field in FIELDS}
                    row["mech_name_by_player"] = label
                    writer.writerow(row)
            cohort = _parse_selected(source, SYNTHETIC_LABELS, human=False, name="test")
            self.assertEqual(len(cohort.groups), 1)
            self.assertEqual(cohort.groups[0]["key"], ["Equal Baseline BC 1", "shared", "0"])
            self.assertEqual(cohort.arrays["next_pool"][0, 19], 200)
            self.assertAlmostEqual(cohort.arrays["next_pool"][0, 39], 114.8)
            self.assertIn("inferred", cohort.groups[0]["pool40_source"])

    def test_human_summaries_keep_residual_floor_and_use_pool_after_twenty(self):
        arrays = {"surplus": np.full((1, 4, 40), .0025), "next_pool": np.full((1, 40), .01)}
        arrays["next_pool"][0, 19] = 12.3
        group = {"index": 0, "key": ["Equal Baseline Exp 1", "a", "0"], "mechanism": "Equal"}
        summary = game_summaries(Cohort("test", arrays, [group], {}))[0]
        self.assertEqual(summary["surplus"], .0025)
        self.assertEqual(summary["pool40"], .01)
        self.assertEqual(summary["pool20"], 12.3)
        self.assertEqual(summary["first_round_below_one"], 1)
        self.assertFalse(summary["survival"])

    def test_experiment2_refuses_absent_or_incomplete_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "receipt is absent"):
                verify_freeze(root)
            folder = root / "results/task06"
            folder.mkdir(parents=True)
            (folder / "freeze_pushed.json").write_text(json.dumps({"manifest_sha256": "not correct"}))
            (folder / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "manifest hash differs"):
                verify_freeze(root)

    def test_damaged_or_missing_frozen_forecast_refuses_before_human_parser(self):
        for damage in ("missing", "changed"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = root / "results/task06"
                folder.mkdir(parents=True)
                code = root / "analysis.py"
                code.write_text("# immutable analysis fixture\n")
                forecast = folder / "forecast.json"
                forecast.write_text("{}")
                forecast_digest = sha256(forecast)
                if damage == "missing":
                    forecast.unlink()
                else:
                    forecast.write_text('{"changed":true}')
                manifest = {"analysis_code_sha256": {"analysis.py": sha256(code)},
                            "forecast_summary_sha256": {"results/task06/forecast.json": forecast_digest},
                            "checkpoints": {"fixture_17": {}}, "seed_table": [{}]}
                manifest_path = folder / "manifest.json"
                manifest_path.write_text(json.dumps(manifest))
                receipt = {"manifest_sha256": sha256(manifest_path),
                           "analysis_code_sha256": manifest["analysis_code_sha256"]}
                (folder / "freeze_pushed.json").write_text(json.dumps(receipt))
                with patch("evopolis.institutional_data._parse_selected") as parse, \
                        patch("evopolis.institutional_data.subprocess.check_output") as remote:
                    with self.assertRaisesRegex(RuntimeError, "frozen artifact missing or changed.*forecast.json"):
                        load_exp2(root / "unreadable.csv", root=root)
                    parse.assert_not_called()
                    remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
