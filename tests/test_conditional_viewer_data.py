"""Playback retains the selected branch's latent effect and pre-choice evidence."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from evopolis.conditional_viewer_data import conditional_catalog, conditional_episode


class ConditionalPlaybackTests(unittest.TestCase):
    def test_branch_effect_history_and_pmf_are_not_mixed_between_illustrations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecasts.sqlite3"
            metadata = {"id": "task05-H1-test", "family": "H1", "budget": "conditional",
                        "bank": "main", "boundary": "recorded", "training_seed": 17,
                        "key": ["Mixed Baseline Exp 1", "launch", "episode"],
                        "mechanism": "Mixed", "origin": 5, "horizon": 20}
            history = {"own_trace": 0.123456789, "peer_trace": 0.678912345,
                       "previous_own_valid": False, "previous_eligible_peers": 2}
            prediction = {"legal_max": 1, "pmf": [0.25, 0.75], "history": history}
            prefix = [{"round_id": index, "offers": [50] * 4} for index in range(40)]
            rounds = [{"round_id": 5 + index, "predictions": [prediction] * 4}
                      for index in range(20)]
            branch_zero = {"branch_index": 0, "rollout_seed": "18446744073709551615",
                           "latent_effects": [-1.2, 0.3, 0.5, 1.0], "rounds": rounds}
            median = {**branch_zero, "branch_index": 31, "latent_effects": [1, 2, 3, 4]}
            posteriors = [{"mean": 0.12, "sd": 0.34}] * 4
            body = {"observed_rounds": prefix, "branch_zero": branch_zero,
                    "selected_branch": median, "bands": [], "prefix_posteriors": posteriors,
                    "parameters": {"beta": -0.41, "eta": 0.23, "sigma": 1.4}}
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE cells(id TEXT PRIMARY KEY, metadata TEXT, body BLOB)")
                database.execute("INSERT INTO cells VALUES(?,?,?)", (metadata["id"], json.dumps(metadata),
                                 zlib.compress(json.dumps(body).encode())))
            before = path.read_bytes()
            self.assertEqual(conditional_catalog(path)["cells"][0]["budget_epochs"], 480)
            first = conditional_episode(metadata["id"], path=path)
            middle = conditional_episode(metadata["id"], "median", path)
            self.assertEqual(first["forecast"]["latent_effects"], branch_zero["latent_effects"])
            self.assertEqual(middle["forecast"]["latent_effects"], median["latent_effects"])
            self.assertEqual(first["forecast"]["prefix_posteriors"], posteriors)
            self.assertEqual(first["forecast"]["parameters"], body["parameters"])
            self.assertEqual(first["forecast"]["rollout_seed"], branch_zero["rollout_seed"])
            self.assertEqual(first["forecast"]["rounds"][5]["predictions"][0], prediction)
            self.assertNotIn("latent_effects", first["observed"])
            self.assertNotIn("predictions", first["forecast"]["rounds"][4])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
