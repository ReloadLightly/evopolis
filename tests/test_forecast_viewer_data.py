"""Forecast replay must preserve the observed boundary and archived numbers."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from evopolis.forecast_viewer_data import forecast_catalog, forecast_episode


class ForecastArchiveTests(unittest.TestCase):
    def test_prefix_branch_padding_and_large_seed_survive_read_only_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecasts.sqlite3"
            metadata = {"id": "forecast-test", "bank": "main", "boundary": "recorded",
                        "mechanism": "Mixed", "family": "recurrent", "training_seed": 17,
                        "budget": "continued", "origin": 20, "horizon": 20,
                        "key": ["Mixed Baseline Exp 1", "launch", "episode"],
                        "selected_epoch": 211, "checkpoint_sha256": "checkpoint"}
            observed = [{"round_id": index, "pool_before": 199.999123 + index / 10000,
                         "contributions": [1, 2, 3, 4], "cumulative_surplus": [index] * 4}
                        for index in range(40)]
            generated = [{"round_id": 20 + index, "pool_before": 0 if index else 200,
                          "padded": index > 0, "cumulative_surplus": [23.9999] * 4}
                         for index in range(20)]
            branch = {"branch_index": 0, "rollout_seed": "18446744073709551615", "rounds": generated}
            body = {"observed_rounds": observed, "selected_branch": {**branch, "branch_index": 12},
                    "branch_zero": branch, "bands": [], "provenance": {"source_sha256": "source"}}
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE cells(id TEXT PRIMARY KEY, metadata TEXT, body BLOB)")
                database.execute("INSERT INTO cells VALUES(?,?,?)", (metadata["id"], json.dumps(metadata), zlib.compress(json.dumps(body).encode())))
            original = path.read_bytes()
            catalog = forecast_catalog(path)
            self.assertEqual(catalog["default_id"], metadata["id"])
            result = forecast_episode(metadata["id"], path=path)
            forecast, actual = result["forecast"], result["observed"]
            self.assertEqual(forecast["branch_index"], 0)
            self.assertEqual(forecast["rollout_seed"], "18446744073709551615")
            self.assertEqual(len(forecast["rounds"]), 40)
            self.assertEqual(len(actual["rounds"]), 40)
            self.assertEqual(forecast["actual_rounds"], 21)
            self.assertTrue(forecast["rounds"][19]["observed"])
            self.assertFalse(forecast["rounds"][20]["observed"])
            self.assertTrue(forecast["rounds"][21]["padded"])
            self.assertEqual(forecast["rounds"][19]["pool_before"], observed[19]["pool_before"])
            self.assertEqual(actual["rounds"][20]["contributions"], [1, 2, 3, 4])
            self.assertEqual(forecast["rounds"][-1]["cumulative_surplus"], [23.9999] * 4)
            self.assertEqual(forecast_episode(metadata["id"], "median", path)["forecast"]["branch_index"], 12)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(KeyError):
                forecast_episode("missing", path=path)

    def test_missing_archive_and_invalid_horizon_do_not_invent_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecasts.sqlite3"
            self.assertEqual(forecast_catalog(path)["cells"], [])
            with self.assertRaises(KeyError):
                forecast_episode("missing", path=path)
            self.assertFalse(path.exists())
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE cells(id TEXT PRIMARY KEY, metadata TEXT, body BLOB)")
                metadata = {"bank": "main", "boundary": "recorded", "origin": 30, "horizon": 20}
                database.execute("INSERT INTO cells VALUES(?,?,?)", ("invalid", json.dumps(metadata), zlib.compress(b"{}")))
            with self.assertRaisesRegex(ValueError, "episode window"):
                forecast_episode("invalid", path=path)


if __name__ == "__main__":
    unittest.main()
