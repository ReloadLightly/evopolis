"""Published trajectory loading must preserve numbers and seed identity."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from evopolis.trained_viewer_data import generated_catalog, generated_episode


class TrainedArchiveTests(unittest.TestCase):
    def test_read_only_archive_preserves_seed_precision_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generated.sqlite3"
            metadata = {"id": "test-game", "cohort": "trained", "family": "recurrent",
                        "training_seed": 17, "mechanism": "Equal", "rollout_seed": "18446744073709551615", "rollout_index": 0,
                        "surplus": 1.25, "gini": None, "actual_rounds": 1}
            rounds = [{"round_id": 0, "pool_before": 200.0, "padded": False},
                      {"round_id": 1, "pool_before": 0.0, "padded": True,
                       "cumulative_surplus": [50.0] * 4}]
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE episodes(id TEXT PRIMARY KEY, metadata TEXT, body BLOB)")
                database.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
                database.execute("INSERT INTO episodes VALUES(?,?,?)", (metadata["id"], json.dumps(metadata), zlib.compress(json.dumps({"rounds": rounds}).encode())))
                database.execute("INSERT INTO metadata VALUES('provenance',?)", (json.dumps({"source_sha256": "source"}),))
            index = generated_catalog(path)
            self.assertEqual(index["default_id"], metadata["id"])
            episode = generated_episode(metadata["id"], path)
            self.assertEqual(episode["rollout_seed"], "18446744073709551615")
            self.assertEqual(episode["rounds"], rounds)
            self.assertIsNone(episode["gini"])
            with self.assertRaises(KeyError):
                generated_episode("unknown", path)

    def test_missing_archive_does_not_create_a_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "absent.sqlite3"
            self.assertEqual(generated_catalog(path)["episodes"], [])
            with self.assertRaises(KeyError):
                generated_episode("unknown", path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
