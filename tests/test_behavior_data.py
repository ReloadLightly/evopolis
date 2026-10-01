"""Consequential input timing, exact-support and group-separation checks."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evopolis.behavior_data import (
    CONDITIONS, FIELDS, assign_splits, build_arrays, content_hash, load,
    read_config, rollout_seed_table, stream_groups, write_json,
)
from evopolis.sources import sha256


def rows(condition="Equal Baseline Exp 1", launch="fixture", episode="0"):
    result = []
    for time in range(40):
        row = {"mech_name_by_player": condition, "launch_id": launch,
               "episode_id": episode, "round_id": str(time),
               "mechanism_observation.pool": "200",
               "next_environment_state.pool": "140"}
        for player in range(4):
            row[f"offer_{player}"] = str(20 + 10 * player)
            row[f"player_action_{player}"] = str(player + 1)
            row[f"player_reward_{player}"] = str(19 + 9 * player)
        result.append(row)
    return result


def grouped(records):
    first = records[0]
    return {(first["mech_name_by_player"], first["launch_id"], first["episode_id"]): records}


class BehavioralDataTests(unittest.TestCase):
    def test_source_boundary_ignores_synthetic_and_transfer_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            with source.open("w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows())
                for condition in ("Equal Baseline BC 1", "Proportional Baseline Exp 2", "RL Agent (M2) Exp 3"):
                    excluded = rows(condition)[0]
                    excluded["player_action_0"] = "should never be parsed"
                    writer.writerow(excluded)
            groups = stream_groups(source)
            self.assertEqual(list(groups), [("Equal Baseline Exp 1", "fixture", "0")])
            arrays, _ = build_arrays(groups)
            self.assertEqual(arrays["y"].shape, (1, 4, 40))

    def test_rotation_timing_and_future_outcome_boundary(self):
        original = rows()
        before, _ = build_arrays(grouped(original))
        np.testing.assert_array_equal(before["x"][0, 2, 0], np.array([40, 50, 20, 30, 0, 0, 0, 0, 200]) / 200)
        np.testing.assert_array_equal(before["x"][0, 2, 1], np.array([40, 50, 20, 30, 3, 4, 1, 2, 200]) / 200)
        changed = copy.deepcopy(original)
        changed[12]["player_action_0"] = "17"
        changed[12]["next_environment_state.pool"] = "3.14"
        changed[12]["player_reward_0"] = "0.123"
        changed[25]["offer_0"] = "19.5"
        after, _ = build_arrays(grouped(changed))
        np.testing.assert_array_equal(before["x"][:, :, :13], after["x"][:, :, :13])
        self.assertEqual(after["x"][0, 0, 13, 4], 17 / 200)
        self.assertNotEqual(before["x"][0, 0, 13, 4], after["x"][0, 0, 13, 4])

    def test_exact_floor_and_retention_of_forced_rounds(self):
        records = rows()
        records[0]["offer_0"] = "49.99999999999999"
        records[0]["player_action_0"] = "49"
        records[1]["offer_0"] = "0.9999999999999999"
        records[1]["player_action_0"] = "0"
        arrays, metadata = build_arrays(grouped(records))
        self.assertEqual(arrays["n"][0, 0, 0], 49)
        self.assertEqual(arrays["n"][0, 0, 1], 0)
        self.assertEqual(arrays["x"][0, 0, 1, 4], 49 / 200)
        self.assertEqual(arrays["x"][0, 0, 2, 4], 0)
        self.assertEqual(metadata[0]["nonforced_choices"], 159)
        self.assertEqual(arrays["x"].shape, (1, 4, 40, 9))
        records[0]["player_action_0"] = "50"
        with self.assertRaisesRegex(ValueError, "Illegal integer target"):
            build_arrays(grouped(records))
        records[0]["player_action_0"] = "3.5"
        with self.assertRaisesRegex(ValueError, "Illegal integer target"):
            build_arrays(grouped(records))

    def test_episode_completeness_and_human_provenance_required(self):
        for records in (rows()[:-1], rows()[:-1] + [rows()[0]]):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                build_arrays(grouped(records))
        with self.assertRaisesRegex(ValueError, "Non-human"):
            build_arrays(grouped(rows("Equal Baseline BC 1")))

    def test_stratified_split_deterministic_under_input_order_and_launch_isolation(self):
        metadata = []
        for stratum, (condition, mechanism) in enumerate(CONDITIONS.items()):
            for index in range(40):
                launch = f"{stratum}-{index:02d}"
                metadata.append({"index": len(metadata), "condition": condition, "mechanism": mechanism,
                                 "key": [condition, launch, "0"], "launch_id": launch})
        first = assign_splits(metadata)
        second = assign_splits(list(reversed(metadata)))
        self.assertEqual({tuple(g["key"]): g["split"] for g in first}, {tuple(g["key"]): g["split"] for g in second})
        for condition in CONDITIONS:
            self.assertEqual([sum(g["condition"] == condition and g["split"] == split for g in first)
                              for split in ("train", "validation", "test")], [24, 8, 8])
        training = next(g for g in first if g["split"] == "train")
        heldout = next(g for g in first if g["split"] == "test")
        contaminated = copy.deepcopy(metadata)
        contaminated[heldout["index"]]["launch_id"] = training["launch_id"]
        with self.assertRaisesRegex(ValueError, "crosses split boundaries"):
            assign_splits(contaminated)

    def test_seed_table_complete_distinct_and_deterministic(self):
        config = read_config()
        table = rollout_seed_table(config)
        self.assertEqual(len(table["games"]), 3072)
        self.assertEqual(len({row["rollout_seed"] for row in table["games"]}), 3072)
        self.assertEqual(table, rollout_seed_table(config))
        self.assertEqual(table["table_hash"], content_hash(table["games"]))
        self.assertEqual(table["games"][0]["family"], "constant")
        self.assertEqual(table["games"][-1]["family"], "recurrent")

    def test_cached_array_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "human_exp1.npz"
            np.savez_compressed(target, x=np.array([1.25]))
            manifest = {"source_sha256": "fixture", "groups": []}
            manifest["manifest_hash"] = content_hash(manifest)
            write_json(root / "split.json", manifest)
            write_json(root / "preparation.json", {"split_hash": manifest["manifest_hash"], "array_sha256": sha256(target)})
            arrays, reloaded = load(cache_dir=root, results_dir=root)
            np.testing.assert_array_equal(arrays["x"], [1.25])
            self.assertEqual(reloaded, manifest)
            np.savez_compressed(target, x=np.array([1.5]))
            with self.assertRaisesRegex(ValueError, "array or split hash"):
                load(cache_dir=root, results_dir=root)
            manifest["groups"] = ["tampered"]
            write_json(root / "split.json", manifest)
            with self.assertRaisesRegex(ValueError, "manifest integrity"):
                load(cache_dir=root, results_dir=root)


if __name__ == "__main__":
    unittest.main()
