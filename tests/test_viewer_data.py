"""Interpretation-sensitive replay and scripted-policy checks."""

import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from evopolis.viewer_data import (
    ROOT, _recorded_episode, catalog, episode_id, get_episode, prepare_cache, sandbox,
)


def fixture_rows(cohort="Exp 1", episode="0"):
    rows = []
    for index in range(40):
        row = {
            "mech_name_by_player": f"Equal Baseline {cohort}",
            "launch_id": "shared-launch", "episode_id": episode,
            "round_id": str(index), "mechanism_observation.pool": "0.01",
            "next_environment_state.pool": "0.01" if cohort == "Exp 1" else "",
            # Focal-player vectors differ from the required group-order scalars.
            "player_observation_1.offers": "[20 30 40 10]",
        }
        for player in range(4):
            row[f"offer_{player}"] = "0.0025"
            row[f"player_action_{player}"] = "0"
            row[f"player_reward_{player}"] = "0.0025"
        rows.append(row)
    rows[0]["mechanism_observation.pool"] = "200.000000"
    for player in range(4):
        rows[0][f"offer_{player}"] = str(10 * (player + 1))
        rows[0][f"player_action_{player}"] = str(player + 1)
        rows[0][f"player_reward_{player}"] = str(9 * (player + 1))
    # Recorded rewards are authoritative even when they have a precision residual.
    rows[0]["player_reward_0"] = "9.0000001"
    return rows


class RecordedReplayTests(unittest.TestCase):
    def test_group_order_recorded_rewards_pool_timing_and_seek_totals(self):
        rows = fixture_rows()
        rows[0]["next_environment_state.pool"] = "0.015000"
        game = _recorded_episode(list(reversed(rows)))
        first = game["rounds"][0]
        self.assertEqual(first["offers"], [10, 20, 30, 40])
        self.assertEqual(first["contributions"], [1, 2, 3, 4])
        self.assertEqual(first["surplus"][0], 9.0000001)
        self.assertEqual(first["pool_before"], 200)
        self.assertEqual(first["pool_after"], 0.015)
        self.assertEqual(first["pool_after_raw"], "0.015000")
        self.assertEqual(first["equation_after"], 114)
        self.assertEqual(first["after_source"], "recorded next-pool field")
        self.assertEqual(first["raw"]["mechanism_observation.pool"], "200.000000")
        self.assertEqual(game["round_count"], 40)
        self.assertEqual(game["rounds"][-1]["pool_after"], 0.01)
        self.assertEqual(game["rounds"][-1]["active_count"], 0)
        # Precomputed history is seek-independent, including depleted padding.
        for index in (39, 3, 0, 39):
            for player in range(4):
                expected = math.fsum(float(row[f"player_reward_{player}"]) for row in rows[:index + 1])
                self.assertEqual(game["rounds"][index]["cumulative_surplus"][player], expected)

    def test_bc1_uses_following_pool_but_final_round_stays_missing(self):
        game = _recorded_episode(fixture_rows("BC 1"))
        self.assertEqual(game["rounds"][0]["pool_after"], 0.01)
        self.assertEqual(game["rounds"][0]["after_source"], "following recorded round")
        final = game["rounds"][-1]
        self.assertIsNone(final["pool_after"])
        self.assertIsNone(final["pool_after_raw"])
        self.assertIsNone(final["pool_residual"])
        self.assertEqual(final["equation_after"], 0)
        self.assertEqual(len(game["rounds"]), 40)

    def test_complete_episode_keys_prevent_condition_and_episode_collisions(self):
        human = _recorded_episode(fixture_rows())
        clone = _recorded_episode(fixture_rows("BC 1"))
        repeated = _recorded_episode(fixture_rows(episode="1"))
        self.assertEqual(len({human["id"], clone["id"], repeated["id"]}), 3)
        self.assertNotEqual(episode_id("a|b", "c", "d"), episode_id("a", "b|c", "d"))
        mixed = fixture_rows()
        mixed[1]["episode_id"] = "1"
        with self.assertRaises(ValueError):
            _recorded_episode(mixed)
        for malformed in (fixture_rows()[:-1], fixture_rows()[:-1] + [fixture_rows()[0]]):
            with self.assertRaises(ValueError):
                _recorded_episode(malformed)

    def test_median_tie_uses_full_key_not_floating_midpoint_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            with sqlite3.connect(path) as database:
                database.execute("CREATE TABLE episodes (metadata TEXT, condition TEXT, launch_id TEXT, episode_id TEXT)")
                database.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
                database.execute("INSERT INTO metadata VALUES ('provenance', '{}')")
                for launch, surplus in (("a", 1.0), ("z", 1.2)):
                    episode = {
                        "id": launch, "condition": "Equal Baseline Exp 1", "cohort": "human",
                        "mechanism": "Equal", "launch_id": launch, "episode_id": "0", "surplus": surplus,
                    }
                    database.execute("INSERT INTO episodes VALUES (?, ?, ?, ?)", (json.dumps(episode), episode["condition"], launch, "0"))
            self.assertEqual(catalog(path)["default_id"], "a")


class ScriptedSandboxTests(unittest.TestCase):
    def test_fractional_allocation_rounding_does_not_create_negative_pool(self):
        # This feasible policy previously failed after the third round because
        # independently rounded shares totaled 9.8 from a 9.799999999999999 pool.
        game = sandbox("interpolating", [0.2, 0.23, 0.11, 0.19])
        self.assertEqual(game["round_count"], 3)
        self.assertEqual(game["termination"], "exact-zero pool")
        self.assertEqual(game["rounds"][-1]["contributions"], [0] * 4)
        self.assertEqual(game["rounds"][-1]["pool_after"], 0)
        for round_ in game["rounds"]:
            self.assertLessEqual(math.fsum(round_["offers"]), round_["pool_before"])
            self.assertGreaterEqual(round_["pool_after"], 0)

    def test_integer_policy_termination_and_zero_denominator(self):
        empty = sandbox("Equal", [0, 0, 0, 0])
        self.assertEqual(empty["round_count"], 1)
        self.assertEqual(empty["termination"], "exact-zero pool")
        self.assertEqual(empty["rounds"][0]["pool_after"], 0)
        full = sandbox("equal", [1, 1, 1, 1])
        self.assertEqual(full["round_count"], 40)
        self.assertEqual(full["termination"], "40-round limit")
        self.assertIsNone(full["gini"])
        half = sandbox("equal", [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(half["rounds"][0]["contributions"], [25] * 4)
        self.assertEqual(half["rounds"][1]["offers"], [35] * 4)
        self.assertEqual(half["rounds"][1]["contributions"], [17] * 4)
        self.assertAlmostEqual(half["rounds"][1]["pool_after"], 95.2)
        # Decimal q=0.58 means 29 of an initial 50-unit offer. Flooring the
        # ordinary binary product incorrectly returned only 28 units.
        decimal = sandbox("equal", [0.58] * 4)
        self.assertEqual(decimal["rounds"][0]["contributions"], [29] * 4)
        for round_ in half["rounds"]:
            self.assertTrue(all(float(value).is_integer() for value in round_["contributions"]))

    def test_mechanism_changes_begin_new_runs_and_expose_opportunity(self):
        fractions = [0, 1, 1, 1]
        equal = sandbox("equal", fractions)
        proportional = sandbox("proportional", fractions)
        self.assertEqual(equal["rounds"][0]["offers"], [50] * 4)
        self.assertEqual(proportional["rounds"][0]["offers"], [50] * 4)
        self.assertEqual(equal["rounds"][1]["active_count"], 4)
        self.assertEqual(proportional["rounds"][1]["active_count"], 3)
        self.assertEqual(proportional["rounds"][1]["offers"][0], 0)
        self.assertEqual(sandbox("equal", fractions), equal)
        for mechanism in ("mixed", "interpolating"):
            game = sandbox(mechanism, fractions)
            self.assertEqual(game["rounds"][0]["pool_before"], 200)
            self.assertEqual(game["cohort"], "scripted")
        with self.assertRaises(ValueError):
            sandbox("Recorded RL M1", fractions)

    def test_invalid_parameters_never_enter_numerical_environment(self):
        for fractions in ([0, 1], [0, 0, 0, -0.01], [0, 0, 0, 1.01], [0, 0, 0, math.nan], [0, 0, 0, math.inf], [0, 0, 0, "0.5"], [0, 0, 0, True], None):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                sandbox("equal", fractions)


@unittest.skipUnless((ROOT / "data/raw/sustainable_behavior.csv").exists(), "Pinned source not prepared; no network use in tests")
class PinnedSourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = prepare_cache(ROOT / "data/raw", ROOT / "data/cache/viewer")
        cls.catalog = catalog(cls.path)

    def test_all_saved_game_statistics_and_default_selection(self):
        games = self.catalog["episodes"]
        self.assertEqual(len(games), 2208)
        self.assertEqual(sum(game["cohort"] == "human" for game in games), 160)
        self.assertTrue(all(game["round_count"] == 40 for game in games))
        self.assertLessEqual(self.catalog["provenance"]["summary_max_abs_difference"], 1e-9)
        default = get_episode(self.path, self.catalog["default_id"])
        self.assertEqual((default["cohort"], default["mechanism"]), ("human", "Equal"))
        with self.assertRaises(KeyError):
            get_episode(self.path, "not-a-game")

    def test_selected_source_values_match_and_offline_cache_reuses(self):
        selected = [self.catalog["default_id"], next(game["id"] for game in self.catalog["episodes"] if game["cohort"] == "bc1")]
        games = {game["id"]: game for game in (get_episode(self.path, identifier) for identifier in selected)}
        count = 0
        with (ROOT / "data/raw/sustainable_behavior.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                identifier = episode_id(row["mech_name_by_player"], row["launch_id"], row["episode_id"])
                if identifier not in games:
                    continue
                round_ = games[identifier]["rounds"][int(row["round_id"])]
                for name in ("mechanism_observation.pool", "next_environment_state.pool", *(f"{prefix}_{i}" for prefix in ("offer", "player_action", "player_reward") for i in range(4))):
                    self.assertEqual(round_["raw"][name], row[name])
                count += 1
        self.assertEqual(count, 80)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cache = folder / "cache"
            cache.mkdir()
            shutil.copyfile(self.path, cache / self.path.name)
            with patch("evopolis.viewer_data.acquire", side_effect=AssertionError("Offline launch must not download")):
                copied = prepare_cache(folder / "missing-raw", cache)
                self.assertEqual(get_episode(copied, selected[0]), games[selected[0]])
            changed = folder / "changed-raw"
            changed.mkdir()
            (changed / "sustainable_behavior.csv").write_text("An altered source must not silently reuse the cache.\n")
            with self.assertRaisesRegex(ValueError, "Source checksum changed"):
                prepare_cache(changed, cache)
            self.assertEqual(get_episode(copied, selected[0]), games[selected[0]])


if __name__ == "__main__":
    unittest.main()
