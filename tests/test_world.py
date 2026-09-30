"""Consequential checks on the reconstructed resource rules (stdlib unittest)."""

import math
import random
import unittest

from evopolis.world import WorldState, allocate, participant_observation, step


class WorldTests(unittest.TestCase):
    def test_surplus_and_unallocated_resource_are_accounted_for(self):
        result = step(200, [30, 40, 50, 60], [10, 20, 30, 40])
        self.assertEqual(result.next_pool, 160)
        self.assertEqual(result.surplus, (20, 20, 20, 20))

    def test_capacity_and_absorbing_zero(self):
        result = step(200, [50] * 4, [50] * 4)
        self.assertEqual(result.next_pool, 200)
        self.assertEqual(step(0, [0] * 4, [0] * 4).next_pool, 0)
        state, _ = WorldState().advance([50] * 4, [0] * 4)
        self.assertTrue(state.done)
        with self.assertRaises(ValueError):
            state.advance([0] * 4, [0] * 4)

    def test_invalid_data_is_rejected_without_clipping(self):
        cases = [
            (200, [51] * 4, [1] * 4),
            (200, [-1, 1, 1, 1], [0] * 4),
            (200, [50] * 4, [51, 0, 0, 0]),
            (200, [50] * 4, [-1, 0, 0, 0]),
            (200, [50] * 4, [math.nan, 0, 0, 0]),
            (201, [50] * 4, [0] * 4),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                step(*args)
        tolerated = step(1, [1, 0, 0, 0], [1.000001, 0, 0, 0], tolerance=0.00001)
        self.assertLess(tolerated.surplus[0], 0)  # No concealed repair.

    def test_human_grid_and_clone_continuity_are_distinct(self):
        with self.assertRaises(ValueError):
            step(2, [0.5] * 4, [0.1] * 4, integer_contributions=True)
        self.assertAlmostEqual(step(2, [0.5] * 4, [0.1] * 4).next_pool, 0.56)
        self.assertEqual(step(2, [0.5] * 4, [0] * 4, integer_contributions=True).next_pool, 0)

    def test_baselines_initialization_and_proportional_exclusion(self):
        for name in ("equal", "proportional", "mixed", "interpolating"):
            self.assertEqual(allocate(200, mechanism=name), (50,) * 4)
            self.assertEqual(allocate(0, [0] * 4, mechanism=name), (0,) * 4)
            self.assertEqual(allocate(100, [0] * 4, mechanism=name), (25,) * 4)
        self.assertEqual(allocate(100, [0, 1, 3, 6], mechanism="proportional"), (0, 10, 30, 60))
        self.assertEqual(allocate(100, [0, 1, 3, 6], mechanism="mixed"), (12.5, 17.5, 27.5, 42.5))
        self.assertEqual(allocate(200, [0, 1, 3, 6], mechanism="interpolating"), (50,) * 4)

    def test_feasible_rounds_obey_accounting_and_capacity(self):
        rng = random.Random(1701)
        for _ in range(300):
            pool = rng.uniform(0, 200)
            weights = [rng.random() for _ in range(5)]
            offers = [pool * w / sum(weights) for w in weights[:4]]
            contributions = [offer * rng.random() for offer in offers]
            result = step(pool, offers, contributions)
            self.assertGreaterEqual(result.next_pool, 0)
            self.assertLessEqual(result.next_pool, 200)
            self.assertTrue(all(s >= 0 for s in result.surplus))
            # Uncapped stock + retained surplus equals old stock + generated growth.
            uncapped = pool - sum(offers) + 1.4 * sum(contributions)
            if uncapped <= 200:
                self.assertAlmostEqual(result.next_pool + sum(result.surplus), pool + 0.4 * sum(contributions))

    def test_observation_is_self_first_and_excludes_hidden_horizon(self):
        obs = participant_observation(2, 150, [10, 20, 30, 40], [1, 2, 3, 4])
        self.assertEqual(obs, (30, 40, 10, 20, 3, 4, 1, 2, 150))
        self.assertEqual(len(obs), 9)

    def test_fixed_horizon_is_40_completed_rounds(self):
        state = WorldState()
        for index in range(40):
            self.assertFalse(state.done)
            state, _ = state.advance([50] * 4, [50] * 4)
            self.assertEqual(state.round_index, index + 1)
        self.assertTrue(state.done)
        with self.assertRaises(ValueError):
            state.advance([50] * 4, [50] * 4)


if __name__ == "__main__":
    unittest.main()
