"""Checks of the estimand, including the undefined-zero convention."""

import math
import unittest

from evopolis.analysis import gini, mean_conf


class EstimandTests(unittest.TestCase):
    def test_four_player_gini_scale_and_limit(self):
        self.assertEqual(gini([2, 2, 2, 2]), 0)
        self.assertEqual(gini([0, 0, 0, 8]), 0.75)
        self.assertEqual(gini([0, 1, 2, 5]), gini([0, 40, 80, 200]))
        self.assertTrue(math.isnan(gini([0, 0, 0, 0])))

    def test_uncertainty_is_over_games_and_preserves_undefined_gini(self):
        mean, width = mean_conf([1, 3])
        self.assertEqual(mean, 2)
        self.assertAlmostEqual(width, 1.96 / math.sqrt(2))
        mean, width = mean_conf([0.1, math.nan])
        self.assertTrue(math.isnan(mean) and math.isnan(width))
