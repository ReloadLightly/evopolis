"""Hand-worked proper scores, exact sum calibration and human-unit uncertainty."""

import math
import unittest

import numpy as np

from evopolis.forecast_evaluate import (
    convolve_returns, crps, energy_score, interval_coverage,
    observed_endpoint, paired_group_summary, renewal_score, score_cell,
)


class ForecastScoreTests(unittest.TestCase):
    def test_off_diagonal_energy_and_crps_hand_calculation(self):
        # Distances to y are 0 and 5, pair distances total 10.
        # Off-diagonal ES = 2.5 - 10/(2*2*1) = 0, not the biased 1.25.
        draws = np.array([[0.0, 0.0], [3.0, 4.0]])
        self.assertEqual(energy_score(draws, [0.0, 0.0]), 0.0)
        self.assertEqual(crps([0.0, 2.0], 1.0), 0.0)
        # A midpoint lies on the triangle-equality boundary; do not add a bias term.
        self.assertLess(energy_score(draws, [1.5, 2.0]), 1e-15)
        triangle = np.array([[0., 0.], [2., 0.], [0., 2.]])
        expected = (0 + 2 + 2) / 3 - (2 * (2 + 2 + math.sqrt(8))) / 12
        self.assertAlmostEqual(energy_score(triangle, [0, 0]), expected)
        self.assertEqual(interval_coverage([0, 1, 2, 3, 4], .4), (1., .4, 3.6))

    def test_exact_convolution_and_renewal_threshold(self):
        # Four independent fair Bernoullis, exact binomial coefficients.
        probabilities = np.array([[.5, .5, 0], [.5, .5, 0], [.5, .5, 0], [.5, .5, 0]])
        expected = np.array([1, 4, 6, 4, 1]) / 16
        np.testing.assert_array_equal(convolve_returns(probabilities, [1] * 4), expected)
        # Current allocated resource 4; renewal iff total >= ceil(4/1.4)=3.
        row = renewal_score(probabilities, [1] * 4, [1] * 4, [1, 1, 1, 0])
        self.assertAlmostEqual(row['aggregate_nll'], -math.log(4/16))
        self.assertEqual(row['renewal_probability'], 5/16)
        self.assertEqual(row['renewal_observed'], 1)
        self.assertEqual(row['renewal_brier'], (1 - 5/16) ** 2)
        # Fractional offers keep their exact threshold and no support epsilon.
        row = renewal_score(probabilities, [1] * 4, [1.1] * 4, [1, 1, 1, 0])
        self.assertEqual(row['renewal_probability'], 1/16)
        self.assertEqual(row['renewal_observed'], 0)
        with self.assertRaises(ValueError):
            convolve_returns(probabilities, [0, 1, 1, 1])
        forced = np.array([[1., 0], [1., 0], [1., 0], [1., 0]])
        np.testing.assert_array_equal(convolve_returns(forced, [0] * 4), [1.])

    def test_group_seed_and_unequal_stratum_weighting_with_paired_uncertainty(self):
        rows = []
        # Group counts 1/2/3 must not turn the 3 equal mechanism means into
        # pooled group weighting. Seed offsets cancel within each group.
        for mechanism, size, value in (('Equal', 1, 1.), ('Mixed', 2, 2.), ('Proportional', 3, 9.)):
            for group in range(size):
                for seed, offset in ((17, -.3), (29, 0), (43, .3)):
                    for budget in ('original', 'continued'):
                        rows.append({'family': 'recurrent', 'budget': budget, 'seed': seed,
                                     'mechanism': mechanism, 'group_index': 10 * value + group,
                                     'energy': value + offset - (.25 if budget == 'continued' else 0)})
        result = paired_group_summary(rows, ('energy',), replicates=100)
        original = next(r for r in result['summary'] if r['procedure'] == 'recurrent_original' and r['mechanism'] == 'all')
        self.assertEqual(original['energy'], 4.)
        effect = result['comparisons'][0]
        self.assertAlmostEqual(effect['differences']['energy'], -.25)
        np.testing.assert_allclose(effect['ci95']['energy'], [-.25, -.25], atol=1e-14)
        self.assertEqual(effect['groups'], 6)
        missing = paired_group_summary([r for r in rows if r['mechanism'] != 'Equal'], ('energy',), replicates=100)
        self.assertEqual(missing['unavailable_mechanisms'], ['Equal'])
        self.assertFalse(missing['all_declared_mechanisms_available'])
        self.assertTrue(all(r['energy'] is None for r in missing['summary'] if r['mechanism'] == 'Equal'))
        self.assertFalse(any(r['mechanism'] == 'all' for r in missing['summary']))
        with self.assertRaises(ValueError):
            paired_group_summary(rows[1:], ('energy',), replicates=100)

    def test_endpoint_uses_window_only_and_saved_rewards_with_correct_horizon(self):
        arrays = {'next_pool': np.arange(40, dtype=float)[None, :] + .01,
                  'surplus': np.ones((1, 4, 40)), 'offers': np.ones((1, 4, 40))}
        arrays['surplus'][:, :, :5] = 999  # prefix must never enter denominator
        expected = np.array([14.01 / 200, 40 / 2000])
        np.testing.assert_allclose(observed_endpoint(arrays, 0, 5, 10), expected)
        with self.assertRaises(ValueError):
            observed_endpoint(arrays, 0, 25, 20)
        pool = np.zeros((64, 10))
        surplus = np.tile(np.arange(1, 11) * 4, (64, 1))
        meta = {'id': 'fixture', 'group_index': 0, 'origin': 5, 'family': 'recurrent',
                'budget': 'continued', 'seed': 17, 'checkpoint_sha256': 'synthetic',
                'mechanism': 'Mixed', 'bank': 'main', 'convention': 'recorded'}
        body = {'endpoints': {}, 'observed_endpoints': {}}
        for h in (1, 5, 10):
            body['endpoints'][str(h)] = np.stack((pool[:, h-1]/200, surplus[:, h-1]/(200*h)), axis=-1)
            body['observed_endpoints'][str(h)] = observed_endpoint(arrays, 0, 5, h)
        rows = score_cell(meta, body, {'pool_after': pool, 'window_surplus': surplus}, arrays)
        self.assertEqual([r['horizon'] for r in rows], [1, 5, 10])
        self.assertEqual(rows[-1]['surplus_crps'], 0)
        self.assertEqual(rows[-1]['pool_crps'], expected[0])
        # One-step trajectory CRPS averages the ten actual post-decision pools.
        self.assertAlmostEqual(rows[-1]['trajectory_pool_crps'], np.mean(np.arange(5, 15) + .01) / 200)
        body['observed_endpoints']['10'] = [0, 0]
        with self.assertRaises(ValueError):
            score_cell(meta, body, {'pool_after': pool, 'window_surplus': surplus}, arrays)


if __name__ == '__main__':
    unittest.main()
