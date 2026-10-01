"""Task 05 temporal, latent and primary-estimand forecast checks."""
import copy
import math
import unittest

import numpy as np
import torch

from evopolis.conditional_models import ConditionalModel
from evopolis.conditional_generate import (generate_cell, indexed_uniforms, inverse_cdf,
                                           simulate_branches, validate_integration)
from evopolis.conditional_evaluate import paired_group_summary


def fixture():
    offers = np.full((1, 4, 40), 49.99975)
    actions = np.full((1, 4, 40), 35, dtype=np.int64)
    actions[0, :, :5] = np.array([5, 16, 27, 39])[:, None]
    arrays = {'pool': np.full((1, 40), 200.), 'next_pool': np.full((1, 40), 200.),
              'offers': offers, 'y': actions, 'surplus': offers-actions}
    group = {'index': 0, 'key': ['Mixed Baseline Exp 1', 'fixture', '0'], 'mechanism': 'Mixed',
             'condition': 'Mixed Baseline Exp 1', 'launch_id': 'fixture', 'episode_id': '0'}
    return arrays, group


class ConditionalForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_origin_uses_only_prefix_and_independent_latent_stays_fixed(self):
        arrays, group = fixture()
        torch.manual_seed(29)
        model = ConditionalModel('H1').eval()
        with torch.no_grad():
            model.beta_parameter.fill_(1.)
        paths, details = simulate_branches(model, arrays, group, 5, [17, 23, 31], horizon=5)
        poisoned = copy.deepcopy(arrays)
        poisoned['y'][:, :, 5:] = 199
        poisoned['offers'][:, :, 6:] = 0
        poisoned['pool'][:, 6:] = 0
        poisoned['next_pool'][:] = 0
        poisoned['surplus'][:, :, 5:] = 999
        again, repeated = simulate_branches(model, poisoned, group, 5, [17, 23, 31], horizon=5)
        for name in paths:
            np.testing.assert_array_equal(paths[name], again[name])
        np.testing.assert_array_equal(details['prefix_posterior_weights'], repeated['prefix_posterior_weights'])
        self.assertTrue(np.any(np.ptp(paths['sampled_u'], axis=0) > 0))
        body, retained = generate_cell(model, arrays, group, 5, [17, 23, 31], horizon=5)
        for branch in (body['branch_zero'], body['selected_branch']):
            for row in branch['rounds']:
                for resident, prediction in enumerate(row['predictions']):
                    self.assertEqual(prediction['sampled_latent_effect'], branch['latent_effects'][resident])
                    self.assertAlmostEqual(sum(prediction['pmf']), 1, places=12)
                    self.assertEqual(len(prediction['pmf']), prediction['legal_max']+1)
        # Each round's saved PMF is conditional on the same branch latent draw.
        for offset in range(5):
            live = np.flatnonzero(~paths['padded'][:, offset])
            if not len(live):
                continue
            history = {name[8:]: torch.tensor(value[live, offset]) for name, value in paths.items() if name.startswith('history_')}
            with torch.no_grad():
                expected = model.probs(torch.tensor(paths['pool_before'][live, offset]), torch.tensor(paths['offers'][live, offset]), history, torch.tensor(paths['sampled_u'][live])).numpy()
            np.testing.assert_array_equal(paths['pmfs'][live, offset], expected)

    def test_residual_padding_and_simultaneous_generated_history(self):
        arrays, group = fixture()
        model = ConditionalModel('P0').eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[0, 0] = 1000
            model.head.weight[1, 0] = -1000
            model.head.weight[2, 0] = -1000
        paths, _ = simulate_branches(model, arrays, group, 0, [7], horizon=5)
        residual = 200-math.fsum(arrays['offers'][0, :, 0])
        self.assertEqual(paths['pool_after'][0, 0], residual)
        self.assertEqual(paths['pool_after'][0, 1], 0.)
        np.testing.assert_array_equal(paths['pool_after'][0, 2:], 0.)
        self.assertEqual(paths['window_surplus'][0, -1], 200.)
        self.assertEqual(paths['actual_rounds'][0], 2)
        np.testing.assert_array_equal(paths['history_own_trace'][0, 1], .25)
        np.testing.assert_array_equal(paths['history_peer_trace'][0, 1], .25)
        # At round one, all ineligible observations carry traces; padded rows do too.
        np.testing.assert_array_equal(paths['history_own_trace'][0, 2:], .25)
        np.testing.assert_array_equal(paths['history_peer_trace'][0, 2:], .25)
        arrays['offers'][0, :, 0] = [20.2, .8, 40.5, 60.25]
        model = ConditionalModel('P1').eval()
        paths, _ = simulate_branches(model, arrays, group, 0, [83], horizon=2)
        offers = paths['offers'][0, 0]
        actions = paths['contributions'][0, 0]
        fractions = np.divide(actions, offers, out=np.zeros(4), where=offers >= 1)
        for p in range(4):
            own = .25+.5*fractions[p] if offers[p] >= 1 else .5
            peers = [fractions[j] for j in range(4) if j != p and offers[j] >= 1]
            self.assertAlmostEqual(paths['history_own_trace'][0, 1, p], own)
            self.assertAlmostEqual(paths['history_peer_trace'][0, 1, p], .25+.5*np.mean(peers))
            self.assertEqual(paths['history_previous_eligible_peers'][0, 1, p], len(peers))

    def test_indexed_streams_and_inverse_cdf_never_leave_legal_support(self):
        latent, actions = indexed_uniforms([1337, 42], 20)
        short_latent, short_actions = indexed_uniforms([1337, 42], 5)
        np.testing.assert_array_equal(latent, short_latent)
        np.testing.assert_array_equal(actions[:, :5], short_actions)
        self.assertFalse(np.array_equal(latent, actions[:, 0]))
        self.assertEqual(inverse_cdf([1.], np.nextafter(1., 0)), 0)
        self.assertEqual(inverse_cdf([.9, .0999999999999999], np.nextafter(1., 0)), 1)
        self.assertEqual(inverse_cdf([.5, .5], .5), 1)
        with self.assertRaises(ValueError):
            inverse_cdf([.2, .2], .5)

    def test_integration_receipt_matches_exact_selected_checkpoints_and_resolution(self):
        registry = [{'family': family, 'seed': seed, 'sha256': f'{family}/{seed}', 'integration_nodes': 41}
                    for family in ('H0', 'H1') for seed in (17, 29, 43)]
        checks = [{'family': r['family'], 'seed': r['seed'], 'checkpoint_sha256': r['sha256'],
                   'nodes_compared': [41, 81], 'passed': True} for r in registry]
        report = {'passed': True, 'all_six_persistent_fits_checked': True, 'checks': checks}
        self.assertIs(validate_integration(registry, report), report)
        wrong = copy.deepcopy(report)
        wrong['checks'][0]['nodes_compared'] = [21, 41]
        with self.assertRaises(ValueError):
            validate_integration(registry, wrong)
        wrong = copy.deepcopy(report)
        wrong['checks'][0]['checkpoint_sha256'] = 'previous numerical attempt'
        with self.assertRaises(ValueError):
            validate_integration(registry, wrong)
        wrong = copy.deepcopy(report)
        wrong['checks'].pop()
        with self.assertRaises(ValueError):
            validate_integration(registry, wrong)

    def test_matched_factorial_group_weighting_and_shared_primary_resampling(self):
        rows = []
        for mechanism, count, base in (('Equal', 1, 1.), ('Mixed', 2, 2.), ('Proportional', 3, 9.)):
            for index in range(count):
                for family, effect in (('P0', 0), ('P1', -.1), ('H0', -.2), ('H1', -.45)):
                    for seed, offset in ((17, -.3), (29, 0), (43, .3)):
                        rows.append({'family': family, 'budget': 'conditional', 'seed': seed, 'mechanism': mechanism,
                                     'group_index': 10*int(base)+index, 'group_key': f'{mechanism}:{index}',
                                     'nll': base+effect+offset, 'energy': 2*(base+effect+offset)})
        joint = paired_group_summary(rows, ('nll', 'energy'), replicates=100)
        separate = paired_group_summary(rows, ('nll',), replicates=100)
        self.assertEqual(joint['bootstrap']['plan_sha256'], separate['bootstrap']['plan_sha256'])
        p0 = next(r for r in joint['summary'] if r['procedure'] == 'P0' and r['mechanism'] == 'all')
        self.assertEqual(p0['nll'], 4.)
        effect = next(r for r in joint['comparisons'] if r['comparison'] == 'H1 minus H0')
        self.assertAlmostEqual(effect['differences']['nll'], -.25)
        np.testing.assert_allclose(effect['ci95']['nll'], [-.25, -.25], atol=1e-14)
        np.testing.assert_allclose(effect['ci95']['energy'], [-.5, -.5], atol=1e-14)
        missing = paired_group_summary([r for r in rows if r['mechanism'] != 'Equal'], ('nll',), replicates=100)
        self.assertEqual(missing['unavailable_mechanisms'], ['Equal'])
        self.assertFalse(any(r['mechanism'] == 'all' for r in missing['summary']))
        with self.assertRaises(ValueError):
            paired_group_summary(rows[1:], ('nll',), replicates=100)


if __name__ == '__main__':
    unittest.main()
