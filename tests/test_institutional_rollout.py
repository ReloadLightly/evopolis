"""Task 06 distribution, random-stream and free-community invariants."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtri
import torch

from evopolis.conditional_models import ConditionalModel, base_log_mass
from evopolis.institutional_models import make_model
from evopolis.institutional_rollout import (
    RULES, seeds_for, simulate, simulate_reference, tilt_log_mass, uniforms_for,
)

torch.set_num_threads(1)


class InstitutionalRolloutChecks(unittest.TestCase):
    def test_tilts_normalize_preserve_support_and_use_actual_offer(self):
        raw = torch.tensor([[.2, -.1, .4, 1.2, -.8]] * 4, dtype=torch.float64)
        offers = torch.tensor([0., .999, 1.999, 200.], dtype=torch.float64)
        base = base_log_mass(raw, offers)
        actions = torch.arange(201)
        for magnitude in (-100., -.3, 0., .6, 100.):
            tilted = tilt_log_mass(base, offers, magnitude)
            probabilities = tilted.exp()
            self.assertTrue(torch.isfinite(probabilities).all())
            self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-13, rtol=0))
            self.assertEqual(float(probabilities[0, 0]), 1.)
            self.assertEqual(float(probabilities[1, 0]), 1.)
            self.assertEqual(float(probabilities[actions[None, :] > offers.floor()[:, None]].sum()), 0.)
            # Normalization cancels from odds. e=1.999 distinguishes actual-e
            # tilting from the historically audited floor(e) perturbation.
            log_odds_change = (tilted[2, 1] - tilted[2, 0]) - (base[2, 1] - base[2, 0])
            self.assertAlmostEqual(float(log_odds_change), magnitude / 1.999, places=12)

    def test_indexed_streams_survive_subsetting_reordering_and_crn_aliases(self):
        seeds = seeds_for("FA-H0", 17, "Mixed", 5)
        self.assertEqual(seeds[:2], seeds_for("FA-H0", 17, "Mixed", 2))
        self.assertEqual(seeds, seeds_for("CL-FA-H0", 17, "Mixed", 5))
        self.assertNotEqual(seeds, seeds_for("FA-H0", 17, "Mixed", 5, namespace=20261103))
        action, effects = uniforms_for(seeds)
        chosen = [4, 0, 2]
        other_action, other_effects = uniforms_for([seeds[i] for i in chosen])
        np.testing.assert_array_equal(other_action, action[chosen])
        np.testing.assert_array_equal(other_effects, effects[chosen])
        self.assertTrue(np.all((action >= 0) & (action < 1)))
        self.assertTrue(np.all((effects >= 0) & (effects < 1)))
        self.assertFalse(np.array_equal(action[:, :, 0], action[:, :, 1]))
        self.assertFalse(np.array_equal(effects, action[:, 0]))

    def test_one_normal_effect_per_resident_persists_through_all_rounds(self):
        torch.manual_seed(17)
        model = ConditionalModel("H0").eval()
        model.requires_grad_(False)
        effects = np.array([[.1, .3, .7, .9], [.2, .4, .6, .8]])
        # Maximum returns keep every resident observable for forty rounds.
        actions = np.full((2, 40, 4), np.nextafter(1., 0.))
        expected = ndtri(effects) * float(model.sigma)
        captured = []

        def capture(logp, offers, tilt):
            captured.append(torch.as_tensor(tilt).numpy().copy())
            return tilt_log_mass(logp, offers, tilt)

        with patch("evopolis.institutional_rollout.tilt_log_mass", side_effect=capture):
            rows, paths, _ = simulate(model, "Equal", [11, 12], uniforms=(actions, effects), retain=2)
        self.assertEqual(len(captured), 40)
        for value in captured:
            np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(paths["sampled_u"], expected)
        self.assertTrue(np.all(~paths["padded"]))
        self.assertTrue(all(row["actual_rounds"] == 40 for row in rows))

    def test_fa_vectorized_actions_and_accounting_match_per_game_reference(self):
        for family in ("FA-GRU", "FA-P0", "FA-H0"):
            torch.manual_seed(17)
            model = make_model(family).eval()
            model.requires_grad_(False)
            for rule in RULES:
                with self.subTest(family=family, rule=rule):
                    seeds = seeds_for(family, 17, rule, 3)
                    rows, paths, _ = simulate(model, rule, seeds, retain=3, tilt=(.15, .25))
                    reference, slow = simulate_reference(model, rule, seeds, tilt=(.15, .25))
                    self.assertEqual(rows, reference)
                    for key in ("actions", "offers", "pool_before", "pool_after", "surplus", "sampled_u"):
                        np.testing.assert_array_equal(paths[key], slow[key])


if __name__ == "__main__":
    unittest.main()
