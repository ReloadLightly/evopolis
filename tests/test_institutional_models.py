import copy
import unittest

import numpy as np
import torch

from evopolis.conditional_models import ConditionalModel, group_loss
from evopolis.institutional_models import (
    FAConditional, FAGRU, augment_arrays, features_fa, initial_history,
    prepare_fa_group, prequential_fa, signal_sequence, signal_step,
)

torch.set_num_threads(1)


def example():
    offers = np.array([[50., 50., 50., 50.], [20., 30., 40., 50.],
                       [15., 30., 25., 40.], [30., 25., .8, 35.]])
    actions = np.array([[10, 20, 30, 40], [10, 20, 25, 40],
                        [10, 20, 15, 25], [20, 20, 0, 25]])
    return {"offers": offers.T[None], "y": actions.T[None],
            "pool": offers.sum(-1)[None]}


class InstitutionFeatureChecks(unittest.TestCase):
    def test_constructed_mixture_slopes_and_interpolating_formula(self):
        previous = torch.tensor([[1., 2., 3., 4.]] * 4, dtype=torch.float64)
        weights = torch.tensor([1., .5, 0., (.65 ** 22)], dtype=torch.float64)
        pools = torch.tensor([200., 150., 100., 130.], dtype=torch.float64)
        shares = weights[:, None] / 4 + (1 - weights[:, None]) * previous / previous.sum(-1)[:, None]
        offers = shares * pools[:, None]
        value, undefined, clipped = signal_step(offers, previous, torch.full((4,), .5), torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.allclose(value, 1 - weights, atol=1e-14, rtol=0))
        self.assertFalse(bool(undefined.any()))
        self.assertFalse(bool(clipped.any()))

    def test_undefined_carry_and_clipping(self):
        last, undefined = torch.tensor(.5), torch.tensor(True)
        previous = torch.zeros(4)
        last, undefined, clipped = signal_step(torch.ones(4), previous, last, undefined)
        self.assertEqual(float(last), .5)
        self.assertTrue(bool(undefined))
        previous = torch.tensor([1., 2., 3., 4.])
        last, undefined, _ = signal_step(previous * 10, previous, last, undefined)
        self.assertAlmostEqual(float(last), 1., places=14)
        for offers, actions in ((torch.zeros(4), previous), (torch.ones(4), torch.zeros(4)),
                                (torch.ones(4), torch.ones(4))):
            value, flag, clipped = signal_step(offers, actions, last, undefined)
            self.assertEqual(float(value), float(last))
            self.assertFalse(bool(flag))
            self.assertFalse(bool(clipped))
        value, flag, clipped = signal_step(torch.tensor([0., 100., 0., 0.]),
                                            torch.tensor([24., 26., 25., 25.]), last, undefined)
        self.assertEqual(float(value), 2.)
        self.assertTrue(bool(clipped))
        self.assertFalse(bool(flag))

    def test_signal_cannot_see_current_or_future_actions(self):
        arrays = example()
        offers, actions = arrays["offers"][0].T, arrays["y"][0].T
        baseline = signal_sequence(offers, actions)
        for origin in range(len(offers)):
            poisoned = actions.copy()
            poisoned[origin:] = 0
            changed = signal_sequence(offers, poisoned)
            for a, b in zip(baseline, changed):
                np.testing.assert_array_equal(a[:origin + 1], b[:origin + 1])
        self.assertEqual(baseline[0][0], .5)
        self.assertTrue(baseline[1][0])


class FAModelChecks(unittest.TestCase):
    def test_augmented_inputs_preserve_original_units(self):
        arrays = example()
        arrays["x"] = np.arange(4 * 4 * 9).reshape(1, 4, 4, 9) / 200
        augmented = augment_arrays(arrays)
        self.assertEqual(augmented["x"].shape, (1, 4, 4, 11))
        np.testing.assert_array_equal(augmented["x"][..., :9], arrays["x"])
        np.testing.assert_array_equal(augmented["x"][0, :, 0, 9:], [[.5, 1.]] * 4)
        model = FAGRU()
        values, hidden = model(torch.tensor(augmented["x"][0], dtype=torch.float32))
        self.assertEqual(tuple(values.shape), (4, 4, 5))
        self.assertEqual(tuple(hidden.shape), (1, 4, 32))

    def test_conditional_sequence_integral_matches_predictive_posteriors(self):
        torch.manual_seed(17)
        arrays = example()
        prepared = prepare_fa_group(arrays, 0)
        self.assertEqual(tuple(prepared["z"].shape), (4, 4, 14))
        for family in ("P0", "H0"):
            model = FAConditional(family)
            result = prequential_fa(model, arrays, 0, nodes=41, stop=4)
            self.assertAlmostEqual(float(group_loss(model, prepared, nodes=41)),
                                   float(-result["log_prob"].sum() / prepared["count"]), places=12)
            self.assertLess(float(abs(result["posterior_weights"].sum(-1) - 1).max()), 1e-12)
            self.assertLess(float(abs(result["pmfs"].sum(-1) - 1).max()), 1e-12)
            poisoned = copy.deepcopy(arrays)
            poisoned["y"][:, :, 2:] = 0
            changed = prequential_fa(model, poisoned, 0, nodes=41, stop=4)
            np.testing.assert_array_equal(result["pmfs"][:3], changed["pmfs"][:3])
            np.testing.assert_array_equal(result["posterior_weights"][:3], changed["posterior_weights"][:3])

    def test_zero_feature_weights_reduce_to_original_emission(self):
        original = ConditionalModel("H0")
        extended = FAConditional("H0")
        with torch.no_grad():
            extended.head.weight[:, :12].copy_(original.head.weight)
            extended.head.weight[:, 12:].zero_()
        offers = torch.tensor([10., 20., 30., 40.], dtype=torch.float64)
        history = initial_history()
        base = original.probs(offers.sum(), offers, history, u=.7)
        extra = extended.probs(offers.sum(), offers, history, u=.7, signal=1., undefined=False)
        self.assertTrue(torch.allclose(base, extra, atol=1e-14, rtol=1e-14))
        z = features_fa(offers.sum(), offers, history, .5, True)
        self.assertEqual(tuple(z.shape), (4, 14))
        self.assertTrue(torch.equal(z[:, -2:], torch.tensor([[.5, 1.]] * 4, dtype=torch.float64)))

    def test_shared_teacher_forcing_matches_fa_calibrated_latent_integration(self):
        from evopolis.institutional_predict import prequential
        arrays = example()
        torch.manual_seed(29)
        model = FAConditional("H0")
        for tilt in ((0., 0.), (.3, .6), (-.3, -1.)):
            expected = prequential_fa(model, arrays, 0, nodes=41, stop=4, tilt=tilt)
            actual = prequential(model, arrays, 0, tilt=tilt, nodes=41, stop=4)
            for key in ("pmfs", "log_prob", "posterior_weights"):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
