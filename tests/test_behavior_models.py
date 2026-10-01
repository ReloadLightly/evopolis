"""Consequential emission, information-boundary, and group-weight checks.

All fixtures are synthetic. Running these checks never opens Task 03 test
outcomes or uses a fitted checkpoint for selection.
"""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import betabinom
import torch

from evopolis.behavior_models import (
    BehaviorModel, emission_log_prob, emission_probs, emission_stats, group_nll,
)
from evopolis.behavior_evaluate import score_groups, summarize_predictions, METRICS


class EmissionTests(unittest.TestCase):
    def test_normalization_endpoints_and_independent_scipy_reference(self):
        vector = np.array([0.7, -0.4, 1.1, -2.0, 3.0])
        weights = np.exp(vector[:3] - np.max(vector[:3]))
        weights /= weights.sum()
        alpha, beta = np.logaddexp(0, vector[3:]) + 0.05
        raw = torch.tensor(vector, dtype=torch.float64)
        for maximum in (0, 1, 2, 49, 50, 199, 200):
            n = torch.tensor(maximum)
            probabilities = emission_probs(raw, n).numpy()
            expected = weights[2] * betabinom.pmf(np.arange(maximum + 1), maximum, alpha, beta)
            expected[0] += weights[0]
            expected[-1] += weights[1]
            np.testing.assert_allclose(probabilities[:maximum + 1], expected, rtol=2e-11, atol=2e-13)
            self.assertAlmostEqual(probabilities.sum(), 1.0, places=11)
            self.assertTrue(np.all(probabilities[maximum + 1:] == 0))
            self.assertTrue(np.all(probabilities[:maximum + 1] > 0))
            stats = emission_stats(raw, n)
            self.assertAlmostEqual(float(stats["mean"]), float(np.arange(201) @ probabilities), places=10)
        bernoulli = emission_probs(raw, torch.tensor(1)).numpy()
        self.assertAlmostEqual(bernoulli[0], weights[0] + weights[2] * beta / (alpha + beta))
        self.assertAlmostEqual(bernoulli[1], weights[1] + weights[2] * alpha / (alpha + beta))

    def test_finite_gradients_and_forced_actions_have_zero_loss_gradient(self):
        raw = torch.tensor([[9.0, -9.0, -3.0, -15.0, 12.0], [-9.0, 9.0, 2.0, 12.0, -15.0]], requires_grad=True)
        loss = -emission_log_prob(raw, torch.tensor([200, 200]), torch.tensor([1, 199])).mean()
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertTrue(torch.isfinite(raw.grad).all())
        forced = torch.randn(4, 5, requires_grad=True)
        forced_loss = -emission_log_prob(forced, torch.zeros(4), torch.zeros(4)).sum()
        self.assertEqual(float(forced_loss.detach()), 0.0)
        forced_loss.backward()
        self.assertTrue(torch.equal(forced.grad, torch.zeros_like(forced)))
        for n, c in ((1, 2), (0, -1), (2, 0.5), (1.5, 1)):
            with self.assertRaises(ValueError):
                emission_log_prob(torch.zeros(5), torch.tensor(n), torch.tensor(c))

    def test_exact_discrete_sampling_keeps_boundary_support_and_distribution(self):
        raw = torch.tensor([0.7, -0.2, 0.1, -0.3, 0.4])
        generator = torch.Generator().manual_seed(61003)
        for n in (0, 1, 7, 200):
            probabilities = emission_probs(raw, torch.tensor(n))
            draws = torch.multinomial(probabilities, 12000, replacement=True, generator=generator)
            self.assertGreaterEqual(int(draws.min()), 0)
            self.assertLessEqual(int(draws.max()), n)
            if n:
                expected = float(emission_stats(raw, torch.tensor(n))["mean"])
                variance = float(((torch.arange(201) - expected) ** 2 * probabilities).sum())
                self.assertLess(abs(float(draws.double().mean()) - expected), 6 * math.sqrt(variance / len(draws)))


class ObservationAndAggregationTests(unittest.TestCase):
    def test_parameter_counts_causal_recurrence_independent_residents_and_reset(self):
        torch.manual_seed(721)
        expected_counts = {"constant": 5, "linear": 50, "feedforward": 4285, "recurrent": 4293}
        for family, count in expected_counts.items():
            model = BehaviorModel(family)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
        model = BehaviorModel("recurrent").eval()
        x = torch.rand(4, 40, 9)
        with torch.no_grad():
            original, _ = model(x)
            contaminated = x.clone()
            contaminated[:, 12:] = 1000
            modified, _ = model(contaminated)
            torch.testing.assert_close(original[:, :12], modified[:, :12], rtol=0, atol=0)
            # Group/position history is isolated by a separate hidden-state axis.
            one, _ = model(x[2:3])
            torch.testing.assert_close(one, original[2:3], atol=1e-7, rtol=1e-6)
            state, streamed = None, []
            for time in range(40):
                prediction, state = model(x[:, time:time + 1], state)
                streamed.append(prediction)
            torch.testing.assert_close(torch.cat(streamed, dim=1), original, atol=1e-7, rtol=1e-6)
            repeated, _ = model(x)
            torch.testing.assert_close(repeated, original, rtol=0, atol=0)
            # Forced observations remain in memory even though their loss is zero.
            _, state_after_forced = model(x[:, :1])
            after_history, _ = model(x[:, 1:2], state_after_forced)
            without_history, _ = model(x[:, 1:2])
            self.assertFalse(torch.equal(after_history, without_history))

    def test_group_balancing_does_not_weight_long_active_groups_more(self):
        raw = torch.zeros(2, 4, 40, 5)
        n = torch.ones(2, 4, 40, dtype=torch.long)
        y = torch.zeros_like(n)
        n[0, :, 1:] = 0
        raw[0, ..., 0] = 2.0
        raw[1, ..., 1] = 2.0
        losses = -emission_log_prob(raw, n, y)
        expected = torch.stack((losses[0, :, 0].mean(), losses[1].mean()))
        torch.testing.assert_close(group_nll(raw, n, y), expected)
        self.assertNotAlmostEqual(float(expected.mean()), float(losses.sum() / (n >= 1).sum()))
        # Relative MAE divides by observed 1.9, not floor(1.9)=1.
        groups = [{"index": i, "key": ["fixture", str(i), "0"], "mechanism": "Equal"} for i in range(2)]
        scores, _ = score_groups(None, n.numpy(), y.numpy(), np.full(n.shape, 1.9), groups, "uniform", "reference", "test-fixture")
        self.assertAlmostEqual(scores[0]["relative_mae"], 0.5 / 1.9)

    def test_paired_bootstrap_averages_seeds_before_human_group_resampling(self):
        rows = []
        for family in ("recurrent", "feedforward", "uniform"):
            for mechanism in ("Equal", "Mixed"):
                for group in range(3):
                    for seed in (("reference",) if family == "uniform" else (17, 29, 43)):
                        score = group + (2.0 if family == "uniform" else (0.0 if family == "recurrent" else 0.5))
                        if seed != "reference":
                            score += {17: -0.1, 29: 0, 43: 0.1}[seed]
                        rows.append({"family": family, "seed": seed, "mechanism": mechanism,
                                     "group_index": group + (0 if mechanism == "Equal" else 3),
                                     **{metric: score for metric in METRICS}})
        result = summarize_predictions(rows, bootstrap_replicates=100)
        difference = result["paired_comparisons"][0]
        self.assertEqual(difference["comparison"], "recurrent - feedforward")
        self.assertAlmostEqual(difference["nll_difference"], -0.5)
        np.testing.assert_allclose(difference["ci95"], [-0.5, -0.5], atol=1e-14)

    def test_resume_matches_uninterrupted_optimizer_and_shuffle_exactly(self):
        from evopolis.behavior_train import atomic_checkpoint, new_fit, train_epoch

        fixture_generator = torch.Generator().manual_seed(73017)
        data = {"x": torch.rand(8, 4, 40, 9, generator=fixture_generator),
                "n": torch.full((8, 4, 40), 4, dtype=torch.long),
                "y": torch.ones(8, 4, 40, dtype=torch.long)}
        indices = np.arange(8)
        uninterrupted, optimizer, shuffle = new_fit("recurrent", 17)
        train_epoch(uninterrupted, optimizer, data, indices, shuffle)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            atomic_checkpoint(path, {"model_state": uninterrupted.state_dict(),
                                    "optimizer_state": optimizer.state_dict(),
                                    "torch_rng_state": torch.get_rng_state(),
                                    "shuffle_state": shuffle.bit_generator.state})
            expected_loss = train_epoch(uninterrupted, optimizer, data, indices, shuffle)
            resumed, new_optimizer, new_shuffle = new_fit("recurrent", 17)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            resumed.load_state_dict(saved["model_state"])
            new_optimizer.load_state_dict(saved["optimizer_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            new_shuffle.bit_generator.state = saved["shuffle_state"]
            actual_loss = train_epoch(resumed, new_optimizer, data, indices, new_shuffle)
            self.assertEqual(actual_loss, expected_loss)
            for name, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
            self.assertEqual(new_shuffle.bit_generator.state, shuffle.bit_generator.state)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
