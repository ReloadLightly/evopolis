"""Learned simulation accounting, fixed horizon and independent replay checks."""

import unittest

import numpy as np
import torch

from evopolis.behavior_generate import generate_game, sample_integer
from evopolis.behavior_models import BehaviorModel, emission_probs


class FixedEmitter(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters_raw = torch.tensor(parameters, dtype=torch.float32)
        self.inputs = []

    def forward(self, x, hidden=None):
        self.inputs.append(x.clone())
        return self.parameters_raw.expand(*x.shape[:-1], 5), None


class BehavioralGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_termination_pads_to_full_horizon_and_maximum_can_have_undefined_gini(self):
        identity = {"family": "constant", "seed": 17, "epoch": 120}
        never = FixedEmitter([1000, -1000, -1000, 0, 0])
        game = generate_game(never, identity, "fixture", rollout_seed=99, mechanism="equal")
        self.assertEqual(game["actual_rounds"], 1)
        self.assertEqual(game["round_count"], 40)
        self.assertEqual(game["surplus"], 200 / 160)
        self.assertEqual(game["gini"], 0)
        self.assertEqual(game["active_allocation_fraction"], 4 / 160)
        self.assertTrue(game["exact_zero_termination"])
        self.assertEqual(len(never.inputs), 1)
        self.assertEqual(never.inputs[0].shape, (4, 1, 9))
        for row in game["rounds"][1:]:
            self.assertTrue(row["padded"])
            self.assertEqual(row["cumulative_surplus"], [50.0] * 4)
            self.assertEqual(row["surplus"], [0.0] * 4)
            self.assertFalse(row["predictions"][0]["evaluated"])
        always = FixedEmitter([-1000, 1000, -1000, 0, 0])
        game = generate_game(always, identity, "fixture", rollout_seed=99, mechanism="equal")
        self.assertEqual(game["actual_rounds"], 40)
        self.assertEqual(game["surplus"], 0)
        self.assertIsNone(game["gini"])
        self.assertEqual(game["final_pool"], 200)
        self.assertTrue(game["final_sustainment"])
        self.assertFalse(game["depleted"])

    def test_sampling_matches_overlapping_endpoint_mass_and_independent_resident_draws(self):
        from evopolis.behavior_models import emission_parts

        raw = torch.tensor([0.2, -0.3, 0.1, 0.9, -0.7])
        logs, alpha, beta = emission_parts(raw)
        for n in (0, 1, 7):
            probabilities = emission_probs(raw, torch.tensor(n)).numpy()[:n + 1]
            rng = np.random.Generator(np.random.PCG64(917 + n))
            samples = [sample_integer(rng, n, logs.exp().numpy(), alpha.item(), beta.item()) for _ in range(20000)]
            observed = np.bincount(samples, minlength=n + 1) / len(samples)
            np.testing.assert_allclose(observed, probabilities, atol=0.012, rtol=0)
        resident_rng = np.random.Generator(np.random.PCG64(5001))
        draws = np.asarray([sample_integer(resident_rng, 50, logs.exp().numpy(), alpha.item(), beta.item()) for _ in range(400)]).reshape(100, 4)
        self.assertTrue(np.any(np.ptp(draws, axis=1) > 0))

    def test_all_families_replay_seeds_and_recurrent_memory_does_not_cross_games(self):
        for family in ("constant", "linear", "feedforward", "recurrent"):
            with self.subTest(family=family):
                torch.manual_seed(23)
                model = BehaviorModel(family).eval()
                checkpoint = {"family": family, "seed": 23, "epoch": 1}
                first = generate_game(model, checkpoint, "fixture", rollout_seed=1988, mechanism="mixed")
                generate_game(model, checkpoint, "fixture", rollout_seed=1990, mechanism="proportional")
                replay = generate_game(model, checkpoint, "fixture", rollout_seed=1988, mechanism="mixed")
                self.assertEqual(first, replay)
                for row in first["rounds"]:
                    for offer, contribution, prediction in zip(row["offers"], row["contributions"], row["predictions"]):
                        self.assertEqual(prediction["legal_max"], int(np.floor(offer)))
                        self.assertGreaterEqual(contribution, 0)
                        self.assertLessEqual(contribution, prediction["legal_max"])


if __name__ == "__main__":
    unittest.main()
