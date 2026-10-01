"""Consequential Task 04 forecast-boundary and reproducibility checks."""

import copy
import math
import unittest

import numpy as np
import torch

from evopolis.behavior_models import BehaviorModel
from evopolis.forecast_generate import branch_episode, generate_cell, simulate_branches
from evopolis.world import participant_observation


def fixture():
    arrays = {"pool": np.full((1, 40), 200.0), "next_pool": np.full((1, 40), 200.0),
              "offers": np.full((1, 4, 40), 49.99975), "y": np.full((1, 4, 40), 35, dtype=np.int64),
              "surplus": np.full((1, 4, 40), 14.99975), "x": np.zeros((1, 4, 40, 9))}
    for t in range(40):
        previous = None if t == 0 else arrays["y"][0, :, t-1]
        for p in range(4):
            arrays["x"][0, p, t] = np.asarray(participant_observation(p, 200., arrays["offers"][0, :, t], previous))/200
    group = {"index": 0, "key": ["Mixed Baseline Exp 1", "fixture", "0"], "condition": "Mixed Baseline Exp 1",
             "mechanism": "Mixed", "launch_id": "fixture", "episode_id": "0"}
    return arrays, group


class InstrumentedEmitter(torch.nn.Module):
    def __init__(self, *, recurrent=False, maximum=False):
        super().__init__()
        self.raw = torch.tensor([-1000, 1000, -1000, 0, 0] if maximum else [1000, -1000, -1000, 0, 0], dtype=torch.float32)
        self.recurrent = recurrent
        self.calls = []

    def forward(self, x, hidden=None):
        self.calls.append((x.clone(), None if hidden is None else hidden.clone()))
        if self.recurrent:
            if hidden is None:
                hidden = torch.zeros(1, x.shape[0], 32)
            hidden = hidden+x.shape[1]
        return self.raw.expand(*x.shape[:-1], 5), hidden


class ForecastGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recorded_unallocated_resource_and_integer_support_survive_first_boundary(self):
        arrays, group = fixture()
        model = InstrumentedEmitter()
        paths = simulate_branches(model, "constant", arrays, group, 0, [917], horizon=5)
        # The first source allocation leaves .001 resource unallocated. It is
        # neither erased nor silently reinjected after the next canonical step.
        residual = 200-math.fsum(arrays["offers"][0, :, 0])
        self.assertEqual(paths["pool_after"][0, 0], residual)
        self.assertEqual(paths["pool_after"][0, 1], 0)
        self.assertEqual(paths["actual_rounds"][0], 2)
        self.assertFalse(paths["padded"][0, 1])
        self.assertTrue(paths["padded"][0, 2:].all())
        self.assertEqual(paths["window_surplus"][0, -1], 200)
        self.assertEqual(len(model.calls), 2)
        maximum = simulate_branches(InstrumentedEmitter(maximum=True), "constant", arrays, group, 0, [917], horizon=1)
        np.testing.assert_array_equal(maximum["contributions"][0, 0], [49]*4)
        canonical = simulate_branches(InstrumentedEmitter(maximum=True), "constant", arrays, group, 0, [917], horizon=1, canonical_first=True)
        np.testing.assert_array_equal(canonical["contributions"][0, 0], [50]*4)
        canonical_zero = simulate_branches(InstrumentedEmitter(), "constant", arrays, group, 0, [917], horizon=5, canonical_first=True)
        self.assertEqual(canonical_zero["actual_rounds"][0], 1)
        np.testing.assert_array_equal(canonical_zero["pool_after"], 0)

    def test_recurrent_prefix_once_current_once_and_no_future_action_or_offer_leakage(self):
        arrays, group = fixture()
        arrays["y"][0, :, 4] = [11, 17, 23, 29]
        first = InstrumentedEmitter(recurrent=True, maximum=True)
        expected = simulate_branches(first, "recurrent", arrays, group, 5, [111, 222], horizon=5)
        self.assertEqual(first.calls[0][0].shape, (4, 5, 9))
        torch.testing.assert_close(first.calls[0][0], torch.tensor(arrays["x"][0, :, :5], dtype=torch.float32))
        self.assertIsNone(first.calls[0][1])
        self.assertEqual(first.calls[1][0].shape, (8, 1, 9))
        torch.testing.assert_close(first.calls[1][1], torch.full((1, 8, 32), 5.))
        for branch in range(2):
            for resident in range(4):
                current = np.asarray(participant_observation(resident, 200., arrays["offers"][0, :, 5], [11, 17, 23, 29]))/200
                np.testing.assert_allclose(first.calls[1][0][4*branch+resident, 0].numpy(), current, atol=1e-8)
        # Current contributions, future observations, and source future offers
        # are intentionally poisoned; the generated branch must be unchanged.
        poisoned = copy.deepcopy(arrays)
        poisoned["y"][:, :, 5:] = 0
        poisoned["x"][:, :, 5:] = 777
        poisoned["offers"][:, :, 6:] = 0
        poisoned["pool"][:, 6:] = 0
        poisoned["next_pool"][:] = 0
        actual = simulate_branches(InstrumentedEmitter(recurrent=True, maximum=True), "recurrent", poisoned, group, 5, [111, 222], horizon=5)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        # The next observation is generated from all four simultaneous returns.
        np.testing.assert_allclose(first.calls[2][0][0, 0, 4:8].numpy(), expected["contributions"][0, 0]/200, atol=1e-8)
        self.assertEqual(len(first.calls), 6)  # one warm-prefix call + five futures

    def test_endpoint_window_indexing_padding_and_observed_prefix_totals(self):
        arrays, group = fixture()
        checkpoint = {"family": "constant"}
        body, paths = generate_cell(InstrumentedEmitter(), checkpoint, arrays, group, 20, [1, 2], horizon=20)
        self.assertEqual([r["round_id"] for r in body["rounds"]], list(range(40)))
        self.assertTrue(all(r["observed"] for r in body["rounds"][:20]))
        self.assertTrue(all(not r["observed"] for r in body["rounds"][20:]))
        self.assertEqual(body["observed_endpoints"]["1"][0], arrays["next_pool"][0, 20]/200)
        self.assertAlmostEqual(body["observed_endpoints"]["5"][1], arrays["surplus"][0, :, 20:25].sum()/1000)
        self.assertEqual(body["endpoints"]["20"][0], [0., 200/4000])
        self.assertEqual(body["simulation_audit"]["executed_rounds"], 4)
        self.assertEqual(body["simulation_audit"]["padded_rounds"], 36)
        self.assertEqual(body["simulation_audit"]["max_abs_transition_residual"], 0)
        first = body["selected_branch"]["rounds"][0]
        self.assertAlmostEqual(first["cumulative_surplus"][0], 20*14.99975+49.99975)
        final = body["selected_branch"]["rounds"][-1]
        self.assertTrue(final["padded"])
        self.assertEqual(final["window_cumulative_surplus"], 200)
        with self.assertRaises(ValueError):
            simulate_branches(InstrumentedEmitter(), "constant", arrays, group, 21, [1], horizon=20)

    def test_paired_exact_checkpoint_copies_and_independent_resident_histories(self):
        arrays, group = fixture()
        torch.manual_seed(17)
        model = BehaviorModel("recurrent").eval()
        cloned = BehaviorModel("recurrent").eval()
        cloned.load_state_dict(model.state_dict())
        seeds = [817, 819, 823, 827]
        first = simulate_branches(model, "recurrent", arrays, group, 5, seeds, horizon=5)
        # Intervening unrelated rollout cannot contaminate any resident memory.
        simulate_branches(model, "recurrent", arrays, group, 10, [901], horizon=5)
        second = simulate_branches(cloned, "recurrent", arrays, group, 5, seeds, horizon=5)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertTrue(np.any(np.ptp(first["contributions"][:, 0], axis=1) > 0))
        self.assertTrue(np.any(first["contributions"][0] != first["contributions"][1]))


if __name__ == "__main__":
    unittest.main()
