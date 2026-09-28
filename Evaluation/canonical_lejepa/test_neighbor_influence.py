"""Small unit tests for neighbour-influence metric safeguards."""

import math
import unittest

import torch

from Evaluation.canonical_lejepa.run_neighbor_influence import _response_metrics


class NeighborInfluenceMetricTest(unittest.TestCase):
    def test_identical_output_has_zero_response(self):
        result = _response_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), 2.0)
        self.assertEqual(result["response_l2"], 0.0)
        self.assertEqual(result["transmission_gain"], 0.0)
        self.assertAlmostEqual(result["cosine_change"], 0.0)

    def test_gain_and_relative_response(self):
        result = _response_metrics(torch.tensor([3.0, 0.0]), torch.tensor([3.0, 4.0]), 2.0)
        self.assertEqual(result["response_l2"], 4.0)
        self.assertEqual(result["transmission_gain"], 2.0)
        self.assertAlmostEqual(result["relative_response"], 4.0 / 3.0)
        self.assertTrue(math.isfinite(result["cosine_change"]))

    def test_zero_input_delta_is_rejected(self):
        with self.assertRaises(ValueError):
            _response_metrics(torch.ones(2), torch.zeros(2), 0.0)

    def test_zero_output_norm_is_guarded(self):
        result = _response_metrics(torch.zeros(2), torch.ones(2), 1.0)
        self.assertTrue(math.isnan(result["cosine_change"]))


if __name__ == "__main__":
    unittest.main()
