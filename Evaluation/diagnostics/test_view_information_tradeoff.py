import unittest

import numpy as np
import torch

from Evaluation.diagnostics.analyze_canonical_view_information_tradeoff import (
    ResidueAgreement,
    summarize_pooled_views,
)


class ViewInformationTradeoffTests(unittest.TestCase):
    def test_identical_views_have_perfect_agreement_and_retrieval(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(12, 6))
        values = np.repeat(base[:, None, :], 5, axis=1)
        result = summarize_pooled_views(values)
        self.assertAlmostEqual(result["raw_paired_cosine"], 1.0)
        self.assertAlmostEqual(result["standardized_paired_cosine"], 1.0)
        self.assertAlmostEqual(result["linear_cka_mean"], 1.0)
        self.assertAlmostEqual(result["cross_view_protein_retrieval_top1"], 1.0)
        self.assertAlmostEqual(result["paired_standardized_mse"], 0.0)

    def test_shuffled_views_reduce_identity_retrieval(self):
        values = np.repeat(np.eye(10)[:, None, :], 5, axis=1)
        for local in range(1, 5):
            values[:, local] = np.roll(values[:, local], 1, axis=0)
        result = summarize_pooled_views(values)
        self.assertEqual(result["cross_view_protein_retrieval_top1"], 0.0)
        self.assertLess(result["standardized_paired_cosine"], 0.0)

    def test_exact_residue_retrieval(self):
        metric = ResidueAgreement()
        values = torch.eye(7)
        metric.add(values, values)
        result = metric.summary()
        self.assertEqual(result["residue_pairs"], 7)
        self.assertEqual(result["within_protein_retrieval_top1"], 1.0)
        self.assertEqual(result["paired_cosine"], 1.0)
        self.assertGreater(result["cosine_margin"], 0.9)


if __name__ == "__main__":
    unittest.main()
