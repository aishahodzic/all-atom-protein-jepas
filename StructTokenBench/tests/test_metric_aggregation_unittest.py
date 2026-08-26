"""Stdlib-unittest wrappers so the pytest-style regression tests cannot skip."""

import unittest

import test_metric_aggregation as checks


class MetricAggregationRegressionTest(unittest.TestCase):
    def test_binary_auroc_pooling(self):
        checks.test_binary_auroc_is_computed_after_pooling_not_meaned_per_batch()

    def test_binary_rebatching(self):
        checks.test_binary_metrics_are_invariant_to_rebatching()

    def test_binary_logit_threshold(self):
        checks.test_binary_logits_use_zero_decision_threshold_even_when_all_are_in_zero_one()

    def test_multiclass_rebatching(self):
        checks.test_multiclass_metrics_are_invariant_to_rebatching()

    def test_fixed_binary_weights(self):
        checks.test_fixed_binary_weights_are_derived_once_from_global_counts()

    def test_single_class_legacy_weight(self):
        checks.test_legacy_weighting_no_longer_divides_by_zero_on_single_class_batch()

    def test_mixed_class_legacy_weight_arithmetic(self):
        checks.test_legacy_mixed_batch_matches_upstream_float32_arithmetic_order()

    def test_non_distributed_gather(self):
        checks.test_gather_is_identity_without_distributed_initialization()

    def test_pooled_checkpoint_metric(self):
        checks.test_pl_model_wires_pooled_metric_into_checkpoint_metric_name()

    def test_no_step_level_global_metric_names(self):
        checks.test_pl_model_does_not_step_log_names_reserved_for_global_metrics()


if __name__ == "__main__":
    unittest.main()
