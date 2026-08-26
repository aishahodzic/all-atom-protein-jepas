import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from laproteina_eval_policy import (  # noqa: E402
    UPSTREAM_L600_PROTOCOL,
    evaluation_cache_tag_suffix,
    evaluation_length_cache_suffix,
    resolve_evaluation_max_length,
    resolve_laproteina_evaluation_policy,
)


class LaProteinaEvaluationPolicyTest(unittest.TestCase):
    def test_default_uses_checkpoint_max_and_legacy_cache_name(self):
        effective = resolve_evaluation_max_length(256, None)

        self.assertEqual(effective, 256)
        self.assertEqual(evaluation_length_cache_suffix(None, effective), "")

    def test_explicit_long_evaluation_has_distinct_cache_identity(self):
        effective = resolve_evaluation_max_length(256, 600)

        self.assertEqual(effective, 600)
        self.assertEqual(
            evaluation_length_cache_suffix(600, effective), "_evalmax600"
        )

    def test_invalid_overrides_are_rejected(self):
        for invalid in [True, 0, -1, 600.5, "600"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    resolve_evaluation_max_length(256, invalid)

    def test_explicit_cache_tag_is_safe_and_distinct(self):
        self.assertEqual(evaluation_cache_tag_suffix(None), "")
        self.assertEqual(
            evaluation_cache_tag_suffix("featurefix-v1"),
            "_cache-featurefix-v1",
        )
        for invalid in ["contains space", "../escape", 1]:
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    evaluation_cache_tag_suffix(invalid)

    def test_named_protocol_locks_stb_cohort_without_whole_chain_filter(self):
        policy = resolve_laproteina_evaluation_policy(
            UPSTREAM_L600_PROTOCOL,
            encoder_source="online",
            evaluation_max_length=600,  # accepted migration spelling
        )

        self.assertEqual(policy.encoder_source, "online")
        self.assertEqual(policy.stb_filter_length, 600)
        self.assertIsNone(policy.evaluation_max_length)
        self.assertEqual(policy.cache_tag, UPSTREAM_L600_PROTOCOL)
        self.assertFalse(policy.apply_checkpoint_training_filters)
        self.assertTrue(policy.strict_cohort)
        self.assertTrue(policy.write_immutable_manifest)

    def test_named_protocol_rejects_conflicting_choices(self):
        conflicts = [
            {"encoder_source": "target"},
            {"evaluation_max_length": 512},
            {"cache_tag": "different"},
        ]
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                with self.assertRaises(ValueError):
                    resolve_laproteina_evaluation_policy(
                        UPSTREAM_L600_PROTOCOL, **conflict
                    )

    def test_no_protocol_preserves_historical_choices(self):
        policy = resolve_laproteina_evaluation_policy(
            None,
            encoder_source="target",
            evaluation_max_length=600,
            cache_tag="legacy-test",
        )

        self.assertIsNone(policy.name)
        self.assertEqual(policy.encoder_source, "target")
        self.assertEqual(policy.evaluation_max_length, 600)
        self.assertEqual(policy.cache_tag, "legacy-test")
        self.assertTrue(policy.apply_checkpoint_training_filters)


if __name__ == "__main__":
    unittest.main()
