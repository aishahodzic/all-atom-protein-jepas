import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from laproteina_checkpoint import extract_jepa_encoder_state_dict


class ExtractJepaEncoderStateDictTest(unittest.TestCase):
    def test_extracts_online_encoder_by_default(self):
        weights, prefix = extract_jepa_encoder_state_dict({
            "online_encoder.layer.weight": "online",
            "ema_encoder.layer.weight": "ema",
        })

        self.assertEqual(weights, {"layer.weight": "online"})
        self.assertEqual(prefix, "online_encoder")

    def test_target_accepts_local_ema_encoder_name(self):
        weights, prefix = extract_jepa_encoder_state_dict({
            "online_encoder.layer.weight": "online",
            "ema_encoder.layer.weight": "ema",
        }, "target")

        self.assertEqual(weights, {"layer.weight": "ema"})
        self.assertEqual(prefix, "ema_encoder")

    def test_target_prefers_explicit_target_encoder_name(self):
        weights, prefix = extract_jepa_encoder_state_dict({
            "target_encoder.layer.weight": "target",
            "ema_encoder.layer.weight": "ema",
        }, "target")

        self.assertEqual(weights, {"layer.weight": "target"})
        self.assertEqual(prefix, "target_encoder")

    def test_missing_target_branch_fails_clearly(self):
        with self.assertRaisesRegex(
            KeyError, "available encoder branches: \\['online_encoder'\\]"
        ):
            extract_jepa_encoder_state_dict({
                "online_encoder.layer.weight": "online",
            }, "target")


if __name__ == "__main__":
    unittest.main()
