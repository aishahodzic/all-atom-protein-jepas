"""Tests for the isolated canonical EMA/raw downstream checkpoint path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import torch

from Evaluation.canonical_lejepa.checkpoint import (
    extract_canonical_encoder_state_dict,
    load_canonical_encoder,
    stage_structtokenbench_checkpoint,
)
from proteinfoundation.jepa.tests.canonical_test_utils import tiny_canonical_config
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


class CanonicalCheckpointTest(unittest.TestCase):
    def _checkpoint(self):
        encoder = EncoderTransformer(**tiny_canonical_config().nn_ae)
        raw = {f"encoder.{name}": value.clone() for name, value in encoder.state_dict().items()}
        ema = {
            name: (value.clone() + 1.0 if value.is_floating_point() else value.clone())
            for name, value in raw.items()
        }
        return encoder, {"state_dict": raw, "state_dict_ema": ema}

    def test_default_is_ema_raw_is_explicit_and_missing_ema_fails(self):
        _, checkpoint = self._checkpoint()
        ema = extract_canonical_encoder_state_dict(checkpoint)
        raw = extract_canonical_encoder_state_dict(checkpoint, source="raw")
        key = "latent_decoder_mean_n_log_scale.1.weight"
        self.assertTrue(torch.equal(ema[key], raw[key] + 1.0))
        with self.assertRaisesRegex(KeyError, "state_dict_ema"):
            extract_canonical_encoder_state_dict({"state_dict": checkpoint["state_dict"]})

    def test_strict_eight_dimensional_validation(self):
        _, checkpoint = self._checkpoint()
        checkpoint["state_dict_ema"][
            "encoder.latent_decoder_mean_n_log_scale.1.weight"
        ] = torch.zeros(128, 16)
        with self.assertRaisesRegex(ValueError, "8D"):
            extract_canonical_encoder_state_dict(checkpoint)

    def test_load_and_stage_round_trip(self):
        _, checkpoint = self._checkpoint()
        cfg = tiny_canonical_config()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint_path = directory / "canonical.ckpt"
            config_path = directory / "config.json"
            staged_ema_path = directory / "staged-ema.ckpt"
            staged_raw_path = directory / "staged-raw.ckpt"
            torch.save(checkpoint, checkpoint_path)
            config_path.write_text(
                json.dumps({"nn_ae": OmegaConf.to_container(cfg.nn_ae, resolve=True)}),
                encoding="utf-8",
            )

            loaded = load_canonical_encoder(checkpoint_path, config_path)
            self.assertFalse(loaded.training)
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            expected = extract_canonical_encoder_state_dict(checkpoint)
            for name, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]))

            stage_structtokenbench_checkpoint(checkpoint_path, staged_ema_path)
            stage_structtokenbench_checkpoint(
                checkpoint_path, staged_raw_path, source="raw"
            )
            staged_ema = torch.load(staged_ema_path, weights_only=False)
            staged_raw = torch.load(staged_raw_path, weights_only=False)
            self.assertEqual(
                staged_ema["canonical_lejepa_evaluation"]["encoder_source"], "ema"
            )
            self.assertEqual(
                staged_raw["canonical_lejepa_evaluation"]["encoder_source"], "raw"
            )
            expected_source_hash = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            for staged in (staged_ema, staged_raw):
                provenance = staged["canonical_lejepa_evaluation"]
                self.assertEqual(
                    provenance["source_checkpoint_sha256"], expected_source_hash
                )
                self.assertNotIn("source_checkpoint", provenance)
            staged_key = "online_encoder.latent_decoder_mean_n_log_scale.1.weight"
            self.assertFalse(
                torch.equal(
                    staged_ema["state_dict"][staged_key],
                    staged_raw["state_dict"][staged_key],
                )
            )
            self.assertNotEqual(staged_ema_path.read_bytes(), staged_raw_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
