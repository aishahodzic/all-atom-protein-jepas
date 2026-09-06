"""Exact evaluation-only EMA behavior and checkpoint tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import lightning as L
import torch
from torch import nn

from proteinfoundation.jepa.canonical_lejepa_ema import CanonicalEncoderEMA
from proteinfoundation.jepa.tests.canonical_test_utils import tiny_canonical_config
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


class ToyCanonicalModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        self.unrelated = nn.Linear(4, 2)


class CanonicalEMATest(unittest.TestCase):
    def setUp(self):
        self.module = ToyCanonicalModule()
        self.trainer = SimpleNamespace(global_step=0)
        self.callback = CanonicalEncoderEMA(
            decay=0.9999,
            update_every=32,
            start_step=0,
            include=("encoder",),
            save_key="state_dict_ema",
            device="cpu",
            dtype="float32",
        )
        self.callback.on_train_start(self.trainer, self.module)

    def test_initialization_tracks_exactly_encoder_float_parameters(self):
        expected_names = {
            name for name, _ in self.module.named_parameters() if name.startswith("encoder.")
        }
        self.assertEqual(set(self.callback._ema_state), expected_names)
        current = self.module.state_dict()
        for name, value in self.callback._ema_state.items():
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.equal(value, current[name].float()))
        self.assertFalse(any(name.startswith("unrelated.") for name in self.callback._ema_state))
        self.assertFalse(hasattr(self.module, "ema_encoder"))

    def test_no_early_update_then_elapsed_step_decay_at_32(self):
        original = {name: value.clone() for name, value in self.callback._ema_state.items()}
        with torch.no_grad():
            for parameter in self.module.encoder.parameters():
                parameter.fill_(2.0)
        for step in (1, 7, 31):
            self.trainer.global_step = step
            self.callback.on_train_batch_end(
                self.trainer, self.module, None, None, 0
            )
            for name in original:
                self.assertTrue(torch.equal(self.callback._ema_state[name], original[name]))

        self.trainer.global_step = 32
        self.callback.on_train_batch_end(self.trainer, self.module, None, None, 0)
        update_decay = 0.9999**32
        current = self.module.state_dict()
        for name in original:
            expected = original[name] * update_decay + current[name].float() * (
                1.0 - update_decay
            )
            self.assertTrue(torch.allclose(self.callback._ema_state[name], expected))
        self.assertEqual(self.callback._last_ema_step, 32)
        self.assertEqual(self.callback._num_updates, 1)

    def test_checkpoint_and_callback_state_resume_without_raw_weight_swap(self):
        self.trainer.global_step = 32
        self.callback.on_train_batch_end(self.trainer, self.module, None, None, 0)
        raw_before = {
            name: value.clone() for name, value in self.module.state_dict().items()
        }
        checkpoint = {"state_dict": raw_before}
        self.callback.on_save_checkpoint(self.trainer, self.module, checkpoint)
        self.assertIn("state_dict_ema", checkpoint)
        self.assertIn("state_dict_ema_meta", checkpoint)
        self.assertEqual(checkpoint["state_dict_ema_meta"]["last_ema_step"], 32)
        self.assertEqual(
            checkpoint["state_dict_ema_meta"]["effective_decay"], 0.9999**32
        )
        for name, value in raw_before.items():
            self.assertTrue(torch.equal(self.module.state_dict()[name], value))

        resumed = CanonicalEncoderEMA(
            include=("encoder",), device="cpu", dtype="float32"
        )
        resumed.on_load_checkpoint(self.trainer, self.module, checkpoint)
        resumed.load_state_dict(self.callback.state_dict())
        resumed.on_train_start(self.trainer, self.module)
        self.assertEqual(resumed._last_ema_step, 32)
        self.assertEqual(resumed._num_updates, 1)
        for name, value in checkpoint["state_dict_ema"].items():
            self.assertTrue(torch.equal(resumed._ema_state[name], value))

    def test_current_encoder_has_no_persistent_nonparameter_state_needed_by_ema(self):
        encoder = EncoderTransformer(**tiny_canonical_config().nn_ae)
        parameter_names = {name for name, _ in encoder.named_parameters()}
        persistent_nonparameters = set(encoder.state_dict()) - parameter_names
        self.assertEqual(persistent_nonparameters, set())

    def test_100k_final_step_is_a_natural_scheduled_update(self):
        scheduled = [step for step in range(1, 100_001) if step % 32 == 0]
        self.assertEqual(scheduled[-1], 100_000)
        self.assertIn(100_000, scheduled)


if __name__ == "__main__":
    unittest.main()
