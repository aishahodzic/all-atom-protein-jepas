"""Parity tests for residue-view activation checkpointing."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

import proteinfoundation.jepa.canonical_lejepa_model as canonical_model_module
from proteinfoundation.jepa.canonical_lejepa_model import (
    CANONICAL_NUM_VIEWS,
    CanonicalLeJEPAModel,
    build_canonical_view_masks,
    canonical_residue_invariance_loss,
)
from proteinfoundation.jepa.tests.canonical_test_utils import (
    clone_batch,
    synthetic_encoder_batch,
    tiny_canonical_config,
)


MASKED_KV_MODES = ("iterative", "visible_only")
FORWARD_SEED = 6143
SIGREG_SEED = 12911


class CanonicalResidueActivationCheckpointingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = synthetic_encoder_batch(batch_size=3, length=8)
        torch.manual_seed(3109)
        self.view_masks = build_canonical_view_masks(self.batch)

    @staticmethod
    def _matched_models(
        mode: str,
    ) -> tuple[CanonicalLeJEPAModel, CanonicalLeJEPAModel]:
        torch.manual_seed(2081)
        direct = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                masked_kv_mode=mode,
                residue_view_checkpointing=False,
            )
        ).train()
        checkpointed = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                masked_kv_mode=mode,
                residue_view_checkpointing=True,
            )
        ).train()
        checkpointed.load_state_dict(copy.deepcopy(direct.state_dict()))
        return direct, checkpointed

    def _encode(self, model: CanonicalLeJEPAModel):
        torch.manual_seed(FORWARD_SEED)
        with mock.patch.object(
            canonical_model_module,
            "build_canonical_view_masks",
            return_value=self.view_masks,
        ):
            return model.encode_residue_views(clone_batch(self.batch))

    def _controlled_total(self, model: CanonicalLeJEPAModel):
        residue_embeddings, embeddings, view_masks, real_mask = self._encode(model)
        inv_loss = canonical_residue_invariance_loss(
            residue_embeddings,
            real_mask,
        )
        torch.manual_seed(SIGREG_SEED)
        sigreg_loss = model.sigreg(embeddings.permute(1, 0, 2))
        total_loss = inv_loss + model.sigreg_weight * sigreg_loss
        return {
            "residue_embeddings": residue_embeddings,
            "embeddings": embeddings,
            "view_masks": view_masks,
            "real_mask": real_mask,
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "loss": total_loss,
        }

    def test_01_checkpointed_and_direct_residue_embeddings_match(self):
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                direct, checkpointed = self._matched_models(mode)
                direct_output = self._encode(direct)
                checkpoint_target = canonical_model_module.activation_checkpoint
                with mock.patch.object(
                    canonical_model_module,
                    "activation_checkpoint",
                    wraps=checkpoint_target,
                ) as checkpoint_spy:
                    checkpointed_output = self._encode(checkpointed)

                self.assertEqual(checkpoint_spy.call_count, CANONICAL_NUM_VIEWS)
                for call in checkpoint_spy.call_args_list:
                    self.assertIs(call.kwargs["use_reentrant"], False)
                    self.assertIs(call.kwargs["preserve_rng_state"], True)
                for direct_value, checkpointed_value in zip(
                    direct_output,
                    checkpointed_output,
                    strict=True,
                ):
                    self.assertTrue(torch.equal(direct_value, checkpointed_value))

    def test_02_total_loss_matches_with_controlled_sigreg_rng(self):
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                direct, checkpointed = self._matched_models(mode)
                direct_output = self._controlled_total(direct)
                checkpointed_output = self._controlled_total(checkpointed)
                for name in ("inv_loss", "sigreg_loss", "loss"):
                    self.assertTrue(
                        torch.equal(direct_output[name], checkpointed_output[name]),
                        name,
                    )

    def test_03_encoder_gradients_match(self):
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                direct, checkpointed = self._matched_models(mode)
                direct.zero_grad(set_to_none=True)
                checkpointed.zero_grad(set_to_none=True)
                self._controlled_total(direct)["loss"].backward()
                self._controlled_total(checkpointed)["loss"].backward()

                direct_gradients = dict(direct.encoder.named_parameters())
                checkpointed_gradients = dict(checkpointed.encoder.named_parameters())
                self.assertEqual(direct_gradients.keys(), checkpointed_gradients.keys())
                nonzero_gradient = False
                for name in direct_gradients:
                    direct_gradient = direct_gradients[name].grad
                    checkpointed_gradient = checkpointed_gradients[name].grad
                    self.assertEqual(
                        direct_gradient is None,
                        checkpointed_gradient is None,
                        name,
                    )
                    if direct_gradient is None:
                        continue
                    self.assertTrue(torch.isfinite(direct_gradient).all(), name)
                    self.assertTrue(torch.isfinite(checkpointed_gradient).all(), name)
                    self.assertTrue(
                        torch.equal(direct_gradient, checkpointed_gradient),
                        name,
                    )
                    nonzero_gradient |= bool(direct_gradient.count_nonzero())
                self.assertTrue(nonzero_gradient)

    def test_04_checkpointing_preserves_rng_state(self):
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                direct, checkpointed = self._matched_models(mode)

                def run(model: CanonicalLeJEPAModel):
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(FORWARD_SEED)
                    with mock.patch.object(
                        canonical_model_module,
                        "build_canonical_view_masks",
                        return_value=self.view_masks,
                    ):
                        loss = model.compute_loss(clone_batch(self.batch))["loss"]
                    state_after_forward = torch.get_rng_state().clone()
                    loss.backward()
                    state_after_backward = torch.get_rng_state().clone()
                    return state_after_forward, state_after_backward

                direct_states = run(direct)
                checkpointed_states = run(checkpointed)
                self.assertTrue(torch.equal(direct_states[0], checkpointed_states[0]))
                self.assertTrue(torch.equal(direct_states[1], checkpointed_states[1]))

    def test_05_centered_sigreg_checkpointing_preserves_loss_gradients_and_rng(
        self,
    ):
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(2081)
                direct = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue",
                        masked_kv_mode=mode,
                        residue_view_checkpointing=False,
                        sigreg_input="protein_centered_residue",
                    )
                ).train()
                checkpointed = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue",
                        masked_kv_mode=mode,
                        residue_view_checkpointing=True,
                        sigreg_input="protein_centered_residue",
                    )
                ).train()
                checkpointed.load_state_dict(copy.deepcopy(direct.state_dict()))

                def run(model: CanonicalLeJEPAModel):
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(FORWARD_SEED)
                    with mock.patch.object(
                        canonical_model_module,
                        "build_canonical_view_masks",
                        return_value=self.view_masks,
                    ):
                        output = model.compute_loss(clone_batch(self.batch))
                    state_after_forward = torch.get_rng_state().clone()
                    output["loss"].backward()
                    state_after_backward = torch.get_rng_state().clone()
                    gradients = {
                        name: None
                        if parameter.grad is None
                        else parameter.grad.detach().clone()
                        for name, parameter in model.encoder.named_parameters()
                    }
                    return (
                        output,
                        gradients,
                        state_after_forward,
                        state_after_backward,
                    )

                (
                    direct_output,
                    direct_gradients,
                    direct_forward_rng,
                    direct_backward_rng,
                ) = run(direct)
                (
                    checkpointed_output,
                    checkpointed_gradients,
                    checkpointed_forward_rng,
                    checkpointed_backward_rng,
                ) = run(checkpointed)

                for name in (
                    "residue_embeddings",
                    "embeddings",
                    "inv_loss",
                    "sigreg_loss",
                    "loss",
                ):
                    self.assertTrue(
                        torch.equal(direct_output[name], checkpointed_output[name]),
                        name,
                    )
                self.assertEqual(
                    direct_gradients.keys(), checkpointed_gradients.keys()
                )
                for name in direct_gradients:
                    if direct_gradients[name] is None:
                        self.assertIsNone(checkpointed_gradients[name], name)
                    else:
                        self.assertTrue(
                            torch.equal(
                                direct_gradients[name], checkpointed_gradients[name]
                            ),
                            name,
                        )
                self.assertTrue(
                    torch.equal(direct_forward_rng, checkpointed_forward_rng)
                )
                self.assertTrue(
                    torch.equal(direct_backward_rng, checkpointed_backward_rng)
                )


if __name__ == "__main__":
    unittest.main()
