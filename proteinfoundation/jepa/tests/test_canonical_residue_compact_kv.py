"""Dense-versus-rectangular parity tests for Option-B residue attention."""

from __future__ import annotations

import copy
import unittest
from contextlib import ExitStack
from unittest import mock

import torch

import proteinfoundation.jepa.canonical_lejepa_model as canonical_model_module
from proteinfoundation.jepa.canonical_lejepa_model import (
    CanonicalLeJEPAModel,
    build_canonical_view_masks,
    mask_batch_for_canonical_residue_view,
)
from proteinfoundation.jepa.tests.canonical_test_utils import (
    clone_batch,
    synthetic_encoder_batch,
    tiny_canonical_config,
)
from proteinfoundation.nn.modules.pair_bias_attn import (
    MultiHeadBiasedAttentionADALN_MM,
    PairBiasAttention,
)


ATOL = 4e-6
RTOL = 4e-5
FORWARD_SEED = 8117


def _assert_close(test: unittest.TestCase, left, right, name: str) -> None:
    if left.dtype == torch.bool or not left.is_floating_point():
        test.assertTrue(torch.equal(left, right), name)
    else:
        torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL, msg=name)


def _matched_models(*, checkpointed_compact: bool = False):
    torch.manual_seed(7201)
    dense = CanonicalLeJEPAModel(
        tiny_canonical_config(
            alignment_level="residue",
            masked_kv_mode="visible_only",
            residue_view_checkpointing=False,
            residue_compact_visible_kv=False,
            sigreg_input="protein_centered_residue",
        )
    ).train()
    compact = CanonicalLeJEPAModel(
        tiny_canonical_config(
            alignment_level="residue",
            masked_kv_mode="visible_only",
            residue_view_checkpointing=checkpointed_compact,
            residue_compact_visible_kv=True,
            sigreg_input="protein_centered_residue",
        )
    ).train()
    compact.load_state_dict(copy.deepcopy(dense.state_dict()), strict=True)
    return dense, compact


def _run_model(model, batch, view_masks, *, backward: bool):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(FORWARD_SEED)
    with mock.patch.object(
        canonical_model_module,
        "build_canonical_view_masks",
        return_value=view_masks.clone(),
    ):
        output = model.compute_loss(clone_batch(batch))
    state_after_forward = torch.get_rng_state().clone()
    if backward:
        output["loss"].backward()
    state_after_backward = torch.get_rng_state().clone()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.encoder.named_parameters()
    }
    return output, gradients, state_after_forward, state_after_backward


class CompactKeyLayoutTest(unittest.TestCase):
    def test_ordered_ragged_layout_and_pair_columns(self):
        key_mask = torch.tensor(
            [
                [False, True, False, True, True, False],
                [True, False, False, False, False, False],
            ]
        )
        indices, compact_mask = PairBiasAttention._compact_key_layout(key_mask)
        self.assertTrue(torch.equal(indices, torch.tensor([[1, 3, 4], [0, 0, 0]])))
        self.assertTrue(
            torch.equal(
                compact_mask,
                torch.tensor([[True, True, True], [True, False, False]]),
            )
        )

        rows = torch.arange(2 * 6 * 2).reshape(2, 6, 2)
        gathered_rows = PairBiasAttention._gather_key_rows(rows, indices)
        self.assertTrue(torch.equal(gathered_rows[0], rows[0, [1, 3, 4]]))
        self.assertTrue(torch.equal(gathered_rows[1, 0], rows[1, 0]))

        pairs = torch.arange(2 * 6 * 6 * 2).reshape(2, 6, 6, 2)
        gathered_pairs = PairBiasAttention._gather_pair_key_columns(pairs, indices)
        self.assertTrue(torch.equal(gathered_pairs[0], pairs[0, :, [1, 3, 4]]))
        self.assertTrue(torch.equal(gathered_pairs[1, :, 0], pairs[1, :, 0]))

    def test_invalid_layouts_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            PairBiasAttention._compact_key_layout(torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "no key/value"):
            PairBiasAttention._compact_key_layout(
                torch.tensor([[True, False], [False, False]])
            )


class CompactAttentionParityTest(unittest.TestCase):
    def test_attention_outputs_inputs_and_parameter_gradients_match(self):
        generator = torch.Generator().manual_seed(9041)
        query_mask = torch.tensor([[True, True, True, True, True, False], [True] * 6])
        key_mask = torch.tensor(
            [
                [True, False, True, False, False, False],
                [False, True, False, True, True, False],
            ]
        )
        dense = MultiHeadBiasedAttentionADALN_MM(
            dim_token=8,
            dim_pair=4,
            nheads=2,
            dim_cond=5,
            use_qkln=True,
        ).double()
        compact = copy.deepcopy(dense)
        dense_inputs = [
            torch.randn(2, 6, 8, dtype=torch.float64, generator=generator),
            torch.randn(2, 6, 6, 4, dtype=torch.float64, generator=generator),
            torch.randn(2, 6, 5, dtype=torch.float64, generator=generator),
        ]
        dense_inputs = [value.requires_grad_() for value in dense_inputs]
        compact_inputs = [
            value.detach().clone().requires_grad_() for value in dense_inputs
        ]

        dense_output = dense(
            *dense_inputs,
            query_mask,
            key_mask=key_mask,
            compact_kv=False,
        )
        compact_output = compact(
            *compact_inputs,
            query_mask,
            key_mask=key_mask,
            compact_kv=True,
        )
        _assert_close(self, dense_output, compact_output, "attention output")

        dense_output.square().sum().backward()
        compact_output.square().sum().backward()
        for index, (dense_input, compact_input) in enumerate(
            zip(dense_inputs, compact_inputs, strict=True)
        ):
            _assert_close(
                self,
                dense_input.grad,
                compact_input.grad,
                f"attention input gradient {index}",
            )
        for (dense_name, dense_parameter), (compact_name, compact_parameter) in zip(
            dense.named_parameters(), compact.named_parameters(), strict=True
        ):
            self.assertEqual(dense_name, compact_name)
            _assert_close(
                self,
                dense_parameter.grad,
                compact_parameter.grad,
                f"attention parameter gradient {dense_name}",
            )

    def test_compact_key_mask_reaches_every_transformer_block(self):
        batch = synthetic_encoder_batch(batch_size=2, length=8)
        real_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
        key_mask = real_mask.clone()
        key_mask[:, 1::2] = False
        masked_batch = mask_batch_for_canonical_residue_view(batch, key_mask)
        model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                masked_kv_mode="visible_only",
                residue_compact_visible_kv=True,
            )
        )
        model.encoder.nlayers = len(model.encoder.transformer_layers)
        seen = []

        def capture(_module, _args, kwargs):
            seen.append((kwargs["key_mask"].clone(), kwargs["compact_kv"]))

        with ExitStack() as stack:
            for layer in model.encoder.transformer_layers:
                stack.enter_context(
                    layer.mhba.register_forward_pre_hook(capture, with_kwargs=True)
                )
            model.encoder(
                masked_batch,
                query_mask=real_mask,
                key_mask=key_mask,
                compact_kv=True,
            )
        self.assertEqual(len(seen), len(model.encoder.transformer_layers))
        for seen_mask, seen_compact in seen:
            self.assertTrue(torch.equal(seen_mask, key_mask))
            self.assertIs(seen_compact, True)


class CompactCanonicalModelParityTest(unittest.TestCase):
    def setUp(self):
        self.batch = synthetic_encoder_batch(batch_size=3, length=9)
        torch.manual_seed(7723)
        self.view_masks = build_canonical_view_masks(self.batch)

    def test_dense_and_compact_outputs_losses_gradients_and_rng_match(self):
        dense, compact = _matched_models()
        dense_run = _run_model(dense, self.batch, self.view_masks, backward=True)
        compact_run = _run_model(compact, self.batch, self.view_masks, backward=True)
        dense_output, dense_gradients, dense_forward_rng, dense_backward_rng = dense_run
        compact_output, compact_gradients, compact_forward_rng, compact_backward_rng = (
            compact_run
        )

        self.assertEqual(dense_output.keys(), compact_output.keys())
        for name in dense_output:
            _assert_close(self, dense_output[name], compact_output[name], name)
        self.assertEqual(dense_gradients.keys(), compact_gradients.keys())
        for name in dense_gradients:
            if dense_gradients[name] is None:
                self.assertIsNone(compact_gradients[name], name)
            else:
                _assert_close(
                    self,
                    dense_gradients[name],
                    compact_gradients[name],
                    f"encoder gradient {name}",
                )
        self.assertTrue(torch.equal(dense_forward_rng, compact_forward_rng))
        self.assertTrue(torch.equal(dense_backward_rng, compact_backward_rng))

    def test_one_adamw_update_and_optimizer_state_match(self):
        dense, compact = _matched_models()
        self.assertEqual(dense.state_dict().keys(), compact.state_dict().keys())
        dense_optimizer = dense.configure_optimizers()["optimizer"]
        compact_optimizer = compact.configure_optimizers()["optimizer"]
        dense_output, _, _, _ = _run_model(
            dense, self.batch, self.view_masks, backward=True
        )
        compact_output, _, _, _ = _run_model(
            compact, self.batch, self.view_masks, backward=True
        )
        _assert_close(self, dense_output["loss"], compact_output["loss"], "loss")
        dense_optimizer.step()
        compact_optimizer.step()

        for (dense_name, dense_parameter), (compact_name, compact_parameter) in zip(
            dense.encoder.named_parameters(),
            compact.encoder.named_parameters(),
            strict=True,
        ):
            self.assertEqual(dense_name, compact_name)
            _assert_close(
                self,
                dense_parameter,
                compact_parameter,
                f"updated encoder parameter {dense_name}",
            )
            dense_state = dense_optimizer.state[dense_parameter]
            compact_state = compact_optimizer.state[compact_parameter]
            self.assertEqual(dense_state.keys(), compact_state.keys())
            for state_name in dense_state:
                left = dense_state[state_name]
                right = compact_state[state_name]
                if isinstance(left, torch.Tensor):
                    _assert_close(
                        self,
                        left,
                        right,
                        f"optimizer state {dense_name}.{state_name}",
                    )
                else:
                    self.assertEqual(left, right)

    def test_checkpointed_compact_matches_noncheckpointed_compact(self):
        _, direct = _matched_models(checkpointed_compact=False)
        _, checkpointed = _matched_models(checkpointed_compact=True)
        checkpointed.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
        direct_run = _run_model(direct, self.batch, self.view_masks, backward=True)
        checkpointed_run = _run_model(
            checkpointed, self.batch, self.view_masks, backward=True
        )
        for name in direct_run[0]:
            _assert_close(
                self,
                direct_run[0][name],
                checkpointed_run[0][name],
                f"checkpointed {name}",
            )
        for name in direct_run[1]:
            if direct_run[1][name] is None:
                self.assertIsNone(checkpointed_run[1][name])
            else:
                _assert_close(
                    self,
                    direct_run[1][name],
                    checkpointed_run[1][name],
                    f"checkpointed gradient {name}",
                )
        self.assertTrue(torch.equal(direct_run[2], checkpointed_run[2]))
        self.assertTrue(torch.equal(direct_run[3], checkpointed_run[3]))

    def test_configuration_boundaries_and_strict_checkpoint_compatibility(self):
        dense, compact = _matched_models()
        incompatible = compact.load_state_dict(dense.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

        for alignment, mode in (
            ("protein", "visible_only"),
            ("residue", "iterative"),
        ):
            with self.subTest(alignment=alignment, mode=mode):
                with self.assertRaisesRegex(
                    ValueError, "residue_compact_visible_kv requires"
                ):
                    CanonicalLeJEPAModel(
                        tiny_canonical_config(
                            alignment_level=alignment,
                            masked_kv_mode=mode,
                            residue_compact_visible_kv=True,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
