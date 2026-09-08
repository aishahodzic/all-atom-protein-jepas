"""Deterministic A/B tests for canonical residue-level attention semantics."""

from __future__ import annotations

import copy
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import nn

from proteinfoundation.jepa.canonical_lejepa_model import (
    CANONICAL_NUM_VIEWS,
    CanonicalLeJEPAModel,
    canonical_residue_invariance_loss,
    mask_batch_for_canonical_residue_view,
)
from proteinfoundation.jepa.tests.canonical_test_utils import (
    clone_batch,
    synthetic_encoder_batch,
    tiny_canonical_config,
)
from proteinfoundation.nn.modules.pair_bias_attn import (
    MultiHeadBiasedAttentionADALN_MM,
)
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


MASKED_KV_MODES = ("iterative", "visible_only")


def _toy_batch() -> dict:
    """One protein with four real residues followed by one padding slot."""
    batch = synthetic_encoder_batch(batch_size=1, length=5)
    batch["coords"][:, 4] = 0
    batch["coords_nm"][:, 4] = 0
    batch["coord_mask"][:, 4] = False
    batch["residue_type"][:, 4] = 0
    for value in batch["mask_dict"].values():
        value[:, 4] = False
    return batch


def _real_and_visible() -> tuple[torch.Tensor, torch.Tensor]:
    real = torch.tensor([[True, True, True, True, False]])
    visible = torch.tensor([[True, False, True, False, False]])
    return real, visible


def _fixed_view_masks() -> torch.Tensor:
    """Global/local1 contain all four visibility combinations."""
    global_visible = torch.tensor([[True, True, False, False, False]])
    local_visible = torch.tensor([[True, False, True, False, False]])
    return torch.stack(
        [global_visible, local_visible, local_visible, local_visible, local_visible],
        dim=1,
    )


def _expected_directed_mask(key_mask: torch.Tensor) -> torch.Tensor:
    real, _ = _real_and_visible()
    return real[:, :, None] & key_mask[:, None, :]


class _RecordingResidueEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_masks: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, batch, *, query_mask=None, key_mask=None):
        self.seen_masks.append((query_mask.detach().clone(), key_mask.detach().clone()))
        batch_size, length = query_mask.shape
        return {
            "mean": batch["coords_nm"].new_zeros(batch_size, length, 8),
        }


class _ScriptedResidueEncoder(nn.Module):
    def __init__(self, residue_views: torch.Tensor, *, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.residue_views = nn.Parameter(residue_views.clone())
        else:
            self.register_buffer("residue_views", residue_views.clone())
        self.next_view = 0

    def forward(self, batch, *, query_mask=None, key_mask=None):
        del batch, query_mask, key_mask
        result = self.residue_views[:, self.next_view]
        self.next_view += 1
        return {"mean": result}


class _RecordingSIGReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.seen = values.detach().clone()
        return values.square().mean()


class _ConstantSIGReg(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.new_tensor(self.value)


class _IdentityAdaptive(nn.Module):
    def forward(self, x, cond, mask):
        del cond
        return x * mask[..., None]


def _deterministic_attention_fixture() -> MultiHeadBiasedAttentionADALN_MM:
    """One-head production attention whose output is the eligible-value mean."""
    attention = MultiHeadBiasedAttentionADALN_MM(
        dim_token=1,
        dim_pair=1,
        nheads=1,
        dim_cond=1,
        use_qkln=False,
    ).eval()
    attention.adaln = _IdentityAdaptive()
    attention.scale_output = _IdentityAdaptive()
    attention.mha.node_norm = nn.Identity()
    attention.mha.pair_norm = nn.Identity()
    with torch.no_grad():
        attention.mha.to_qkv.weight.copy_(torch.tensor([[0.0], [0.0], [1.0]]))
        attention.mha.to_qkv.bias.zero_()
        attention.mha.to_g.weight.zero_()
        attention.mha.to_g.bias.fill_(1000.0)
        attention.mha.to_out_node.weight.fill_(1.0)
        attention.mha.to_out_node.bias.zero_()
        attention.mha.to_bias.weight.zero_()
    return attention


def _run_deterministic_attention(
    attention: MultiHeadBiasedAttentionADALN_MM,
    values: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    query_mask = torch.ones(1, 4, dtype=torch.bool)
    pair = torch.zeros(1, 4, 4, 1)
    cond = torch.zeros(1, 4, 1)
    return (
        attention(
            values.reshape(1, 4, 1),
            pair,
            cond,
            query_mask,
            key_mask=key_mask,
        )
        .squeeze(0)
        .squeeze(-1)
    )


class CanonicalResidueAttentionModeTest(unittest.TestCase):
    def test_01_exact_query_and_key_masks(self):
        batch = _toy_batch()
        real, visible = _real_and_visible()
        view_masks = visible[:, None, :].expand(-1, CANONICAL_NUM_VIEWS, -1).clone()

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                model = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue", masked_kv_mode=mode
                    )
                )
                recorder = _RecordingResidueEncoder()
                model.encoder = recorder
                with mock.patch(
                    "proteinfoundation.jepa.canonical_lejepa_model."
                    "build_canonical_view_masks",
                    return_value=view_masks,
                ):
                    model.encode_residue_views(batch)

                expected_key = real if mode == "iterative" else visible
                self.assertEqual(len(recorder.seen_masks), CANONICAL_NUM_VIEWS)
                for query_mask, key_mask in recorder.seen_masks:
                    self.assertTrue(torch.equal(query_mask, real))
                    self.assertTrue(torch.equal(key_mask, expected_key))

    def test_02_exact_directed_attention_matrix(self):
        real, visible = _real_and_visible()
        expected = {
            "iterative": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
            )[None],
            "visible_only": torch.tensor(
                [
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.bool,
            )[None],
        }

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                module = MultiHeadBiasedAttentionADALN_MM(
                    dim_token=4,
                    dim_pair=2,
                    nheads=1,
                    dim_cond=3,
                    use_qkln=False,
                ).eval()
                key_mask = real if mode == "iterative" else visible
                with mock.patch.object(
                    module.mha, "forward", wraps=module.mha.forward
                ) as attention_spy:
                    module(
                        torch.randn(1, 5, 4),
                        torch.zeros(1, 5, 5, 2),
                        torch.zeros(1, 5, 3),
                        real,
                        key_mask=key_mask,
                    )
                actual = attention_spy.call_args.kwargs["mask"]
                self.assertEqual(actual.dtype, torch.bool)
                self.assertTrue(torch.equal(actual, expected[mode]))
                self.assertTrue(torch.equal(actual, _expected_directed_mask(key_mask)))

    def test_03_masked_biological_input_is_scrubbed_in_both_modes(self):
        original = _toy_batch()
        real, visible = _real_and_visible()
        hidden = real & ~visible
        changed = clone_batch(original)
        changed["residue_type"][hidden] = (changed["residue_type"][hidden] + 13) % 20
        changed["coords"][hidden] = torch.randn_like(changed["coords"][hidden]) * 1e4
        changed["coords_nm"][hidden] = (
            torch.randn_like(changed["coords_nm"][hidden]) * 1e3
        )
        changed["coord_mask"][hidden] = ~changed["coord_mask"][hidden]
        changed["mask_dict"]["coord_mask"][hidden] = ~changed["mask_dict"][
            "coord_mask"
        ][hidden]

        masked_original = mask_batch_for_canonical_residue_view(original, visible)
        masked_changed = mask_batch_for_canonical_residue_view(changed, visible)
        for key in ("residue_type", "coords", "coords_nm", "coord_mask"):
            self.assertTrue(torch.equal(masked_original[key], masked_changed[key]), key)
        for key in ("residue_type", "coords", "coords_nm", "coord_mask"):
            self.assertTrue(
                torch.equal(
                    masked_original["mask_dict"][key],
                    masked_changed["mask_dict"][key],
                ),
                f"mask_dict.{key}",
            )

        torch.manual_seed(3003)
        encoder = EncoderTransformer(**tiny_canonical_config().nn_ae).eval()
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                key_mask = real if mode == "iterative" else visible
                torch.manual_seed(8103)
                with torch.no_grad():
                    before = encoder(
                        masked_original, query_mask=real, key_mask=key_mask
                    )["mean"]
                torch.manual_seed(8103)
                with torch.no_grad():
                    after = encoder(masked_changed, query_mask=real, key_mask=key_mask)[
                        "mean"
                    ]
                self.assertTrue(torch.equal(before, after))

    def test_04_masked_queries_use_visible_context_in_both_modes(self):
        original = _toy_batch()
        real, visible = _real_and_visible()
        changed = clone_batch(original)
        changed["residue_type"][0, 2] = (changed["residue_type"][0, 2] + 7) % 20
        changed["coords_nm"][0, 2] += 7.5
        changed["coords"][0, 2] = changed["coords_nm"][0, 2] * 10.0
        masked_original = mask_batch_for_canonical_residue_view(original, visible)
        masked_changed = mask_batch_for_canonical_residue_view(changed, visible)
        torch.manual_seed(4004)
        encoder = EncoderTransformer(**tiny_canonical_config().nn_ae).eval()

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                key_mask = real if mode == "iterative" else visible
                torch.manual_seed(4404)
                with torch.no_grad():
                    before = encoder(
                        masked_original, query_mask=real, key_mask=key_mask
                    )["mean"]
                torch.manual_seed(4404)
                with torch.no_grad():
                    after = encoder(masked_changed, query_mask=real, key_mask=key_mask)[
                        "mean"
                    ]
                delta = (before[0, 1] - after[0, 1]).abs().max()
                self.assertGreater(float(delta), 1e-7)

    def test_05_deterministic_attention_has_exact_mode_outputs(self):
        attention = _deterministic_attention_fixture()
        values = torch.tensor([1.0, 10.0, 3.0, 20.0])
        visible = torch.tensor([[True, False, True, False]])
        expected = {
            "iterative": torch.full((4,), 8.5),
            "visible_only": torch.full((4,), 2.0),
        }
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                key_mask = torch.ones_like(visible) if mode == "iterative" else visible
                actual = _run_deterministic_attention(attention, values, key_mask)
                self.assertTrue(torch.equal(actual, expected[mode]))

    def test_06_masked_value_influences_others_only_in_iterative_mode(self):
        attention = _deterministic_attention_fixture()
        before_values = torch.tensor([1.0, 10.0, 3.0, 20.0])
        after_values = torch.tensor([1.0, 110.0, 3.0, 20.0])
        visible = torch.tensor([[True, False, True, False]])

        before_b = _run_deterministic_attention(attention, before_values, visible)
        after_b = _run_deterministic_attention(attention, after_values, visible)
        self.assertTrue(torch.equal(before_b, after_b))

        all_real = torch.ones_like(visible)
        before_a = _run_deterministic_attention(attention, before_values, all_real)
        after_a = _run_deterministic_attention(attention, after_values, all_real)
        other_queries = torch.tensor([True, False, True, True])
        self.assertTrue(
            torch.equal(
                after_a[other_queries] - before_a[other_queries],
                torch.full((3,), 25.0),
            )
        )

    def test_07_key_mask_semantics_reach_every_transformer_block(self):
        batch = _toy_batch()
        real, visible = _real_and_visible()
        masked = mask_batch_for_canonical_residue_view(batch, visible)
        cfg = tiny_canonical_config()
        cfg.nn_ae.encoder.nlayers = 3
        torch.manual_seed(7007)
        encoder = EncoderTransformer(**cfg.nn_ae).eval()

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                key_mask = real if mode == "iterative" else visible
                seen: list[tuple[torch.Tensor, torch.Tensor]] = []

                def capture(_module, args, kwargs):
                    seen.append(
                        (args[3].detach().clone(), kwargs["key_mask"].detach().clone())
                    )

                with ExitStack() as stack:
                    for layer in encoder.transformer_layers:
                        stack.enter_context(
                            layer.mhba.register_forward_pre_hook(
                                capture, with_kwargs=True
                            )
                        )
                    torch.manual_seed(7707)
                    with torch.no_grad():
                        encoder(masked, query_mask=real, key_mask=key_mask)

                self.assertEqual(len(seen), 3)
                for query_seen, key_seen in seen:
                    self.assertTrue(torch.equal(query_seen, real))
                    self.assertTrue(torch.equal(key_seen, key_mask))

    def test_08_all_visibility_cases_enter_residue_invariance(self):
        embeddings = torch.tensor(
            [
                [
                    [[1.0], [2.0], [3.0], [4.0], [100.0]],
                    [[1.5], [1.0], [2.0], [6.0], [-100.0]],
                    [[1.0], [2.0], [3.0], [4.0], [100.0]],
                    [[1.0], [2.0], [3.0], [4.0], [100.0]],
                    [[1.0], [2.0], [3.0], [4.0], [100.0]],
                ]
            ]
        )
        real, _ = _real_and_visible()
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(
                    float(canonical_residue_invariance_loss(embeddings, real)),
                    0.3125,
                )

    def test_09_sigreg_pooling_uses_all_real_residues_in_both_modes(self):
        batch = _toy_batch()
        scalar_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        residue_views = scalar_values.view(1, 1, 5, 1).expand(1, 5, 5, 8).clone()
        view_masks = _fixed_view_masks()

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                model = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue", masked_kv_mode=mode
                    )
                )
                model.encoder = _ScriptedResidueEncoder(residue_views)
                recorder = _RecordingSIGReg()
                model.sigreg = recorder
                with mock.patch(
                    "proteinfoundation.jepa.canonical_lejepa_model."
                    "build_canonical_view_masks",
                    return_value=view_masks,
                ):
                    model.compute_loss(batch)
                self.assertEqual(tuple(recorder.seen.shape), (5, 1, 8))
                self.assertTrue(torch.equal(recorder.seen, torch.full((5, 1, 8), 2.5)))

    def test_10_exact_total_loss_in_both_modes(self):
        batch = _toy_batch()
        scalars = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 100.0],
                    [1.5, 1.0, 2.0, 6.0, -100.0],
                    [1.0, 2.0, 3.0, 4.0, 100.0],
                    [1.0, 2.0, 3.0, 4.0, 100.0],
                    [1.0, 2.0, 3.0, 4.0, 100.0],
                ]
            ]
        )
        residue_views = scalars[..., None].expand(-1, -1, -1, 8).clone()
        view_masks = _fixed_view_masks()

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                model = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue", masked_kv_mode=mode
                    )
                )
                model.encoder = _ScriptedResidueEncoder(residue_views)
                model.sigreg = _ConstantSIGReg(3.0)
                with mock.patch(
                    "proteinfoundation.jepa.canonical_lejepa_model."
                    "build_canonical_view_masks",
                    return_value=view_masks,
                ):
                    output = model.compute_loss(batch)
                self.assertEqual(float(output["inv_loss"]), 0.3125)
                self.assertEqual(float(output["sigreg_loss"]), 3.0)
                self.assertAlmostEqual(float(output["loss"]), 0.3725, places=7)

    def test_11_gradients_reach_masked_outputs_and_not_padding(self):
        batch = _toy_batch()
        view_masks = _fixed_view_masks()
        real, _ = _real_and_visible()
        residue_views = torch.zeros(1, 5, 5, 8)
        for view_index in range(1, 5):
            residue_views[:, view_index] = float(view_index)

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                model = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue", masked_kv_mode=mode
                    )
                )
                encoder = _ScriptedResidueEncoder(residue_views, learnable=True)
                model.encoder = encoder
                model.sigreg = _ConstantSIGReg(0.0)
                with mock.patch(
                    "proteinfoundation.jepa.canonical_lejepa_model."
                    "build_canonical_view_masks",
                    return_value=view_masks,
                ):
                    model.compute_loss(batch)["loss"].backward()

                gradient = encoder.residue_views.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                for view_index in range(CANONICAL_NUM_VIEWS):
                    masked_real = real & ~view_masks[:, view_index]
                    self.assertGreater(
                        float(gradient[:, view_index][masked_real].abs().sum()),
                        0.0,
                    )
                self.assertEqual(float(gradient[:, :, 4].abs().sum()), 0.0)

    def test_12_protein_alignment_ignores_valid_masked_kv_mode(self):
        reference_cfg = tiny_canonical_config(
            alignment_level="protein", masked_kv_mode=None
        )
        torch.manual_seed(12012)
        reference = CanonicalLeJEPAModel(reference_cfg).eval()
        batch = synthetic_encoder_batch(batch_size=3, length=8)

        def run(model):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1212)
            output = model.compute_loss(batch)
            output["loss"].backward()
            gradients = {
                name: None
                if parameter.grad is None
                else parameter.grad.detach().clone()
                for name, parameter in model.encoder.named_parameters()
            }
            return output, gradients

        reference_output, reference_gradients = run(reference)
        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                candidate = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="protein", masked_kv_mode=mode
                    )
                ).eval()
                candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
                output, gradients = run(candidate)
                self.assertEqual(output.keys(), reference_output.keys())
                for name in output:
                    self.assertTrue(
                        torch.equal(output[name], reference_output[name]), name
                    )
                self.assertEqual(gradients.keys(), reference_gradients.keys())
                for name in gradients:
                    if reference_gradients[name] is None:
                        self.assertIsNone(gradients[name], name)
                    else:
                        self.assertTrue(
                            torch.equal(gradients[name], reference_gradients[name]),
                            name,
                        )

    def test_13_configuration_and_fail_fast_guards(self):
        for mode in MASKED_KV_MODES:
            with self.subTest(valid_residue_mode=mode):
                model = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue", masked_kv_mode=mode
                    )
                )
                self.assertEqual(model.masked_kv_mode, mode)
            with self.subTest(valid_protein_mode=mode):
                CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="protein", masked_kv_mode=mode
                    )
                )
            with self.subTest(packed_residue_mode=mode):
                with self.assertRaisesRegex(
                    ValueError, "permits views.packed_views=true only"
                ):
                    CanonicalLeJEPAModel(
                        tiny_canonical_config(
                            alignment_level="residue",
                            masked_kv_mode=mode,
                            packed_views=True,
                        )
                    )

        for alignment_level in ("protein", "residue"):
            with self.subTest(invalid_alignment=alignment_level):
                with self.assertRaisesRegex(ValueError, "masked_kv_mode"):
                    CanonicalLeJEPAModel(
                        tiny_canonical_config(
                            alignment_level=alignment_level,
                            masked_kv_mode="invalid",
                        )
                    )


if __name__ == "__main__":
    unittest.main()
