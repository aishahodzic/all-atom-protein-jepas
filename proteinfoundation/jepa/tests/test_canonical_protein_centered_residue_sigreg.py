"""Focused tests for opt-in protein-centred residue SIGReg input."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

import proteinfoundation.jepa.canonical_lejepa_model as canonical_model_module
from proteinfoundation.jepa.canonical_lejepa_model import (
    CANONICAL_NUM_VIEWS,
    CanonicalLeJEPAModel,
    protein_centered_residue_sigreg_input,
)
from proteinfoundation.jepa.tests.canonical_test_utils import (
    synthetic_encoder_batch,
    tiny_canonical_config,
)


MASKED_KV_MODES = ("iterative", "visible_only")


class _ScriptedResidueEncoder(nn.Module):
    """Return one prescribed dense residue tensor for each canonical view."""

    def __init__(self, residue_views: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("residue_views", residue_views.clone())
        self.next_view = 0

    def forward(self, batch, *, query_mask=None, key_mask=None):
        del batch, query_mask, key_mask
        if self.next_view >= CANONICAL_NUM_VIEWS:
            raise RuntimeError("scripted encoder received more than five views")
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


def _expected_centered_sample(
    residue_embeddings: torch.Tensor,
    real_mask: torch.Tensor,
    sampled_indices: torch.Tensor,
) -> torch.Tensor:
    per_protein = []
    for batch_index, sampled_index in enumerate(sampled_indices.tolist()):
        real_values = residue_embeddings[batch_index, :, real_mask[batch_index]]
        protein_means = real_values.mean(dim=1)
        per_protein.append(
            residue_embeddings[batch_index, :, sampled_index] - protein_means
        )
    return torch.stack(per_protein, dim=1)


class ProteinCenteredResidueSIGRegTest(unittest.TestCase):
    def test_hand_math_excludes_padding_and_gathers_same_row_across_views(self):
        values = torch.arange(2 * 5 * 5 * 2, dtype=torch.float64).reshape(2, 5, 5, 2)
        real_mask = torch.tensor(
            [
                [True, False, True, True, False],
                [False, True, True, False, True],
            ]
        )
        values = values.clone()
        values[~real_mask[:, None, :, None].expand_as(values)] = 1_000_000.0
        sampled_indices = torch.tensor([2, 4])

        actual, actual_indices = protein_centered_residue_sigreg_input(
            values,
            real_mask,
            sampled_indices=sampled_indices,
        )
        expected = _expected_centered_sample(values, real_mask, sampled_indices)

        self.assertEqual(tuple(actual.shape), (5, 2, 2))
        self.assertTrue(torch.equal(actual_indices, sampled_indices))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_random_sampling_is_deterministic_and_selects_only_real_rows(self):
        values = torch.randn(4, 5, 7, 8, generator=torch.Generator().manual_seed(73))
        real_mask = torch.tensor(
            [
                [True, False, True, False, True, False, False],
                [False, True, False, True, False, True, False],
                [True, True, False, False, False, False, False],
                [False, False, True, True, True, True, True],
            ]
        )

        torch.manual_seed(191)
        first_values, first_indices = protein_centered_residue_sigreg_input(
            values, real_mask
        )
        torch.manual_seed(191)
        second_values, second_indices = protein_centered_residue_sigreg_input(
            values, real_mask
        )
        self.assertTrue(torch.equal(first_indices, second_indices))
        self.assertTrue(torch.equal(first_values, second_values))
        self.assertTrue(
            real_mask.gather(1, first_indices.unsqueeze(1)).all().item()
        )

        for seed in range(64):
            torch.manual_seed(seed)
            _, sampled = protein_centered_residue_sigreg_input(values, real_mask)
            self.assertTrue(real_mask.gather(1, sampled.unsqueeze(1)).all().item())

    def test_per_protein_per_view_additive_shifts_cancel(self):
        generator = torch.Generator().manual_seed(293)
        values = torch.randn(3, 5, 6, 8, dtype=torch.float64, generator=generator)
        real_mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, False, True, True, False],
                [False, True, True, True, False, False],
            ]
        )
        sampled_indices = torch.tensor([4, 3, 2])
        shifts = torch.randn(3, 5, 1, 8, dtype=torch.float64, generator=generator)

        baseline, _ = protein_centered_residue_sigreg_input(
            values, real_mask, sampled_indices=sampled_indices
        )
        shifted, _ = protein_centered_residue_sigreg_input(
            values + shifts, real_mask, sampled_indices=sampled_indices
        )
        torch.testing.assert_close(shifted, baseline, rtol=1e-12, atol=1e-12)

    def test_gradients_reach_sampled_and_all_real_rows_but_not_padding(self):
        values = torch.zeros(1, 5, 4, 2, dtype=torch.float64, requires_grad=True)
        real_mask = torch.tensor([[True, True, True, False]])
        sigreg_input, _ = protein_centered_residue_sigreg_input(
            values,
            real_mask,
            sampled_indices=torch.tensor([1]),
        )
        sigreg_input.sum().backward()

        expected = torch.zeros_like(values)
        expected[:, :, 0] = -1.0 / 3.0
        expected[:, :, 1] = 2.0 / 3.0
        expected[:, :, 2] = -1.0 / 3.0
        torch.testing.assert_close(values.grad, expected, rtol=1e-12, atol=1e-12)

    def test_helper_validation_and_config_fail_fast(self):
        values = torch.zeros(2, 5, 4, 8)
        real_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            protein_centered_residue_sigreg_input(values[:, :4], real_mask)
        with self.assertRaisesRegex(ValueError, "boolean"):
            protein_centered_residue_sigreg_input(values, real_mask.float())
        with self.assertRaisesRegex(ValueError, "no real residues"):
            protein_centered_residue_sigreg_input(
                values, torch.zeros_like(real_mask)
            )
        with self.assertRaisesRegex(ValueError, "integer tensor"):
            protein_centered_residue_sigreg_input(
                values, real_mask, sampled_indices=torch.tensor([0.0, 1.0])
            )
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            protein_centered_residue_sigreg_input(
                values, real_mask, sampled_indices=torch.tensor([0, 4])
            )
        with self.assertRaisesRegex(ValueError, "only real/non-padding"):
            protein_centered_residue_sigreg_input(
                values, real_mask, sampled_indices=torch.tensor([2, 1])
            )

        omitted = CanonicalLeJEPAModel(tiny_canonical_config(sigreg_input=None))
        self.assertEqual(omitted.sigreg_input, "protein_mean")
        with self.assertRaisesRegex(ValueError, "sigreg.input"):
            CanonicalLeJEPAModel(tiny_canonical_config(sigreg_input="invalid"))
        with self.assertRaisesRegex(ValueError, "requires alignment_level=residue"):
            CanonicalLeJEPAModel(
                tiny_canonical_config(
                    alignment_level="protein",
                    sigreg_input="protein_centered_residue",
                )
            )
        accepted = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
            )
        )
        self.assertEqual(accepted.sigreg_input, "protein_centered_residue")

    def test_default_protein_mean_path_is_exact_and_never_calls_new_helper(self):
        batch = synthetic_encoder_batch(batch_size=2, length=5)
        real_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
        values = torch.arange(2 * 5 * 5 * 8, dtype=torch.float32).reshape(2, 5, 5, 8)
        values[1, :, ~real_mask[1]] = 1_000_000.0
        view_masks = real_mask[:, None].expand(-1, 5, -1).clone()

        observed = []
        outputs = []
        for sigreg_input in (None, "protein_mean"):
            model = CanonicalLeJEPAModel(
                tiny_canonical_config(
                    alignment_level="residue", sigreg_input=sigreg_input
                )
            )
            model.encoder = _ScriptedResidueEncoder(values)
            recorder = _RecordingSIGReg()
            model.sigreg = recorder
            with (
                mock.patch.object(
                    canonical_model_module,
                    "build_canonical_view_masks",
                    return_value=view_masks,
                ),
                mock.patch.object(
                    canonical_model_module,
                    "protein_centered_residue_sigreg_input",
                    side_effect=AssertionError("default path called opt-in helper"),
                ),
            ):
                outputs.append(model.compute_loss(batch))
            observed.append(recorder.seen)

        expected = torch.stack(
            [
                torch.stack(
                    [values[b, view, real_mask[b]].mean(dim=0) for view in range(5)]
                )
                for b in range(2)
            ]
        ).permute(1, 0, 2)
        self.assertTrue(torch.equal(observed[0], expected))
        self.assertTrue(torch.equal(observed[1], expected))
        for key in outputs[0]:
            self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]), key)

    def test_both_attention_modes_feed_exact_centered_samples_and_keep_total_formula(self):
        batch = synthetic_encoder_batch(batch_size=2, length=5)
        real_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
        values = torch.arange(2 * 5 * 5 * 8, dtype=torch.float32).reshape(2, 5, 5, 8)
        values[1, :, ~real_mask[1]] = 1_000_000.0
        sampled_indices = torch.tensor([3, 1])
        view_masks = real_mask[:, None].expand(-1, 5, -1).clone()
        expected = _expected_centered_sample(values, real_mask, sampled_indices)
        production_helper = protein_centered_residue_sigreg_input

        def fixed_sample(residue_embeddings, mask):
            return production_helper(
                residue_embeddings,
                mask,
                sampled_indices=sampled_indices,
            )

        for mode in MASKED_KV_MODES:
            with self.subTest(mode=mode):
                model = CanonicalLeJEPAModel(
                    tiny_canonical_config(
                        alignment_level="residue",
                        masked_kv_mode=mode,
                        sigreg_input="protein_centered_residue",
                    )
                )
                model.encoder = _ScriptedResidueEncoder(values)
                recorder = _RecordingSIGReg()
                model.sigreg = recorder
                with (
                    mock.patch.object(
                        canonical_model_module,
                        "build_canonical_view_masks",
                        return_value=view_masks,
                    ),
                    mock.patch.object(
                        canonical_model_module,
                        "protein_centered_residue_sigreg_input",
                        side_effect=fixed_sample,
                    ),
                ):
                    output = model.compute_loss(batch)

                self.assertEqual(tuple(recorder.seen.shape), (5, 2, 8))
                torch.testing.assert_close(recorder.seen, expected)
                self.assertTrue(
                    torch.equal(
                        output["loss"],
                        output["inv_loss"] + 0.02 * output["sigreg_loss"],
                    )
                )


if __name__ == "__main__":
    unittest.main()
