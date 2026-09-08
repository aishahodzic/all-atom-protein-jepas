"""Tests for independently shifted canonical residue crops."""

import unittest
from unittest.mock import patch

import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CanonicalLeJEPAModel,
    build_canonical_crop_masks,
    build_canonical_view_masks,
    canonical_invariance_loss,
    canonical_residue_invariance_loss,
    gather_mask_for_packed_canonical_view,
    pack_batch_for_canonical_view,
    protein_centered_residue_sigreg_input,
    scatter_packed_canonical_residues,
)
from proteinfoundation.jepa.tests.canonical_test_utils import (
    synthetic_encoder_batch,
    tiny_canonical_config,
)


class CanonicalCroppingTest(unittest.TestCase):
    def test_fixed_length_crops_share_anchor_and_hide_padding(self):
        torch.manual_seed(7)
        batch = synthetic_encoder_batch(batch_size=3, length=12)
        crops, anchors = build_canonical_crop_masks(batch, crop_length=8)
        self.assertEqual(tuple(crops.shape), (3, 5, 12))
        self.assertTrue(torch.equal(crops.sum(dim=2), torch.full((3, 5), 8)))
        self.assertTrue(crops[torch.arange(3), :, anchors].all())
        real = batch["mask_dict"]["coords"][..., 0, 0].bool()
        self.assertFalse((crops & ~real[:, None]).any())

    def test_masking_occurs_inside_each_crop(self):
        torch.manual_seed(9)
        batch = synthetic_encoder_batch(batch_size=2, length=12)
        crops, _ = build_canonical_crop_masks(batch, crop_length=8)
        visible = build_canonical_view_masks(batch, crop_masks=crops)
        self.assertFalse((visible & ~crops).any())
        self.assertTrue(torch.equal(visible[:, 0].sum(1), torch.full((2,), 7)))
        self.assertTrue(torch.equal(visible[:, 1:].sum(2), torch.full((2, 4), 2)))

    def test_asymmetric_global_and_local_crop_lengths_and_mask_rates(self):
        torch.manual_seed(11)
        batch = synthetic_encoder_batch(batch_size=2, length=12)
        crops, anchors = build_canonical_crop_masks(
            batch, global_crop_length=8, local_crop_length=6
        )
        self.assertTrue(torch.equal(crops[:, 0].sum(1), torch.full((2,), 8)))
        self.assertTrue(torch.equal(crops[:, 1:].sum(2), torch.full((2, 4), 6)))
        self.assertTrue(crops[torch.arange(2), :, anchors].all())
        visible = build_canonical_view_masks(
            batch,
            global_mask_pct=0.20,
            local_mask_pct=0.50,
            crop_masks=crops,
        )
        self.assertTrue(torch.equal(visible[:, 0].sum(1), torch.full((2,), 7)))
        self.assertTrue(torch.equal(visible[:, 1:].sum(2), torch.full((2, 4), 3)))

    def test_fractional_global_crop_contains_fixed_locals(self):
        torch.manual_seed(17)
        batch = synthetic_encoder_batch(batch_size=2, length=12)
        crops, anchors = build_canonical_crop_masks(
            batch,
            global_crop_fraction_min=0.80,
            global_crop_fraction_max=1.00,
            global_crop_min_length=8,
            local_crop_length=4,
            locals_within_global=True,
        )
        global_counts = crops[:, 0].sum(dim=1)
        self.assertTrue(((global_counts >= 8) & (global_counts <= 12)).all())
        self.assertTrue(torch.equal(crops[:, 1:].sum(2), torch.full((2, 4), 4)))
        self.assertTrue(crops[torch.arange(2), :, anchors].all())
        self.assertFalse((crops[:, 1:] & ~crops[:, :1]).any())

    def test_independent_local_starts_stay_inside_global_without_shared_anchor(self):
        torch.manual_seed(0)
        batch = synthetic_encoder_batch(batch_size=1, length=12)
        crops, anchors = build_canonical_crop_masks(
            batch,
            global_crop_length=8,
            local_crop_length=4,
            locals_within_global=True,
            shared_crop_anchor=False,
        )
        self.assertTrue(torch.equal(crops[:, 0].sum(1), torch.full((1,), 8)))
        self.assertTrue(torch.equal(crops[:, 1:].sum(2), torch.full((1, 4), 4)))
        self.assertFalse((crops[:, 1:] & ~crops[:, :1]).any())
        # Independent local starts are not constrained to contain the global
        # crop's sampling anchor.
        local_contains_anchor = crops[torch.arange(1), 1:, anchors]
        self.assertTrue((~local_contains_anchor).any())

    def test_fixed_crop_keeps_real_residues_with_missing_backbone_atoms(self):
        batch = synthetic_encoder_batch(batch_size=1, length=8)
        missing_rows = torch.tensor([1, 3, 5])
        batch["coord_mask"][0, missing_rows, 0] = False
        batch["mask_dict"]["coords"][0, missing_rows, 0] = False
        batch["mask_dict"]["coords_nm"][0, missing_rows, 0] = False
        batch["mask_dict"]["coord_mask"][0, missing_rows, 0] = False
        batch["coords"][0, missing_rows, 0] = 0
        batch["coords_nm"][0, missing_rows, 0] = 0

        crops, _ = build_canonical_crop_masks(batch, crop_length=8)
        self.assertTrue(torch.equal(crops.sum(dim=2), torch.full((1, 5), 8)))
        packed = pack_batch_for_canonical_view(
            batch,
            crops[:, 0],
            row_valid_mask=batch["mask_dict"]["residue_type"].bool(),
        )
        self.assertTrue(packed.valid_mask.all())
        self.assertFalse(packed.batch["coord_mask"][0, missing_rows, 0].any())

        model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                cropping=True,
                crop_length=8,
                packed_views=True,
                residue_compact_visible_kv=True,
            )
        )
        torch.manual_seed(13)
        output = model.compute_loss(batch)
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_crop_invariance_uses_only_pairwise_overlap(self):
        values = torch.zeros(1, 5, 6, 1)
        values[:, 1:, 0] = 1000.0  # outside every local/global overlap
        masks = torch.zeros(1, 5, 6, dtype=torch.bool)
        masks[:, 0, 1:5] = True
        masks[:, 1:, 2:6] = True
        self.assertEqual(float(canonical_residue_invariance_loss(values, masks)), 0.0)

    def test_sigreg_sample_is_shared_and_centered_per_crop(self):
        values = torch.arange(30.0).reshape(1, 5, 6, 1)
        masks = torch.zeros(1, 5, 6, dtype=torch.bool)
        masks[:, 0, 1:5] = True
        masks[:, 1:, 2:6] = True
        selected, indices = protein_centered_residue_sigreg_input(
            values, masks, sampled_indices=torch.tensor([3])
        )
        self.assertEqual(indices.tolist(), [3])
        self.assertEqual(tuple(selected.shape), (5, 1, 1))
        self.assertTrue(torch.isfinite(selected).all())

    def test_flag_supports_protein_and_residue_alignment_and_default_remains_off(self):
        self.assertFalse(CanonicalLeJEPAModel(tiny_canonical_config()).cropping)
        protein_model = CanonicalLeJEPAModel(
            tiny_canonical_config(cropping=True, packed_views=True)
        )
        self.assertEqual(protein_model.alignment_level, "protein")
        self.assertTrue(protein_model.cropping)
        model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                cropping=True,
            )
        )
        self.assertTrue(model.cropping)

    def test_protein_level_dense_and_packed_crops_match(self):
        batch = synthetic_encoder_batch(batch_size=2, length=12)
        dense = CanonicalLeJEPAModel(
            tiny_canonical_config(
                cropping=True,
                packed_views=False,
                global_crop_length=8,
                local_crop_length=6,
                local_mask_pct=0.50,
            )
        )
        packed = CanonicalLeJEPAModel(
            tiny_canonical_config(
                cropping=True,
                packed_views=True,
                global_crop_length=8,
                local_crop_length=6,
                local_mask_pct=0.50,
            )
        )
        packed.load_state_dict(dense.state_dict())

        torch.manual_seed(771)
        dense_embeddings, dense_masks = dense.encode_views(batch)
        torch.manual_seed(771)
        packed_embeddings, packed_masks = packed.encode_views(batch)

        self.assertTrue(torch.equal(dense_masks, packed_masks))
        self.assertTrue(
            torch.allclose(
                dense_embeddings, packed_embeddings,
                atol=2e-5, rtol=2e-5,
            )
        )
        dense_inv = canonical_invariance_loss(dense_embeddings)
        packed_inv = canonical_invariance_loss(packed_embeddings)
        torch.manual_seed(772)
        dense_sigreg = dense.sigreg(dense_embeddings.permute(1, 0, 2))
        torch.manual_seed(772)
        packed_sigreg = packed.sigreg(packed_embeddings.permute(1, 0, 2))
        self.assertTrue(torch.allclose(dense_inv, packed_inv, atol=2e-6, rtol=2e-5))
        self.assertTrue(
            torch.allclose(dense_sigreg, packed_sigreg, atol=2e-5, rtol=2e-5)
        )

    def test_joint_view_sigreg_uses_every_view_as_one_distribution(self):
        model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                cropping=True,
                packed_views=True,
                global_crop_length=8,
                local_crop_length=6,
                local_mask_pct=0.50,
                sigreg_joint_views=True,
            )
        )
        observed_shapes = []

        def capture(values):
            observed_shapes.append(tuple(values.shape))
            return values.square().mean()

        with patch.object(model.sigreg, "forward", side_effect=capture):
            output = model.compute_loss(
                synthetic_encoder_batch(batch_size=3, length=12)
            )
        self.assertEqual(observed_shapes, [(1, 15, 8)])
        self.assertTrue(torch.isfinite(output["loss"]))

        residue_model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                sigreg_joint_views=True,
                cropping=True,
                packed_views=True,
                global_crop_length=8,
                local_crop_length=6,
                local_mask_pct=0.50,
                residue_compact_visible_kv=True,
            )
        )
        residue_shapes = []

        def capture_residue(values):
            residue_shapes.append(tuple(values.shape))
            return values.square().mean()

        with patch.object(
            residue_model.sigreg, "forward", side_effect=capture_residue
        ):
            residue_output = residue_model.compute_loss(
                synthetic_encoder_batch(batch_size=3, length=12)
            )
        self.assertEqual(residue_shapes, [(1, 15, 8)])
        self.assertTrue(torch.isfinite(residue_output["loss"]))

    def test_cropped_forward_backward_is_finite(self):
        torch.manual_seed(19)
        model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                cropping=True,
                crop_length=8,
            )
        )
        output = model.compute_loss(synthetic_encoder_batch(batch_size=3, length=12))
        self.assertEqual(tuple(output["alignment_masks"].shape), (3, 5, 12))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))

    def test_crop_pack_gathers_masks_and_scatters_outputs_by_original_row(self):
        batch = synthetic_encoder_batch(batch_size=2, length=12)
        crop = torch.zeros(2, 12, dtype=torch.bool)
        crop[0, [1, 3, 7, 8]] = True
        crop[1, [0, 4, 5, 9]] = True
        packed = pack_batch_for_canonical_view(batch, crop)
        visible = crop.clone()
        visible[0, 3] = False
        visible[1, 5] = False
        packed_visible = gather_mask_for_packed_canonical_view(visible, packed)
        self.assertEqual(packed.original_indices.tolist(), [[1, 3, 7, 8], [0, 4, 5, 9]])
        self.assertEqual(
            packed.batch["residue_pdb_idx"].tolist(),
            [[1, 3, 7, 8], [0, 4, 5, 9]],
        )
        self.assertEqual(
            packed_visible.tolist(),
            [[True, False, True, True], [True, True, False, True]],
        )
        packed_values = torch.arange(8.0).reshape(2, 4, 1)
        dense = scatter_packed_canonical_residues(
            packed_values, packed, dense_length=12
        )
        self.assertEqual(dense[0, [1, 3, 7, 8], 0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(dense[1, [0, 4, 5, 9], 0].tolist(), [4.0, 5.0, 6.0, 7.0])

    def test_packed_crop_encoder_never_sees_full_protein_length(self):
        model = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                cropping=True,
                crop_length=8,
                packed_views=True,
                residue_compact_visible_kv=True,
            )
        )
        lengths = []
        original_forward = model.encoder.forward

        def recording_forward(batch, **kwargs):
            lengths.append(batch["coords"].shape[1])
            return original_forward(batch, **kwargs)

        with patch.object(model.encoder, "forward", side_effect=recording_forward):
            model.encode_residue_views(
                synthetic_encoder_batch(batch_size=2, length=12)
            )
        self.assertEqual(lengths, [8] * 5)

    def test_dense_and_physically_packed_crops_have_equal_outputs_gradients_and_step(self):
        dense = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                cropping=True,
                crop_length=8,
                global_crop_length=8,
                local_crop_length=6,
                local_mask_pct=0.50,
                packed_views=False,
                residue_compact_visible_kv=True,
            )
        )
        packed = CanonicalLeJEPAModel(
            tiny_canonical_config(
                alignment_level="residue",
                sigreg_input="protein_centered_residue",
                cropping=True,
                crop_length=8,
                global_crop_length=8,
                local_crop_length=6,
                local_mask_pct=0.50,
                packed_views=True,
                residue_compact_visible_kv=True,
            )
        )
        packed.load_state_dict(dense.state_dict())
        batch = synthetic_encoder_batch(batch_size=2, length=12)

        torch.manual_seed(29)
        dense_residue, dense_protein, dense_visible, dense_crops = (
            dense.encode_residue_views(batch)
        )
        torch.manual_seed(29)
        packed_residue, packed_protein, packed_visible, packed_crops = (
            packed.encode_residue_views(batch)
        )
        self.assertTrue(torch.equal(dense_visible, packed_visible))
        self.assertTrue(torch.equal(dense_crops, packed_crops))
        self.assertTrue(
            torch.allclose(dense_protein, packed_protein, atol=2e-6, rtol=2e-5)
        )
        crop_values = dense_crops[..., None].expand_as(dense_residue)
        self.assertTrue(
            torch.allclose(
                dense_residue[crop_values],
                packed_residue[crop_values],
                atol=2e-6,
                rtol=2e-5,
            )
        )

        dense_loss = canonical_residue_invariance_loss(dense_residue, dense_crops)
        packed_loss = canonical_residue_invariance_loss(packed_residue, packed_crops)
        self.assertTrue(torch.allclose(dense_loss, packed_loss, atol=2e-7, rtol=2e-5))
        sampled_indices = dense_crops.all(dim=1).long().argmax(dim=1)
        dense_sigreg_input, _ = protein_centered_residue_sigreg_input(
            dense_residue, dense_crops, sampled_indices=sampled_indices
        )
        packed_sigreg_input, _ = protein_centered_residue_sigreg_input(
            packed_residue, packed_crops, sampled_indices=sampled_indices
        )
        torch.manual_seed(31)
        dense_sigreg = dense.sigreg(dense_sigreg_input)
        torch.manual_seed(31)
        packed_sigreg = packed.sigreg(packed_sigreg_input)
        self.assertTrue(
            torch.allclose(dense_sigreg, packed_sigreg, atol=2e-6, rtol=2e-5)
        )
        dense_total = dense_loss + dense.sigreg_weight * dense_sigreg
        packed_total = packed_loss + packed.sigreg_weight * packed_sigreg
        dense_total.backward()
        packed_total.backward()
        dense_gradients = []
        packed_gradients = []
        for dense_parameter, packed_parameter in zip(
            dense.encoder.parameters(), packed.encoder.parameters()
        ):
            if dense_parameter.grad is None or packed_parameter.grad is None:
                self.assertIs(dense_parameter.grad, packed_parameter.grad)
            else:
                self.assertTrue(
                    torch.allclose(
                        dense_parameter.grad,
                        packed_parameter.grad,
                        atol=2e-6,
                        rtol=2e-4,
                    )
                )
                dense_gradients.append(dense_parameter.grad.flatten())
                packed_gradients.append(packed_parameter.grad.flatten())
        dense_gradient = torch.cat(dense_gradients)
        packed_gradient = torch.cat(packed_gradients)
        relative_l2 = torch.linalg.vector_norm(dense_gradient - packed_gradient) / (
            torch.linalg.vector_norm(dense_gradient).clamp_min(1e-12)
        )
        cosine = torch.nn.functional.cosine_similarity(
            dense_gradient, packed_gradient, dim=0
        )
        self.assertLessEqual(float(relative_l2), 1e-5)
        self.assertGreaterEqual(float(cosine), 0.999999)
        dense_optimizer = dense.configure_optimizers()["optimizer"]
        packed_optimizer = packed.configure_optimizers()["optimizer"]
        dense_optimizer.step()
        packed_optimizer.step()
        for dense_parameter, packed_parameter in zip(
            dense.encoder.parameters(), packed.encoder.parameters()
        ):
            self.assertTrue(
                torch.allclose(
                    dense_parameter,
                    packed_parameter,
                    atol=5e-5,
                    rtol=2e-4,
                ),
                msg=(dense_parameter - packed_parameter).abs().max().item(),
            )

    def test_independent_k4_residue_sigreg_matches_dense_and_packed(self):
        common = dict(
            alignment_level="residue",
            sigreg_input="protein_centered_residue",
            cropping=True,
            global_crop_length=8,
            local_crop_length=6,
            local_mask_pct=0.50,
            locals_within_global=True,
            shared_crop_anchor=False,
            residue_samples_per_protein=4,
            independent_residue_samples=True,
            residue_compact_visible_kv=True,
        )
        dense = CanonicalLeJEPAModel(
            tiny_canonical_config(packed_views=False, **common)
        )
        packed = CanonicalLeJEPAModel(
            tiny_canonical_config(packed_views=True, **common)
        )
        packed.load_state_dict(dense.state_dict())
        batch = synthetic_encoder_batch(batch_size=2, length=12)

        torch.manual_seed(101)
        dense_residue, _, dense_visible, dense_crops = dense.encode_residue_views(batch)
        torch.manual_seed(101)
        packed_residue, _, packed_visible, packed_crops = (
            packed.encode_residue_views(batch)
        )
        self.assertTrue(torch.equal(dense_visible, packed_visible))
        self.assertTrue(torch.equal(dense_crops, packed_crops))

        row_numbers = torch.arange(12).expand(2, 5, -1)
        indices = row_numbers.masked_fill(~dense_crops, 12).topk(
            4, dim=2, largest=False
        ).indices
        dense_input, _ = protein_centered_residue_sigreg_input(
            dense_residue,
            dense_crops,
            sampled_indices=indices,
            samples_per_protein=4,
            independent_views=True,
        )
        packed_input, _ = protein_centered_residue_sigreg_input(
            packed_residue,
            packed_crops,
            sampled_indices=indices,
            samples_per_protein=4,
            independent_views=True,
        )
        torch.testing.assert_close(dense_input, packed_input, rtol=2e-5, atol=2e-6)
        torch.manual_seed(103)
        dense_loss = dense.sigreg(dense_input)
        torch.manual_seed(103)
        packed_loss = packed.sigreg(packed_input)
        torch.testing.assert_close(dense_loss, packed_loss, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
