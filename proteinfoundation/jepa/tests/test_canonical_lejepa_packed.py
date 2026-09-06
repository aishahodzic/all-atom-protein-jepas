"""Dense-versus-packed semantic parity tests for canonical protein LeJEPA.

The dense mask-in-place path is the reference implementation.  These tests use
identical proteins and fixed view masks to verify that canonical-only physical
packing removes hidden rows without changing visible residue features,
representations, losses, or encoder gradients.
"""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CANONICAL_PACKED_REQUIRED_FIELDS,
    CanonicalLeJEPAModel,
    canonical_invariance_loss,
    mask_batch_for_canonical_view,
    mean_pool_visible_residues,
    pack_batch_for_canonical_view,
)
from proteinfoundation.jepa.tests.canonical_test_utils import tiny_canonical_config
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


# On the deterministic fixture below, a one-layer fp32 encoder has observed
# dense/packed discrepancies below 7e-7 for projected features and below 4e-7
# for means.  These limits allow ordinary fp32 reduction-order noise while
# remaining far below differences that could indicate changed semantics.
FP32_ATOL = 1e-6
FP32_RTOL = 1e-5
GRAD_ATOL = 1e-6
GRAD_RTOL = 1e-4
# BF16 unit roundoff is 2^-7 ~= 0.0078125.  These limits are declared before
# the CUDA parity run and allow a few bf16 rounding units for changed GEMM and
# softmax shapes while remaining tight enough to expose structured divergence.
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2
BF16_GRAD_ATOL = 5e-3
BF16_GRAD_RTOL = 5e-2


def _composite_edge_case_batch() -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Return four proteins covering every audited packing edge case.

    Rows respectively exercise a single chain, multiple chains with a PDB-index
    restart, large sequence gaps/chain breaks/missing atoms, and short-sequence
    padding.  ``coords_nm`` is centered over each complete valid protein before
    any view is selected.
    """

    batch_size, length = 4, 10
    generator = torch.Generator().manual_seed(8128)
    coords_angstrom = torch.randn(
        batch_size, length, 37, 3, generator=generator, dtype=torch.float32
    )
    valid_lengths = torch.tensor([10, 8, 7, 5], dtype=torch.long)
    valid = torch.arange(length)[None, :] < valid_lengths[:, None]

    # Keep the Angstrom representation uncentered, matching the canonical data
    # transform ordering.  Center only coords_nm using the full valid protein.
    coords_nm = coords_angstrom / 10.0
    for row, n_valid in enumerate(valid_lengths.tolist()):
        centre = coords_nm[row, :n_valid, 1].mean(dim=0, keepdim=True)
        coords_nm[row, :n_valid] -= centre[:, None, :]
    coords_angstrom = coords_angstrom.masked_fill(
        ~valid[..., None, None], 0.0
    )
    coords_nm = coords_nm.masked_fill(~valid[..., None, None], 0.0)

    coord_mask = valid[..., None].expand(-1, -1, 37).clone()
    # Missing Atom37 entries on both retained and hidden residues.  In
    # particular, cover N, C, CB, and a side-chain atom without invalidating the
    # residue-level padding mask.
    coord_mask[0, 4, 3] = False
    coord_mask[1, 5, 0] = False
    coord_mask[2, 2, 2] = False
    coord_mask[2, 4, 3] = False
    coord_mask[2, 5, 10] = False
    coords_angstrom = coords_angstrom * coord_mask[..., None]
    coords_nm = coords_nm * coord_mask[..., None]

    residue_type = (
        torch.arange(batch_size * length, dtype=torch.long).reshape(batch_size, length)
        % 20
    ).masked_fill(~valid, -1)
    residue_pdb_idx = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [101, 102, 103, 104, 1, 2, 3, 4, -1, -1],
            [10, 11, 50, 51, 300, 301, 302, -1, -1, -1],
            [5, 6, 7, 8, 9, -1, -1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    chains = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, -1, -1],
            [2, 2, 2, 2, 2, 2, 2, -1, -1, -1],
            [0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    chain_breaks = torch.zeros(batch_size, length, dtype=torch.bool)
    chain_breaks[0, 3] = True
    chain_breaks[1, 3] = True
    chain_breaks[2, 2] = True

    tensors = {
        "coords": coords_angstrom,
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "residue_pdb_idx": residue_pdb_idx,
        "chains": chains,
        "chain_breaks_per_residue": chain_breaks,
    }
    mask_dict = {
        key: valid.reshape(batch_size, length, *([1] * (value.ndim - 2))).expand_as(
            value
        ).clone()
        for key, value in tensors.items()
    }
    return {**tensors, "mask_dict": mask_dict}


def _fixed_view_masks(batch: dict) -> torch.Tensor:
    """Return explicit global/local masks with a different packed K per view."""

    selections = (
        # Global view: max K = 8.
        ((0, 1, 2, 3, 5, 6, 8, 9), (0, 1, 2, 4, 5, 7), (0, 1, 2, 4, 6), (0, 1, 3, 4)),
        # Local 1: max K = 3.
        ((0, 4, 9), (1, 4), (0, 2), (3,)),
        # Local 2: max K = 3.
        ((2, 7), (0, 4, 7), (4,), (1, 4)),
        # Local 3: max K = 4.
        ((1, 3, 6, 8), (5,), (1, 5), (2,)),
        # Local 4: max K = 3; row 2 packs far-apart [10, 50, 300].
        ((0, 5), (3, 6), (0, 2, 4), (0,)),
    )
    masks = torch.zeros(4, 5, 10, dtype=torch.bool)
    for view_index, per_protein in enumerate(selections):
        for row, indices in enumerate(per_protein):
            masks[row, view_index, list(indices)] = True
    valid = batch["mask_dict"]["coords"][..., 0, 0]
    if (masks & ~valid[:, None]).any():
        raise AssertionError("fixed parity fixture selected padding")
    return masks


def _visible_seq(tensor: torch.Tensor, visible: torch.Tensor, row: int) -> torch.Tensor:
    return tensor[row].index_select(0, torch.where(visible[row])[0])


def _visible_pairs(tensor: torch.Tensor, visible: torch.Tensor, row: int) -> torch.Tensor:
    indices = torch.where(visible[row])[0]
    return tensor[row].index_select(0, indices).index_select(1, indices)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = _batch_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = copy.deepcopy(value)
    return moved


class CanonicalPackedParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = _composite_edge_case_batch()
        self.view_masks = _fixed_view_masks(self.batch)
        self.cfg_dense = tiny_canonical_config(num_proj=16, packed_views=False)
        self.cfg_packed = tiny_canonical_config(num_proj=16, packed_views=True)

    def assertClose(self, actual: torch.Tensor, expected: torch.Tensor, message=""):
        max_error = float((actual - expected).abs().max()) if actual.numel() else 0.0
        self.assertTrue(
            torch.allclose(
                actual, expected, atol=FP32_ATOL, rtol=FP32_RTOL, equal_nan=False
            ),
            message or f"max absolute error={max_error:.9g}",
        )

    def test_each_view_gathers_original_fields_without_renumbering_or_recentering(self):
        expected_lengths = (8, 3, 3, 4, 3)
        for view_index, expected_length in enumerate(expected_lengths):
            with self.subTest(view=view_index):
                visible = self.view_masks[:, view_index]
                packed = pack_batch_for_canonical_view(self.batch, visible)
                self.assertEqual(packed.valid_mask.shape[1], expected_length)
                self.assertEqual(
                    expected_length, int(visible.sum(dim=1).max().item())
                )
                self.assertTrue(torch.equal(packed.batch["mask"], packed.valid_mask))

                for row in range(visible.shape[0]):
                    original_indices = torch.where(visible[row])[0]
                    count = int(original_indices.numel())
                    self.assertTrue(
                        torch.equal(
                            packed.original_indices[row, :count], original_indices
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            packed.original_indices[row, count:],
                            torch.full_like(packed.original_indices[row, count:], -1),
                        )
                    )
                    for key in CANONICAL_PACKED_REQUIRED_FIELDS:
                        expected = self.batch[key][row].index_select(
                            0, original_indices
                        )
                        self.assertTrue(
                            torch.equal(packed.batch[key][row, :count], expected), key
                        )
                        expected_mask = self.batch["mask_dict"][key][row].index_select(
                            0, original_indices
                        )
                        self.assertTrue(
                            torch.equal(
                                packed.batch["mask_dict"][key][row, :count],
                                expected_mask,
                            ),
                            f"mask_dict[{key}]",
                        )
                    self.assertFalse(packed.valid_mask[row, count:].any())
                    for mask in packed.batch["mask_dict"].values():
                        self.assertFalse(mask[row, count:].any())

        # The full protein was centered, but this deliberately asymmetric
        # visible subset is not. Packing must preserve that non-zero mean.
        valid_row = self.batch["mask_dict"]["coords"][0, :, 0, 0]
        full_ca_mean = self.batch["coords_nm"][0, valid_row, 1].mean(dim=0)
        self.assertClose(full_ca_mean, torch.zeros_like(full_ca_mean))
        visible = self.view_masks[:, 1]
        packed = pack_batch_for_canonical_view(self.batch, visible)
        count = int(visible[0].sum())
        selected_ca_mean = packed.batch["coords_nm"][0, :count, 1].mean(dim=0)
        self.assertGreater(float(selected_ca_mean.abs().max()), 1e-3)

        # Gathered metadata, not packed adjacency, determines these semantics.
        far_gap = pack_batch_for_canonical_view(self.batch, self.view_masks[:, 4])
        self.assertTrue(
            torch.equal(
                far_gap.batch["residue_pdb_idx"][2, :3],
                torch.tensor([10, 50, 300]),
            )
        )
        self.assertTrue(far_gap.batch["chain_breaks_per_residue"][2, 1])

    def test_raw_and_projected_active_features_match_on_visible_residues(self):
        torch.manual_seed(101)
        encoder = EncoderTransformer(**self.cfg_dense.nn_ae).eval()
        for view_index in range(5):
            visible = self.view_masks[:, view_index]
            dense_batch = mask_batch_for_canonical_view(self.batch, visible)
            packed = pack_batch_for_canonical_view(self.batch, visible)

            for mode, factory in (
                ("seq", encoder.init_repr_factory),
                ("pair", encoder.pair_rep_factory),
            ):
                for creator_index, creator in enumerate(factory.feat_creators):
                    dense_feature = creator(dense_batch)
                    packed_feature = creator(packed.batch)
                    for row in range(visible.shape[0]):
                        count = int(visible[row].sum())
                        expected = (
                            _visible_seq(dense_feature, visible, row)
                            if mode == "seq"
                            else _visible_pairs(dense_feature, visible, row)
                        )
                        actual = (
                            packed_feature[row, :count]
                            if mode == "seq"
                            else packed_feature[row, :count, :count]
                        )
                        self.assertClose(
                            actual,
                            expected,
                            f"view={view_index} {mode} creator={creator_index}",
                        )

                dense_projected = factory(dense_batch)
                packed_projected = factory(packed.batch)
                for row in range(visible.shape[0]):
                    count = int(visible[row].sum())
                    expected = (
                        _visible_seq(dense_projected, visible, row)
                        if mode == "seq"
                        else _visible_pairs(dense_projected, visible, row)
                    )
                    actual = (
                        packed_projected[row, :count]
                        if mode == "seq"
                        else packed_projected[row, :count, :count]
                    )
                    self.assertClose(
                        actual, expected, f"view={view_index} projected {mode}"
                    )

    def test_encoder_means_and_pooled_embeddings_match_for_every_view(self):
        torch.manual_seed(202)
        encoder = EncoderTransformer(**self.cfg_dense.nn_ae).eval()
        with torch.no_grad():
            for view_index in range(5):
                visible = self.view_masks[:, view_index]
                dense_batch = mask_batch_for_canonical_view(self.batch, visible)
                packed = pack_batch_for_canonical_view(self.batch, visible)

                # The production encoder samples an unused z after mean. Resetting
                # here makes that incidental draw controlled without changing it.
                torch.manual_seed(303)
                dense_mean = encoder(dense_batch)["mean"]
                torch.manual_seed(303)
                packed_mean = encoder(packed.batch)["mean"]
                for row in range(visible.shape[0]):
                    count = int(visible[row].sum())
                    self.assertClose(
                        packed_mean[row, :count],
                        _visible_seq(dense_mean, visible, row),
                        f"view={view_index} row={row}",
                    )

                dense_pool = mean_pool_visible_residues(dense_mean, visible)
                packed_pool = mean_pool_visible_residues(
                    packed_mean, packed.valid_mask
                )
                self.assertClose(packed_pool, dense_pool, f"view={view_index} pool")

    def test_fixed_five_view_losses_and_encoder_gradients_match(self):
        torch.manual_seed(404)
        dense_model = CanonicalLeJEPAModel(self.cfg_dense).train()
        packed_model = CanonicalLeJEPAModel(self.cfg_packed).train()
        packed_model.load_state_dict(dense_model.state_dict())

        patch_target = (
            "proteinfoundation.jepa.canonical_lejepa_model."
            "build_canonical_view_masks"
        )
        torch.manual_seed(505)
        with mock.patch(patch_target, return_value=self.view_masks.clone()):
            dense_embeddings, dense_masks = dense_model.encode_views(self.batch)
        torch.manual_seed(505)
        with mock.patch(patch_target, return_value=self.view_masks.clone()):
            packed_embeddings, packed_masks = packed_model.encode_views(self.batch)

        self.assertTrue(torch.equal(dense_masks, self.view_masks))
        self.assertTrue(torch.equal(packed_masks, self.view_masks))
        self.assertClose(packed_embeddings, dense_embeddings, "[B,5,8] embeddings")

        dense_inv = canonical_invariance_loss(dense_embeddings)
        packed_inv = canonical_invariance_loss(packed_embeddings)
        self.assertClose(packed_inv, dense_inv, "invariance loss")

        # Reset immediately before the SIGReg projection draw. Dense and packed
        # encoders consume different amounts of RNG for their unused z samples;
        # production RNG semantics remain untouched.
        torch.manual_seed(606)
        dense_sigreg = dense_model.sigreg(dense_embeddings.permute(1, 0, 2))
        torch.manual_seed(606)
        packed_sigreg = packed_model.sigreg(packed_embeddings.permute(1, 0, 2))
        self.assertClose(packed_sigreg, dense_sigreg, "SIGReg loss")

        dense_total = dense_inv + dense_model.sigreg_weight * dense_sigreg
        packed_total = packed_inv + packed_model.sigreg_weight * packed_sigreg
        self.assertClose(packed_total, dense_total, "total loss")
        dense_total.backward()
        packed_total.backward()

        dense_parameters = dict(dense_model.encoder.named_parameters())
        packed_parameters = dict(packed_model.encoder.named_parameters())
        self.assertEqual(dense_parameters.keys(), packed_parameters.keys())
        dense_gradients = []
        packed_gradients = []
        for name in dense_parameters:
            dense_grad = dense_parameters[name].grad
            packed_grad = packed_parameters[name].grad
            self.assertIsNotNone(dense_grad, name)
            self.assertIsNotNone(packed_grad, name)
            self.assertTrue(torch.isfinite(dense_grad).all(), name)
            self.assertTrue(torch.isfinite(packed_grad).all(), name)
            max_error = float((dense_grad - packed_grad).abs().max())
            self.assertTrue(
                torch.allclose(
                    dense_grad,
                    packed_grad,
                    atol=GRAD_ATOL,
                    rtol=GRAD_RTOL,
                ),
                f"gradient {name} max absolute error={max_error:.9g}",
            )
            dense_gradients.append(dense_grad.flatten())
            packed_gradients.append(packed_grad.flatten())

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

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA device with bf16 support required",
    )
    def test_cuda_bf16_embeddings_losses_and_gradients_match(self):
        device = torch.device("cuda", torch.cuda.current_device())
        batch = _batch_to_device(self.batch, device)
        fixed_masks = self.view_masks.to(device)

        torch.manual_seed(707)
        dense_model = CanonicalLeJEPAModel(self.cfg_dense).to(device).train()
        packed_model = CanonicalLeJEPAModel(self.cfg_packed).to(device).train()
        packed_model.load_state_dict(dense_model.state_dict())
        patch_target = (
            "proteinfoundation.jepa.canonical_lejepa_model."
            "build_canonical_view_masks"
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            torch.manual_seed(808)
            with mock.patch(patch_target, return_value=fixed_masks.clone()):
                dense_embeddings, _ = dense_model.encode_views(batch)
            torch.manual_seed(808)
            with mock.patch(patch_target, return_value=fixed_masks.clone()):
                packed_embeddings, _ = packed_model.encode_views(batch)

            dense_inv = canonical_invariance_loss(dense_embeddings)
            packed_inv = canonical_invariance_loss(packed_embeddings)
            torch.manual_seed(909)
            dense_sigreg = dense_model.sigreg(dense_embeddings.permute(1, 0, 2))
            torch.manual_seed(909)
            packed_sigreg = packed_model.sigreg(
                packed_embeddings.permute(1, 0, 2)
            )
            dense_total = dense_inv + dense_model.sigreg_weight * dense_sigreg
            packed_total = packed_inv + packed_model.sigreg_weight * packed_sigreg

        comparisons = {
            "embeddings": (packed_embeddings, dense_embeddings),
            "invariance": (packed_inv, dense_inv),
            "SIGReg": (packed_sigreg, dense_sigreg),
            "total": (packed_total, dense_total),
        }
        observed = {}
        for name, (actual, expected) in comparisons.items():
            max_error = float((actual.float() - expected.float()).abs().max())
            observed[name] = max_error
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    expected.float(),
                    atol=BF16_ATOL,
                    rtol=BF16_RTOL,
                ),
                f"bf16 {name} max absolute error={max_error:.9g}",
            )

        dense_total.backward()
        packed_total.backward()
        dense_parameters = dict(dense_model.encoder.named_parameters())
        packed_parameters = dict(packed_model.encoder.named_parameters())
        dense_gradients = []
        packed_gradients = []
        max_gradient_error = 0.0
        for name in dense_parameters:
            dense_grad = dense_parameters[name].grad
            packed_grad = packed_parameters[name].grad
            self.assertIsNotNone(dense_grad, name)
            self.assertIsNotNone(packed_grad, name)
            max_gradient_error = max(
                max_gradient_error,
                float((dense_grad.float() - packed_grad.float()).abs().max()),
            )
            self.assertTrue(
                torch.allclose(
                    dense_grad.float(),
                    packed_grad.float(),
                    atol=BF16_GRAD_ATOL,
                    rtol=BF16_GRAD_RTOL,
                ),
                f"bf16 gradient mismatch: {name}",
            )
            dense_gradients.append(dense_grad.float().flatten())
            packed_gradients.append(packed_grad.float().flatten())

        dense_gradient = torch.cat(dense_gradients)
        packed_gradient = torch.cat(packed_gradients)
        relative_l2 = torch.linalg.vector_norm(dense_gradient - packed_gradient) / (
            torch.linalg.vector_norm(dense_gradient).clamp_min(1e-12)
        )
        cosine = torch.nn.functional.cosine_similarity(
            dense_gradient, packed_gradient, dim=0
        )
        self.assertLessEqual(float(relative_l2), 2e-2)
        self.assertGreaterEqual(float(cosine), 0.999)
        print(
            "bf16_packed_parity "
            f"max_embedding_error={observed['embeddings']:.9g} "
            f"invariance_error={observed['invariance']:.9g} "
            f"sigreg_error={observed['SIGReg']:.9g} "
            f"total_error={observed['total']:.9g} "
            f"max_gradient_error={max_gradient_error:.9g} "
            f"gradient_relative_l2={float(relative_l2):.9g} "
            f"gradient_cosine={float(cosine):.9g}",
            flush=True,
        )

    def test_dense_reference_never_calls_packing_helper(self):
        model = CanonicalLeJEPAModel(self.cfg_dense).eval()
        patch_masks = (
            "proteinfoundation.jepa.canonical_lejepa_model."
            "build_canonical_view_masks"
        )
        patch_pack = (
            "proteinfoundation.jepa.canonical_lejepa_model."
            "pack_batch_for_canonical_view"
        )
        with mock.patch(patch_masks, return_value=self.view_masks.clone()), mock.patch(
            patch_pack, side_effect=AssertionError("dense path invoked packing")
        ):
            embeddings, masks = model.encode_views(self.batch)
        self.assertEqual(tuple(embeddings.shape), (4, 5, 8))
        self.assertTrue(torch.equal(masks, self.view_masks))


class CanonicalPackedFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = _composite_edge_case_batch()
        self.visible = _fixed_view_masks(self.batch)[:, 1]

    def test_exact_feature_allowlist_rejects_position_neighbor_and_future_features(self):
        mutations = {
            "absolute/rebased position": lambda cfg: cfg.nn_ae.encoder.feats_seq.append(
                "res_seq_pdb_idx"
            ),
            "backbone torsion": lambda cfg: cfg.nn_ae.encoder.feats_seq.append(
                "x1_bb_angles"
            ),
            "backbone bond": lambda cfg: cfg.nn_ae.encoder.feats_seq.append(
                "x1_bond_angles"
            ),
            "sidechain angle": lambda cfg: cfg.nn_ae.encoder.feats_seq.append(
                "x1_sidechain_angles"
            ),
            "future feature": lambda cfg: cfg.nn_ae.encoder.feats_seq.append(
                "cropped_flag_seq"
            ),
            "condition feature": lambda cfg: setattr(
                cfg.nn_ae.encoder, "feats_cond_seq", ["chain_break_per_res"]
            ),
            "pair reorder": lambda cfg: setattr(
                cfg.nn_ae.encoder,
                "feats_pair_repr",
                list(reversed(cfg.nn_ae.encoder.feats_pair_repr)),
            ),
            "pair update": lambda cfg: setattr(
                cfg.nn_ae.encoder, "update_pair_repr", True
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                cfg = tiny_canonical_config(packed_views=True)
                mutate(cfg)
                with self.assertRaisesRegex(ValueError, "exact audited"):
                    CanonicalLeJEPAModel(cfg)

    def test_missing_required_runtime_fields_and_masks_fail_closed(self):
        for key in CANONICAL_PACKED_REQUIRED_FIELDS:
            with self.subTest(field=key):
                batch = copy.deepcopy(self.batch)
                del batch[key]
                with self.assertRaisesRegex(ValueError, key):
                    pack_batch_for_canonical_view(batch, self.visible)
            with self.subTest(mask=key):
                batch = copy.deepcopy(self.batch)
                del batch["mask_dict"][key]
                with self.assertRaisesRegex(ValueError, key):
                    pack_batch_for_canonical_view(batch, self.visible)

        batch = copy.deepcopy(self.batch)
        batch["residue_pdb_idx"] = None
        with self.assertRaisesRegex(ValueError, "residue_pdb_idx"):
            pack_batch_for_canonical_view(batch, self.visible)

    def test_wrong_index_shape_and_dtype_cannot_trigger_contiguous_fallback(self):
        batch = copy.deepcopy(self.batch)
        batch["residue_pdb_idx"] = batch["residue_pdb_idx"][:, :-1]
        with self.assertRaisesRegex(ValueError, "residue_pdb_idx"):
            pack_batch_for_canonical_view(batch, self.visible)

        batch = copy.deepcopy(self.batch)
        batch["residue_pdb_idx"] = batch["residue_pdb_idx"].float()
        batch["mask_dict"]["residue_pdb_idx"] = batch["mask_dict"][
            "residue_pdb_idx"
        ].clone()
        with self.assertRaisesRegex(TypeError, "residue_pdb_idx"):
            pack_batch_for_canonical_view(batch, self.visible)

    def test_padding_selection_zero_visible_and_bad_masks_fail(self):
        selected_padding = self.visible.clone()
        selected_padding[3, 9] = True
        with self.assertRaisesRegex(ValueError, "padded residues"):
            pack_batch_for_canonical_view(self.batch, selected_padding)

        zero_visible = self.visible.clone()
        zero_visible[2] = False
        with self.assertRaisesRegex(ValueError, "zero visible"):
            pack_batch_for_canonical_view(self.batch, zero_visible)

        with self.assertRaisesRegex(ValueError, "boolean"):
            pack_batch_for_canonical_view(self.batch, self.visible.float())

        batch = copy.deepcopy(self.batch)
        batch["mask_dict"] = []
        with self.assertRaisesRegex(ValueError, "mapping"):
            pack_batch_for_canonical_view(batch, self.visible)


if __name__ == "__main__":
    unittest.main()
