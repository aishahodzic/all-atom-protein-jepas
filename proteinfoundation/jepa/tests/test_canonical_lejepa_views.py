"""View construction, pooling, and direct masking regression tests."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CANONICAL_NUM_VIEWS,
    CanonicalLeJEPAModel,
    _sample_canonical_view_rotations,
    build_canonical_view_masks,
    mask_batch_for_canonical_view,
    mask_batch_for_canonical_residue_view,
    mean_pool_visible_residues,
    pack_batch_for_canonical_view,
    rotate_batch_for_canonical_view,
)
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.coors_utils import sample_uniform_rotation
from proteinfoundation.jepa.tests.canonical_test_utils import (
    clone_batch,
    synthetic_encoder_batch,
    tiny_canonical_config,
)


class CanonicalViewTest(unittest.TestCase):
    def test_encoder_default_mask_equals_explicit_symmetric_masks_exactly(self):
        """The optional asymmetric-mask API must not alter historical callers."""
        cfg = tiny_canonical_config()
        encoder = EncoderTransformer(**cfg.nn_ae).eval()
        batch = synthetic_encoder_batch(batch_size=3, length=9)
        historical_mask = batch["mask_dict"]["coords"][..., 0, 0]
        batch["mask"] = historical_mask.clone()

        torch.manual_seed(1907)
        with torch.no_grad():
            default_output = encoder(batch)
        torch.manual_seed(1907)
        with torch.no_grad():
            explicit_output = encoder(
                batch,
                query_mask=historical_mask,
                key_mask=historical_mask,
            )

        self.assertEqual(default_output.keys(), explicit_output.keys())
        for name in default_output:
            self.assertTrue(
                torch.equal(default_output[name], explicit_output[name]),
                f"legacy and explicit symmetric encoder outputs differ for {name}",
            )

    def test_residue_view_scrubs_direct_inputs_and_preserves_metadata(self):
        batch = synthetic_encoder_batch(batch_size=2, length=8)
        real = batch["mask_dict"]["coords"][..., 0, 0]
        batch["residue_pdb_idx"] = torch.arange(101, 109).expand(2, -1).clone()
        batch["chains"][0, 4:] = 1
        batch["chains"][1, 3:] = 2
        batch["chain_breaks_per_residue"][0, 4] = True
        batch["chain_breaks_per_residue"][1, 3] = True
        for key in ("residue_pdb_idx", "chains", "chain_breaks_per_residue"):
            batch["mask_dict"][key] = real.clone()
        original = clone_batch(batch)

        visible = torch.zeros_like(real)
        visible[:, ::2] = real[:, ::2]
        masked = mask_batch_for_canonical_residue_view(batch, visible)
        hidden = real & ~visible

        self.assertTrue((masked["residue_type"][hidden] == -1).all())
        self.assertTrue((masked["coords"][hidden] == 0).all())
        self.assertTrue((masked["coords_nm"][hidden] == 0).all())
        self.assertFalse(masked["coord_mask"][hidden].any())
        for key in ("coords", "coords_nm", "coord_mask", "residue_type"):
            self.assertFalse(masked["mask_dict"][key][hidden].any(), key)

        self.assertTrue(torch.equal(masked["mask"], real))
        for key in ("residue_pdb_idx", "chains", "chain_breaks_per_residue"):
            self.assertTrue(torch.equal(masked[key], original[key]), key)
            self.assertTrue(
                torch.equal(masked["mask_dict"][key], original["mask_dict"][key]),
                f"mask_dict metadata changed for {key}",
            )
        for key in ("coords", "coords_nm", "coord_mask", "residue_type"):
            self.assertTrue(
                torch.equal(masked[key][visible], original[key][visible]), key
            )

        for key in (
            "coords",
            "coords_nm",
            "coord_mask",
            "residue_type",
            "residue_pdb_idx",
            "chains",
            "chain_breaks_per_residue",
        ):
            self.assertTrue(torch.equal(batch[key], original[key]), key)

    def test_masked_query_output_changes_when_visible_context_changes(self):
        """A masked slot must be a contextual query, not a fixed zero token."""
        cfg = tiny_canonical_config()
        encoder = EncoderTransformer(**cfg.nn_ae).eval()
        batch_a = synthetic_encoder_batch(batch_size=1, length=7)
        real = batch_a["mask_dict"]["coords"][..., 0, 0]
        visible = torch.tensor(
            [[True, False, True, True, False, True, True]], dtype=torch.bool
        )
        query_index = 1
        context_index = 3

        batch_b = clone_batch(batch_a)
        batch_b["residue_type"][0, context_index] = (
            batch_b["residue_type"][0, context_index] + 7
        ) % 20
        batch_b["coords_nm"][0, context_index] += 2.75
        batch_b["coords"][0, context_index] = (
            batch_b["coords_nm"][0, context_index] * 10.0
        )

        masked_a = mask_batch_for_canonical_residue_view(batch_a, visible)
        masked_b = mask_batch_for_canonical_residue_view(batch_b, visible)
        with torch.no_grad():
            output_a = encoder(masked_a, query_mask=real, key_mask=visible)["mean"]
            output_b = encoder(masked_b, query_mask=real, key_mask=visible)["mean"]

        self.assertGreater(
            (output_a[0, query_index] - output_b[0, query_index]).abs().max().item(),
            1e-7,
            "masked query ignored a changed visible contextual residue",
        )
        self.assertGreater(output_a[0, query_index].abs().max().item(), 0.0)

    def test_masked_query_cannot_see_its_own_aa_coordinates_or_atom_mask(self):
        cfg = tiny_canonical_config()
        encoder = EncoderTransformer(**cfg.nn_ae).eval()
        original = synthetic_encoder_batch(batch_size=1, length=7)
        real = original["mask_dict"]["coords"][..., 0, 0]
        visible = torch.tensor(
            [[True, False, True, False, True, True, True]], dtype=torch.bool
        )
        hidden = real & ~visible

        changed = clone_batch(original)
        changed["residue_type"][hidden] = (changed["residue_type"][hidden] + 11) % 20
        changed["coords"][hidden] = torch.randn_like(changed["coords"][hidden]) * 1e4
        changed["coords_nm"][hidden] = (
            torch.randn_like(changed["coords_nm"][hidden]) * 1e3
        )
        changed["coord_mask"][hidden] = ~changed["coord_mask"][hidden]

        masked_original = mask_batch_for_canonical_residue_view(original, visible)
        masked_changed = mask_batch_for_canonical_residue_view(changed, visible)
        with torch.no_grad():
            input_original = encoder.init_repr_factory(masked_original)
            input_changed = encoder.init_repr_factory(masked_changed)
            pair_original = encoder.pair_rep_factory(masked_original)
            pair_changed = encoder.pair_rep_factory(masked_changed)
            output_original = encoder(
                masked_original, query_mask=real, key_mask=visible
            )["mean"]
            output_changed = encoder(masked_changed, query_mask=real, key_mask=visible)[
                "mean"
            ]

        self.assertTrue(torch.equal(input_original, input_changed))
        self.assertTrue(torch.equal(pair_original, pair_changed))
        self.assertTrue(torch.equal(output_original, output_changed))

    def test_masked_slot_cannot_influence_other_queries_as_key_value(self):
        """Information injected into a masked query must not become K/V context."""
        cfg = tiny_canonical_config()
        encoder = EncoderTransformer(**cfg.nn_ae).eval()
        batch = synthetic_encoder_batch(batch_size=1, length=7)
        real = batch["mask_dict"]["coords"][..., 0, 0]
        visible = torch.tensor(
            [[True, False, True, False, True, True, True]], dtype=torch.bool
        )
        injected_index = 3

        clean = mask_batch_for_canonical_residue_view(batch, visible)
        injected = clone_batch(clean)
        injected["residue_type"][0, injected_index] = 11
        injected["mask_dict"]["residue_type"][0, injected_index] = True
        injected["coords_nm"][0, injected_index] = 13.0
        injected["coords"][0, injected_index] = 130.0
        injected["coord_mask"][0, injected_index] = True
        injected["mask_dict"]["coord_mask"][0, injected_index] = True
        injected["mask_dict"]["coords_nm"][0, injected_index] = True
        injected["mask_dict"]["coords"][0, injected_index] = True

        with torch.no_grad():
            clean_output = encoder(clean, query_mask=real, key_mask=visible)["mean"]
            injected_output = encoder(injected, query_mask=real, key_mask=visible)[
                "mean"
            ]

        self.assertGreater(
            (clean_output[0, injected_index] - injected_output[0, injected_index])
            .abs()
            .max()
            .item(),
            1e-7,
        )
        other_queries = real.clone()
        other_queries[0, injected_index] = False
        self.assertTrue(
            torch.equal(clean_output[other_queries], injected_output[other_queries]),
            "a masked slot influenced another query despite key_mask=false",
        )

    def test_exact_order_counts_padding_independence_and_seed_replay(self):
        batch = synthetic_encoder_batch(batch_size=3, length=101)
        valid = batch["mask_dict"]["coords"][..., 0, 0]

        torch.manual_seed(7)
        masks_a = build_canonical_view_masks(batch)
        torch.manual_seed(7)
        masks_b = build_canonical_view_masks(batch)

        self.assertTrue(torch.equal(masks_a, masks_b))
        self.assertEqual(tuple(masks_a.shape), (3, 5, 101))
        self.assertFalse((masks_a & ~valid[:, None]).any())
        for row, n_valid_tensor in enumerate(valid.sum(dim=1)):
            n_valid = int(n_valid_tensor)
            self.assertEqual(int(masks_a[row, 0].sum()), n_valid - int(n_valid * 0.20))
            for view in range(1, 5):
                self.assertEqual(
                    int(masks_a[row, view].sum()), n_valid - int(n_valid * 0.80)
                )
        for left in range(1, 5):
            for right in range(left + 1, 5):
                self.assertFalse(torch.equal(masks_a[:, left], masks_a[:, right]))

    def test_mean_pool_visible_exact_arithmetic_and_zero_visible_failure(self):
        values = torch.tensor(
            [
                [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
                [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ]
        )
        visible = torch.tensor([[True, True, False], [False, True, True]])
        expected = torch.tensor([[3.0, 5.0], [8.0, 10.0]])
        self.assertTrue(
            torch.equal(mean_pool_visible_residues(values, visible), expected)
        )
        with self.assertRaisesRegex(ValueError, "zero visible"):
            mean_pool_visible_residues(values, torch.zeros_like(visible))

    def test_hidden_residue_changes_do_not_change_visible_encoder_means(self):
        """Mandatory direct leakage test on the actual EncoderTransformer path."""
        cfg = tiny_canonical_config()
        encoder = EncoderTransformer(**cfg.nn_ae).eval()
        original = synthetic_encoder_batch(batch_size=2, length=10)
        valid = original["mask_dict"]["coords"][..., 0, 0]
        visible = torch.zeros_like(valid)
        visible[:, ::2] = valid[:, ::2]

        changed = clone_batch(original)
        hidden = valid & ~visible
        changed["residue_type"][hidden] = (changed["residue_type"][hidden] + 9) % 20
        changed["coords"][hidden] = torch.randn_like(changed["coords"][hidden]) * 1e4
        changed["coords_nm"][hidden] = (
            torch.randn_like(changed["coords_nm"][hidden]) * 1e3
        )
        changed["coord_mask"][hidden] = ~changed["coord_mask"][hidden]
        changed["mask_dict"]["residue_type"][hidden] = False

        masked_original = mask_batch_for_canonical_view(original, visible)
        masked_changed = mask_batch_for_canonical_view(changed, visible)
        self.assertTrue(
            torch.equal(masked_original["coords"], masked_changed["coords"])
        )
        self.assertTrue(
            torch.equal(masked_original["coords_nm"], masked_changed["coords_nm"])
        )
        self.assertTrue(
            torch.equal(
                masked_original["mask_dict"]["coords"],
                masked_changed["mask_dict"]["coords"],
            )
        )

        with torch.no_grad():
            mean_original = encoder(masked_original)["mean"]
            mean_changed = encoder(masked_changed)["mean"]
        self.assertTrue(
            torch.equal(mean_original[visible], mean_changed[visible]),
            "hidden AA/Atom37 changes altered visible encoder means",
        )

    def test_sampled_view_rotations_are_distinct_proper_rotations(self):
        rotations = sample_uniform_rotation(
            shape=(3, CANONICAL_NUM_VIEWS), dtype=torch.float64
        )
        identity = torch.eye(3, dtype=torch.float64).expand_as(rotations)
        self.assertTrue(
            torch.allclose(
                rotations.transpose(-1, -2) @ rotations,
                identity,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.linalg.det(rotations),
                torch.ones(3, CANONICAL_NUM_VIEWS, dtype=torch.float64),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        flattened = rotations.reshape(-1, 9)
        pairwise_differences = torch.pdist(flattened)
        self.assertTrue((pairwise_differences > 1e-8).all())

    def test_fixed_rotation_preserves_geometry_and_noncoordinate_fields(self):
        batch = synthetic_encoder_batch(batch_size=2, length=5)
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0].clone()
        original = clone_batch(batch)
        rotations = torch.tensor(
            [
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            ]
        )

        rotated = rotate_batch_for_canonical_view(batch, rotations)

        for key in ("coords", "coords_nm"):
            expected = torch.einsum("blac,bcd->blad", batch[key], rotations)
            self.assertTrue(torch.equal(rotated[key], expected))
            self.assertTrue(torch.equal(batch[key], original[key]))
        self.assertTrue(torch.equal(rotated["coords"], rotated["coords_nm"] * 10.0))

        for row in range(2):
            valid_atoms = batch["coord_mask"][row]
            before = batch["coords_nm"][row][valid_atoms]
            after = rotated["coords_nm"][row][valid_atoms]
            self.assertTrue(
                torch.allclose(
                    torch.pdist(before),
                    torch.pdist(after),
                    atol=2e-6,
                    rtol=2e-6,
                )
            )

        for key in (
            "coord_mask",
            "residue_type",
            "residue_pdb_idx",
            "chains",
            "chain_breaks_per_residue",
            "mask",
        ):
            self.assertIs(rotated[key], batch[key])
            self.assertTrue(torch.equal(rotated[key], original[key]))
        self.assertIs(rotated["mask_dict"], batch["mask_dict"])
        for key, value in batch["mask_dict"].items():
            self.assertTrue(torch.equal(rotated["mask_dict"][key], value))

        padding = ~batch["mask_dict"]["coords"]
        self.assertTrue(
            torch.equal(rotated["coords"][padding], batch["coords"][padding])
        )
        self.assertTrue(
            torch.equal(rotated["coords_nm"][padding], batch["coords_nm"][padding])
        )

    def test_fixed_rotation_has_hand_computable_row_vector_result(self):
        batch = synthetic_encoder_batch(batch_size=1, length=1)
        batch["coords"].zero_()
        batch["coords_nm"].zero_()
        batch["coords"][0, 0, 0] = torch.tensor([10.0, 0.0, 0.0])
        batch["coords_nm"][0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        batch["coords"][0, 0, 1] = torch.tensor([0.0, 20.0, 0.0])
        batch["coords_nm"][0, 0, 1] = torch.tensor([0.0, 2.0, 0.0])
        rotation = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

        rotated = rotate_batch_for_canonical_view(batch, rotation)

        self.assertTrue(
            torch.equal(
                rotated["coords_nm"][0, 0, :2],
                torch.tensor([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                rotated["coords"][0, 0, :2],
                torch.tensor([[0.0, 10.0, 0.0], [-20.0, 0.0, 0.0]]),
            )
        )

    def test_disabled_rotations_are_exact_existing_dense_and_packed_paths(self):
        batch = synthetic_encoder_batch(batch_size=2, length=9)
        for packed_views in (False, True):
            with self.subTest(packed_views=packed_views):
                cfg = tiny_canonical_config(packed_views=packed_views)
                cfg.independent_view_rotations = False
                model = CanonicalLeJEPAModel(cfg).eval()

                torch.manual_seed(123)
                with mock.patch(
                    "proteinfoundation.jepa.canonical_lejepa_model."
                    "sample_uniform_rotation",
                    side_effect=AssertionError("disabled path sampled a rotation"),
                ):
                    actual_embeddings, actual_masks = model.encode_views(batch)

                torch.manual_seed(123)
                expected_masks = build_canonical_view_masks(batch)
                expected_views = []
                for view_index in range(CANONICAL_NUM_VIEWS):
                    visible = expected_masks[:, view_index]
                    if packed_views:
                        packed = pack_batch_for_canonical_view(batch, visible)
                        encoded = model.encoder(packed.batch)["mean"]
                        pool_mask = packed.valid_mask
                    else:
                        masked = mask_batch_for_canonical_view(batch, visible)
                        encoded = model.encoder(masked)["mean"]
                        pool_mask = visible
                    expected_views.append(
                        mean_pool_visible_residues(encoded, pool_mask)
                    )
                expected_embeddings = torch.stack(expected_views, dim=1)

                self.assertTrue(torch.equal(actual_masks, expected_masks))
                self.assertTrue(torch.equal(actual_embeddings, expected_embeddings))

    def test_enabled_model_applies_one_independent_rotation_per_protein_view(self):
        class RecordingEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_batches = []

            def forward(self, batch):
                self.seen_batches.append(batch)
                batch_size, length = batch["coords_nm"].shape[:2]
                return {"mean": batch["coords_nm"].new_zeros(batch_size, length, 8)}

        batch = synthetic_encoder_batch(batch_size=2, length=7)
        angles = (
            torch.arange(2 * CANONICAL_NUM_VIEWS, dtype=torch.float32).reshape(
                2, CANONICAL_NUM_VIEWS
            )
            * 0.23
        )
        cosines = angles.cos()
        sines = angles.sin()
        rotation_bank = torch.zeros(2, CANONICAL_NUM_VIEWS, 3, 3)
        rotation_bank[..., 0, 0] = cosines
        rotation_bank[..., 0, 1] = sines
        rotation_bank[..., 1, 0] = -sines
        rotation_bank[..., 1, 1] = cosines
        rotation_bank[..., 2, 2] = 1.0

        cfg = tiny_canonical_config(packed_views=False)
        cfg.independent_view_rotations = True
        model = CanonicalLeJEPAModel(cfg).eval()
        recorder = RecordingEncoder()
        model.encoder = recorder

        torch.manual_seed(71)
        with mock.patch(
            "proteinfoundation.jepa.canonical_lejepa_model.sample_uniform_rotation",
            return_value=rotation_bank.unsqueeze(0),
        ) as sampler:
            _, view_masks = model.encode_views(batch)

        sampler.assert_called_once_with(
            shape=(1, 2, CANONICAL_NUM_VIEWS),
            dtype=batch["coords_nm"].dtype,
            device=batch["coords_nm"].device,
        )
        self.assertEqual(len(recorder.seen_batches), CANONICAL_NUM_VIEWS)
        for view_index, seen in enumerate(recorder.seen_batches):
            rotated = rotate_batch_for_canonical_view(
                batch, rotation_bank[:, view_index]
            )
            expected = mask_batch_for_canonical_view(rotated, view_masks[:, view_index])
            self.assertTrue(torch.equal(seen["coords"], expected["coords"]))
            self.assertTrue(torch.equal(seen["coords_nm"], expected["coords_nm"]))
            for key in (
                "coord_mask",
                "residue_type",
                "residue_pdb_idx",
                "chains",
                "chain_breaks_per_residue",
            ):
                self.assertIs(seen[key], batch[key])

    def test_ddp_ranks_select_disjoint_slices_of_the_rotation_bank(self):
        batch = synthetic_encoder_batch(batch_size=2, length=4)
        full_bank = sample_uniform_rotation(
            shape=(2, 2, CANONICAL_NUM_VIEWS), dtype=batch["coords_nm"].dtype
        )
        with (
            mock.patch(
                "proteinfoundation.jepa.canonical_lejepa_model.dist.is_available",
                return_value=True,
            ),
            mock.patch(
                "proteinfoundation.jepa.canonical_lejepa_model.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "proteinfoundation.jepa.canonical_lejepa_model.dist.get_world_size",
                return_value=2,
            ),
            mock.patch(
                "proteinfoundation.jepa.canonical_lejepa_model.dist.get_rank",
                return_value=1,
            ),
            mock.patch(
                "proteinfoundation.jepa.canonical_lejepa_model.sample_uniform_rotation",
                return_value=full_bank,
            ) as sampler,
        ):
            selected = _sample_canonical_view_rotations(batch)

        sampler.assert_called_once_with(
            shape=(2, 2, CANONICAL_NUM_VIEWS),
            dtype=batch["coords_nm"].dtype,
            device=batch["coords_nm"].device,
        )
        self.assertTrue(torch.equal(selected, full_bank[1]))
        self.assertFalse(torch.equal(selected, full_bank[0]))


if __name__ == "__main__":
    unittest.main()
