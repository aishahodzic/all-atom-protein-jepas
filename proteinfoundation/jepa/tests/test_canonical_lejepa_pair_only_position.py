"""Adversarial tests for pair-only positional information in canonical LeJEPA."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CANONICAL_EVOLVING_PAIR_HIDDEN_DIM,
    CANONICAL_EVOLVING_PAIR_MODE,
    CANONICAL_EVOLVING_PAIR_TRIANGLE_DIM,
    CANONICAL_EVOLVING_PAIR_UPDATE_INTERVAL,
    CANONICAL_PACKED_SAFE_SEQ_FEATURES,
    CANONICAL_PAIR_ONLY_POSITION_SEQ_FEATURES,
    CanonicalLeJEPAModel,
    canonical_invariance_loss,
    canonical_vicreg_rescue_loss,
    resolve_canonical_position_information_config,
)
from proteinfoundation.jepa.tests.test_canonical_packed_adversarial_agent2 import (
    _batch as adversarial_batch,
    _view_masks as adversarial_view_masks,
)
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.jepa.tests.canonical_test_utils import (
    clone_batch,
    synthetic_encoder_batch,
    tiny_canonical_config,
)


def _encoder_mean(model: CanonicalLeJEPAModel, batch: dict) -> torch.Tensor:
    torch.manual_seed(913)
    return model.encoder(batch)["mean"]


def _synthetic_full_encoder_batch(*, batch_size: int, length: int) -> dict:
    batch = synthetic_encoder_batch(batch_size=batch_size, length=length)
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0].clone()
    return batch


def _translate_available_atoms(batch: dict, shift: torch.Tensor) -> dict:
    translated = clone_batch(batch)
    atom_mask = translated["coord_mask"][..., None]
    translated["coords_nm"] = (
        translated["coords_nm"] + shift.reshape(1, 1, 1, 3)
    ) * atom_mask
    return translated


def _permute_residue_rows(batch: dict, permutation: torch.Tensor) -> dict:
    permuted = clone_batch(batch)
    for key, value in permuted.items():
        if key == "mask_dict":
            for mask_key, mask_value in value.items():
                value[mask_key] = mask_value.index_select(1, permutation)
        elif isinstance(value, torch.Tensor) and value.ndim >= 2:
            value = value.index_select(1, permutation)
            permuted[key] = value
    return permuted


class CanonicalPairOnlyPositionConfigTest(unittest.TestCase):
    def test_flag_resolves_to_exact_pair_only_sequence_features(self):
        cfg = tiny_canonical_config()
        cfg.pair_only_position_information = True
        original_features = tuple(cfg.nn_ae.encoder.feats_seq)

        resolved = resolve_canonical_position_information_config(cfg)

        self.assertEqual(original_features, CANONICAL_PACKED_SAFE_SEQ_FEATURES)
        self.assertEqual(
            tuple(resolved.nn_ae.encoder.feats_seq),
            CANONICAL_PAIR_ONLY_POSITION_SEQ_FEATURES,
        )
        self.assertEqual(
            tuple(cfg.nn_ae.encoder.feats_seq),
            CANONICAL_PACKED_SAFE_SEQ_FEATURES,
            "resolution must not mutate the caller's config",
        )
        self.assertEqual(
            tuple(
                resolve_canonical_position_information_config(
                    resolved
                ).nn_ae.encoder.feats_seq
            ),
            CANONICAL_PAIR_ONLY_POSITION_SEQ_FEATURES,
            "resolution must be idempotent",
        )

    def test_flag_fails_closed_for_unreviewed_feature_configurations(self):
        mutations = (
            lambda cfg: cfg.nn_ae.encoder.feats_seq.append("res_seq_pdb_idx"),
            lambda cfg: cfg.nn_ae.encoder.feats_pair_repr.pop(),
            lambda cfg: setattr(
                cfg.nn_ae.encoder, "feats_cond_seq", ["chain_idx_seq"]
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                cfg = tiny_canonical_config(
                    pair_only_position_information=True,
                )
                mutate(cfg)
                with self.assertRaisesRegex(ValueError, "exact audited"):
                    CanonicalLeJEPAModel(cfg)

    def test_exact_lightweight_pair_update_configuration_is_accepted(self):
        cfg = tiny_canonical_config(
            packed_views=True,
            pair_only_position_information=True,
            lightweight_pair_updates=True,
        )
        cfg.nn_ae.encoder.nlayers = 4
        model = CanonicalLeJEPAModel(cfg)

        self.assertTrue(model.encoder.update_pair_repr)
        self.assertFalse(model.encoder.use_tri_mult)
        self.assertIsNotNone(model.encoder.pair_update_layers[0])
        self.assertIsNone(model.encoder.pair_update_layers[1])
        self.assertIsNone(model.encoder.pair_update_layers[2])

    def test_other_pair_update_configurations_still_fail_closed(self):
        for name, mutate in (
            (
                "wrong interval",
                lambda cfg: setattr(
                    cfg.nn_ae.encoder, "update_pair_repr_every_n", 1
                ),
            ),
            (
                "triangle multiplication",
                lambda cfg: setattr(cfg.nn_ae.encoder, "use_tri_mult", True),
            ),
        ):
            with self.subTest(name=name):
                cfg = tiny_canonical_config(
                    packed_views=True,
                    pair_only_position_information=True,
                    lightweight_pair_updates=True,
                )
                mutate(cfg)
                with self.assertRaisesRegex(ValueError, "pair updates"):
                    CanonicalLeJEPAModel(cfg)

    def test_one_evolving_pair_flag_resolves_the_complete_lite_stack(self):
        cfg = tiny_canonical_config(
            packed_views=True,
            pair_only_position_information=True,
            evolving_pair_repr=True,
        )
        cfg.nn_ae.encoder.nlayers = 12

        model = CanonicalLeJEPAModel(cfg)
        encoder_cfg = model.cfg.nn_ae.encoder

        self.assertTrue(model.pair_only_evolving_pair_repr)
        self.assertTrue(encoder_cfg.update_pair_repr)
        self.assertEqual(
            encoder_cfg.update_pair_repr_every_n,
            CANONICAL_EVOLVING_PAIR_UPDATE_INTERVAL,
        )
        self.assertTrue(encoder_cfg.use_tri_mult)
        self.assertEqual(encoder_cfg.pair_update_mode, CANONICAL_EVOLVING_PAIR_MODE)
        self.assertEqual(
            encoder_cfg.pair_update_hidden_dim,
            CANONICAL_EVOLVING_PAIR_HIDDEN_DIM,
        )
        self.assertEqual(
            encoder_cfg.tri_mult_c,
            CANONICAL_EVOLVING_PAIR_TRIANGLE_DIM,
        )
        active_updates = [
            index
            for index, layer in enumerate(model.encoder.pair_update_layers)
            if layer is not None
        ]
        self.assertEqual(active_updates, [0, 3, 6, 9])
        for index in active_updates:
            layer = model.encoder.pair_update_layers[index]
            self.assertEqual(
                layer.sequence_to_pair_mode, CANONICAL_EVOLVING_PAIR_MODE
            )
            self.assertTrue(layer.use_tri_mult)
            self.assertEqual(layer.tri_mult_out.c_hidden, min(8, 32))
            self.assertEqual(layer.tri_mult_in.c_hidden, min(8, 32))

    def test_evolving_pair_resolution_is_idempotent(self):
        cfg = tiny_canonical_config(
            pair_only_position_information=True,
            evolving_pair_repr=True,
        )
        once = resolve_canonical_position_information_config(cfg)
        twice = resolve_canonical_position_information_config(once)
        self.assertEqual(once, twice)

    def test_evolving_pair_flag_requires_pair_only_position_mode(self):
        cfg = tiny_canonical_config(evolving_pair_repr=True)
        with self.assertRaisesRegex(
            ValueError, "requires pair_only_position_information=true"
        ):
            CanonicalLeJEPAModel(cfg)

    def test_evolving_pair_flag_fails_on_conflicting_manual_controls(self):
        mutations = (
            lambda cfg: setattr(cfg.nn_ae.encoder, "pair_context_to_sequence", True),
            lambda cfg: setattr(cfg.nn_ae.encoder, "update_pair_repr", True),
            lambda cfg: setattr(cfg.nn_ae.encoder, "use_tri_mult", True),
            lambda cfg: setattr(
                cfg.nn_ae.encoder, "pair_update_mode", "gated_multiplicative"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                cfg = tiny_canonical_config(
                    pair_only_position_information=True,
                    evolving_pair_repr=True,
                )
                mutate(cfg)
                with self.assertRaisesRegex(ValueError, "mutually exclusive|conflicting"):
                    CanonicalLeJEPAModel(cfg)

    def test_pair_updates_cannot_be_combined_with_uniform_pair_context(self):
        cfg = tiny_canonical_config(
            pair_only_position_information=True,
            pair_context_to_sequence=True,
            lightweight_pair_updates=True,
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            CanonicalLeJEPAModel(cfg)

    def test_absent_flag_and_explicit_false_are_identical(self):
        cfg_absent = tiny_canonical_config()
        del cfg_absent.pair_only_position_information
        cfg_false = tiny_canonical_config()

        torch.manual_seed(101)
        absent = CanonicalLeJEPAModel(cfg_absent).eval()
        torch.manual_seed(101)
        explicit_false = CanonicalLeJEPAModel(cfg_false).eval()

        absent_state = absent.state_dict()
        false_state = explicit_false.state_dict()
        self.assertEqual(absent_state.keys(), false_state.keys())
        for name in absent_state:
            self.assertTrue(torch.equal(absent_state[name], false_state[name]), name)

        batch = _synthetic_full_encoder_batch(batch_size=3, length=9)
        absent_mean = _encoder_mean(absent, batch)
        false_mean = _encoder_mean(explicit_false, batch)
        self.assertTrue(torch.equal(absent_mean, false_mean))

    def test_resolved_evaluation_config_reconstructs_checkpoint_encoder(self):
        cfg = tiny_canonical_config()
        cfg.pair_only_position_information = True
        resolved = resolve_canonical_position_information_config(cfg)
        model = CanonicalLeJEPAModel(resolved)
        evaluation_encoder = EncoderTransformer(**resolved.nn_ae)

        evaluation_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        self.assertEqual(
            evaluation_encoder.init_repr_factory.linear_out.in_features,
            model.encoder.init_repr_factory.linear_out.in_features,
        )

    def test_pair_context_bridge_requires_pair_only_position_mode(self):
        cfg = tiny_canonical_config(pair_context_to_sequence=True)
        with self.assertRaisesRegex(
            ValueError, "only supported with pair_only_position_information=true"
        ):
            CanonicalLeJEPAModel(cfg)

    def test_absent_and_disabled_pair_context_preserve_state_and_output(self):
        cfg_absent = tiny_canonical_config(
            pair_only_position_information=True,
        )
        del cfg_absent.nn_ae.encoder.pair_context_to_sequence
        cfg_disabled = tiny_canonical_config(
            pair_only_position_information=True,
            pair_context_to_sequence=False,
        )

        torch.manual_seed(1701)
        absent = CanonicalLeJEPAModel(cfg_absent).eval()
        torch.manual_seed(1701)
        disabled = CanonicalLeJEPAModel(cfg_disabled).eval()
        self.assertEqual(absent.state_dict().keys(), disabled.state_dict().keys())
        for name, value in absent.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[name]), name)

        batch = _synthetic_full_encoder_batch(batch_size=3, length=9)
        torch.testing.assert_close(
            _encoder_mean(absent, batch),
            _encoder_mean(disabled, batch),
            atol=0,
            rtol=0,
        )


class CanonicalPairContextBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = tiny_canonical_config(
            pair_only_position_information=True,
            pair_context_to_sequence=True,
        )
        self.encoder = CanonicalLeJEPAModel(cfg).encoder

    def test_only_visible_keys_contribute_and_padding_queries_stay_zero(self):
        generator = torch.Generator().manual_seed(1801)
        seqs = torch.randn(2, 5, 16, generator=generator)
        pair_rep = torch.randn(2, 5, 5, 8, generator=generator)
        query_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )
        key_mask = torch.tensor(
            [[True, False, True, False, False], [False, True, False, False, False]]
        )

        original = self.encoder.add_pair_context_to_sequence(
            seqs, pair_rep, query_mask, key_mask
        )
        perturbed = pair_rep.clone()
        perturbed.masked_fill_(~key_mask[:, None, :, None], 1.0e6)
        changed_masked = self.encoder.add_pair_context_to_sequence(
            seqs, perturbed, query_mask, key_mask
        )
        torch.testing.assert_close(changed_masked, original, atol=0, rtol=0)
        self.assertTrue(torch.equal(original[~query_mask], torch.zeros_like(original[~query_mask])))

        perturbed = pair_rep.clone()
        perturbed[0, 0, 0, 0] += 3.0
        changed_visible = self.encoder.add_pair_context_to_sequence(
            seqs, perturbed, query_mask, key_mask
        )
        self.assertGreater(
            float((changed_visible[0, 0] - original[0, 0]).abs().max()),
            1e-5,
        )

    def test_pair_context_receives_gradients(self):
        generator = torch.Generator().manual_seed(1901)
        seqs = torch.randn(2, 4, 16, generator=generator)
        pair_rep = torch.randn(
            2, 4, 4, 8, generator=generator, requires_grad=True
        )
        mask = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        output = self.encoder.add_pair_context_to_sequence(
            seqs, pair_rep, mask, mask
        )
        output.square().sum().backward()

        self.assertIsNotNone(pair_rep.grad)
        self.assertGreater(float(pair_rep.grad.abs().sum()), 0.0)
        projection_grad = self.encoder.pair_context_projection.weight.grad
        self.assertIsNotNone(projection_grad)
        self.assertGreater(float(projection_grad.abs().sum()), 0.0)


class CanonicalEvolvingPairUpdateSemanticsTest(unittest.TestCase):
    def _update(self) -> PairReprUpdate:
        return PairReprUpdate(
            token_dim=12,
            pair_dim=8,
            use_tri_mult=True,
            tri_mult_c=4,
            sequence_to_pair_mode="gated_multiplicative",
            sequence_to_pair_hidden_dim=6,
        )

    def test_zero_initialized_residual_is_identity_on_valid_pairs(self):
        torch.manual_seed(2001)
        update = self._update().eval()
        x = torch.randn(2, 5, 12)
        pair = torch.randn(2, 5, 5, 8)
        mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        result = update(x, pair, mask)
        pair_mask = mask[:, :, None] & mask[:, None, :]

        torch.testing.assert_close(result[pair_mask], pair[pair_mask], atol=0, rtol=0)
        self.assertTrue(torch.equal(result[~pair_mask], torch.zeros_like(result[~pair_mask])))

    def test_multiplicative_update_is_pair_specific_not_a_broadcast_mean(self):
        torch.manual_seed(2101)
        update = self._update().eval()
        with torch.no_grad():
            torch.nn.init.normal_(update.sequence_to_pair_out.weight, std=0.1)
        x = torch.randn(1, 4, 12)
        pair = torch.randn(1, 4, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        original = update(x, pair, mask)
        changed_x = x.clone()
        changed_x[:, 2, 0] += 8.0
        changed = update(changed_x, pair, mask)

        # Changing residue 2 may change pairs (2,j) and (i,2), but cannot
        # broadcast into an unrelated pair as the rejected mean bridge did.
        torch.testing.assert_close(changed[:, 0, 1], original[:, 0, 1], atol=0, rtol=0)
        self.assertGreater(float((changed[:, 2, 1] - original[:, 2, 1]).abs().max()), 1e-6)
        self.assertGreater(float((changed[:, 0, 2] - original[:, 0, 2]).abs().max()), 1e-6)

    def test_triangle_update_couples_pairs_through_a_third_residue(self):
        torch.manual_seed(2201)
        update = self._update().eval()
        with torch.no_grad():
            torch.nn.init.normal_(update.tri_mult_out.linear_z.weight, std=0.1)
        x = torch.randn(1, 4, 12)
        pair = torch.randn(1, 4, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        original = update(x, pair, mask)
        changed_pair = pair.clone()
        changed_pair[:, 0, 2, 0] += 8.0
        changed = update(x, changed_pair, mask)

        self.assertGreater(
            float((changed[:, 0, 1] - original[:, 0, 1]).abs().max()), 1e-6
        )

    def test_padding_cannot_mediate_triangle_updates(self):
        torch.manual_seed(2301)
        update = self._update().eval()
        with torch.no_grad():
            torch.nn.init.normal_(update.tri_mult_out.linear_z.weight, std=0.1)
            torch.nn.init.normal_(update.tri_mult_in.linear_z.weight, std=0.1)
        x = torch.randn(1, 4, 12)
        pair = torch.randn(1, 4, 4, 8)
        mask = torch.tensor([[True, True, True, False]])
        original = update(x, pair, mask)
        changed_x = x.clone()
        changed_pair = pair.clone()
        changed_x[:, 3] = 1.0e5
        changed_pair[:, 3, :] = 1.0e5
        changed_pair[:, :, 3] = -1.0e5
        changed = update(changed_x, changed_pair, mask)

        pair_mask = mask[:, :, None] & mask[:, None, :]
        torch.testing.assert_close(changed[pair_mask], original[pair_mask], atol=0, rtol=0)
        self.assertTrue(torch.equal(changed[~pair_mask], torch.zeros_like(changed[~pair_mask])))

    def test_update_is_residue_permutation_equivariant(self):
        torch.manual_seed(2401)
        update = self._update().eval()
        with torch.no_grad():
            torch.nn.init.normal_(update.sequence_to_pair_out.weight, std=0.05)
            torch.nn.init.normal_(update.tri_mult_out.linear_z.weight, std=0.05)
            torch.nn.init.normal_(update.tri_mult_in.linear_z.weight, std=0.05)
            torch.nn.init.normal_(update.transition_out.linear_2.weight, std=0.05)
        x = torch.randn(2, 5, 12)
        pair = torch.randn(2, 5, 5, 8)
        mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, True, False]]
        )
        permutation = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.argsort(permutation)

        expected = update(x, pair, mask)
        actual = update(
            x.index_select(1, permutation),
            pair.index_select(1, permutation).index_select(2, permutation),
            mask.index_select(1, permutation),
        )
        restored = actual.index_select(1, inverse).index_select(2, inverse)
        torch.testing.assert_close(restored, expected, atol=2e-6, rtol=2e-5)


class CanonicalVicregRescueTest(unittest.TestCase):
    def test_collapsed_proteins_are_penalized(self):
        embeddings = torch.zeros(16, 5, 8)
        loss = canonical_vicreg_rescue_loss(embeddings)
        self.assertAlmostEqual(float(loss), 0.99, places=6)

    def test_covariance_penalizes_correlated_dimensions(self):
        # Eight centered, mutually orthogonal +/-1 columns from a 16x16
        # Hadamard matrix have unit population standard deviation.
        values = torch.tensor([[1.0]])
        for _ in range(4):
            values = torch.cat(
                (
                    torch.cat((values, values), dim=1),
                    torch.cat((values, -values), dim=1),
                ),
                dim=0,
            )
        diverse = values[:, 1:9]
        diverse_embeddings = diverse[:, None, :].expand(-1, 5, -1).clone()
        correlated = diverse[:, :1].expand(-1, 8)
        correlated_embeddings = correlated[:, None, :].expand(-1, 5, -1).clone()

        diverse_loss = canonical_vicreg_rescue_loss(diverse_embeddings)
        correlated_loss = canonical_vicreg_rescue_loss(correlated_embeddings)
        self.assertLess(float(diverse_loss), 1e-10)
        self.assertGreater(float(correlated_loss), 1.0)

    def test_view_disagreement_cannot_fake_protein_diversity(self):
        view_offsets = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        embeddings = view_offsets[None, :, None].expand(16, -1, 8).clone()
        loss = canonical_vicreg_rescue_loss(embeddings)
        self.assertAlmostEqual(float(loss), 0.99, places=6)

    def test_loss_is_protein_and_view_permutation_invariant(self):
        generator = torch.Generator().manual_seed(2501)
        embeddings = torch.randn(16, 5, 8, generator=generator)
        expected = canonical_vicreg_rescue_loss(embeddings)
        protein_order = torch.randperm(16, generator=generator)
        view_order = torch.randperm(5, generator=generator)
        actual = canonical_vicreg_rescue_loss(
            embeddings.index_select(0, protein_order).index_select(1, view_order)
        )
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    def test_loss_has_finite_nonzero_gradients(self):
        embeddings = torch.randn(12, 5, 8, requires_grad=True)
        loss = canonical_vicreg_rescue_loss(embeddings)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)

    def test_flag_is_restricted_to_the_audited_rescue_configuration(self):
        for cfg in (
            tiny_canonical_config(vicreg_rescue=True),
            tiny_canonical_config(
                pair_only_position_information=True,
                vicreg_rescue=True,
            ),
        ):
            with self.subTest(cfg=cfg):
                with self.assertRaisesRegex(ValueError, "pair-only evolving-pair"):
                    CanonicalLeJEPAModel(cfg)

    def test_enabled_loss_composes_without_changing_embeddings_or_sigreg(self):
        base_cfg = tiny_canonical_config(
            pair_only_position_information=True,
            evolving_pair_repr=True,
        )
        rescue_cfg = tiny_canonical_config(
            pair_only_position_information=True,
            evolving_pair_repr=True,
            vicreg_rescue=True,
            vicreg_weight=0.7,
        )
        base_cfg.nn_ae.encoder.nlayers = 4
        rescue_cfg.nn_ae.encoder.nlayers = 4
        torch.manual_seed(2601)
        base = CanonicalLeJEPAModel(base_cfg).train()
        rescue = CanonicalLeJEPAModel(rescue_cfg).train()
        rescue.load_state_dict(base.state_dict(), strict=True)
        batch = adversarial_batch()
        masks = adversarial_view_masks()
        patch_target = (
            "proteinfoundation.jepa.canonical_lejepa_model."
            "build_canonical_view_masks"
        )
        torch.manual_seed(2602)
        with mock.patch(patch_target, return_value=masks.clone()):
            base_output = base.compute_loss(batch)
        torch.manual_seed(2602)
        with mock.patch(patch_target, return_value=masks.clone()):
            rescue_output = rescue.compute_loss(batch)

        torch.testing.assert_close(
            rescue_output["embeddings"], base_output["embeddings"]
        )
        torch.testing.assert_close(rescue_output["inv_loss"], base_output["inv_loss"])
        torch.testing.assert_close(
            rescue_output["sigreg_loss"], base_output["sigreg_loss"]
        )
        expected = (
            base_output["loss"] + 0.7 * rescue_output["vicreg_loss"]
        )
        torch.testing.assert_close(rescue_output["loss"], expected)


class CanonicalPairOnlyPositionSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = _synthetic_full_encoder_batch(batch_size=3, length=9)
        # Exercise multiple chains and an explicit internal chain break.
        self.batch["chains"][0, 4:] = 1
        self.batch["chains"][1, 3:7] = 2
        self.batch["chain_breaks_per_residue"][0, 3] = True
        self.batch["chain_breaks_per_residue"][1, 2] = True
        self.pair_only = CanonicalLeJEPAModel(
            tiny_canonical_config(pair_only_position_information=True)
        ).eval()

    def test_single_stream_cannot_see_index_chain_or_absolute_coordinates(self):
        perturbed = _translate_available_atoms(
            self.batch, torch.tensor([3.25, -1.5, 2.0])
        )
        perturbed["residue_pdb_idx"] += 113
        perturbed["chains"] += 7
        perturbed["chain_breaks_per_residue"] = (
            ~perturbed["chain_breaks_per_residue"]
        )

        original = self.pair_only.encoder.init_repr_factory(self.batch)
        changed = self.pair_only.encoder.init_repr_factory(perturbed)
        torch.testing.assert_close(changed, original, atol=2e-6, rtol=2e-6)

    def test_global_index_offset_and_coordinate_translation_leave_encoder_unchanged(self):
        perturbed = _translate_available_atoms(
            self.batch, torch.tensor([-4.0, 2.5, 7.0])
        )
        valid = perturbed["mask_dict"]["residue_pdb_idx"]
        offsets = torch.tensor([37, 101, 509])[:, None]
        perturbed["residue_pdb_idx"] = torch.where(
            valid, perturbed["residue_pdb_idx"] + offsets, perturbed["residue_pdb_idx"]
        )

        original = _encoder_mean(self.pair_only, self.batch)
        changed = _encoder_mean(self.pair_only, perturbed)
        torch.testing.assert_close(changed, original, atol=5e-6, rtol=2e-5)

    def test_relative_index_information_still_changes_pair_path_and_output(self):
        perturbed = clone_batch(self.batch)
        perturbed["residue_pdb_idx"][:, 1::2] += 11

        original_pair = self.pair_only.encoder.pair_rep_factory(self.batch)
        changed_pair = self.pair_only.encoder.pair_rep_factory(perturbed)
        self.assertGreater(float((changed_pair - original_pair).abs().max()), 1e-4)

        original = _encoder_mean(self.pair_only, self.batch)
        changed = _encoder_mean(self.pair_only, perturbed)
        self.assertGreater(float((changed - original).abs().max()), 1e-6)

    def test_residue_tensor_order_is_only_a_permutation(self):
        permutation = torch.tensor([4, 1, 7, 0, 8, 3, 6, 2, 5])
        inverse = torch.argsort(permutation)
        permuted_batch = _permute_residue_rows(self.batch, permutation)

        original = _encoder_mean(self.pair_only, self.batch)
        permuted = _encoder_mean(self.pair_only, permuted_batch)
        restored = permuted.index_select(1, inverse)
        torch.testing.assert_close(restored, original, atol=2e-6, rtol=2e-5)

    def test_default_encoder_does_see_absolute_coordinates(self):
        default_model = CanonicalLeJEPAModel(tiny_canonical_config()).eval()
        translated = _translate_available_atoms(
            self.batch, torch.tensor([3.25, -1.5, 2.0])
        )
        original = _encoder_mean(default_model, self.batch)
        changed = _encoder_mean(default_model, translated)
        self.assertGreater(float((changed - original).abs().max()), 1e-5)

    def test_exact_pair_only_independent_k4_residue_objective_is_trainable(self):
        """The production ablation composes pair-only tokens with residue K=4."""
        cfg = tiny_canonical_config(
            alignment_level="residue",
            masked_kv_mode="visible_only",
            residue_view_checkpointing=True,
            residue_compact_visible_kv=True,
            sigreg_input="protein_centered_residue",
            residue_samples_per_protein=4,
            independent_residue_samples=True,
            pair_only_position_information=True,
        )
        model = CanonicalLeJEPAModel(cfg).train()
        self.assertEqual(
            tuple(model.cfg.nn_ae.encoder.feats_seq),
            CANONICAL_PAIR_ONLY_POSITION_SEQ_FEATURES,
        )
        self.assertFalse(model.encoder.pair_context_to_sequence)
        self.assertFalse(model.encoder.update_pair_repr)
        self.assertFalse(model.encoder.use_tri_mult)

        batch = synthetic_encoder_batch(batch_size=4, length=12)
        torch.manual_seed(2701)
        output = model.compute_loss(batch)
        self.assertEqual(tuple(output["residue_embeddings"].shape), (4, 5, 12, 8))
        self.assertTrue(torch.isfinite(output["loss"]))

        output["loss"].backward()
        gradients = [
            parameter.grad
            for parameter in model.encoder.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)


class CanonicalPairOnlyPositionOptimizerParityTest(unittest.TestCase):
    def test_one_adamw_step_and_optimizer_state_match_dense_vs_packed(self):
        batch = adversarial_batch()
        fixed_masks = adversarial_view_masks()
        dense_cfg = tiny_canonical_config(
            packed_views=False,
            pair_only_position_information=True,
            pair_context_to_sequence=True,
            num_proj=11,
        )
        packed_cfg = tiny_canonical_config(
            packed_views=True,
            pair_only_position_information=True,
            pair_context_to_sequence=True,
            num_proj=11,
        )
        torch.manual_seed(1201)
        dense = CanonicalLeJEPAModel(dense_cfg).train()
        packed = CanonicalLeJEPAModel(packed_cfg).train()
        packed.load_state_dict(dense.state_dict())
        dense_optimizer = dense.configure_optimizers()["optimizer"]
        packed_optimizer = packed.configure_optimizers()["optimizer"]
        patch_target = (
            "proteinfoundation.jepa.canonical_lejepa_model."
            "build_canonical_view_masks"
        )

        torch.manual_seed(1301)
        with mock.patch(patch_target, return_value=fixed_masks.clone()):
            dense_embeddings, _ = dense.encode_views(batch)
        torch.manual_seed(1301)
        with mock.patch(patch_target, return_value=fixed_masks.clone()):
            packed_embeddings, _ = packed.encode_views(batch)

        dense_inv = canonical_invariance_loss(dense_embeddings)
        packed_inv = canonical_invariance_loss(packed_embeddings)
        torch.manual_seed(1401)
        dense_sigreg = dense.sigreg(dense_embeddings.permute(1, 0, 2))
        torch.manual_seed(1401)
        packed_sigreg = packed.sigreg(packed_embeddings.permute(1, 0, 2))
        dense_loss = dense_inv + dense.sigreg_weight * dense_sigreg
        packed_loss = packed_inv + packed.sigreg_weight * packed_sigreg
        torch.testing.assert_close(packed_loss, dense_loss, atol=2e-6, rtol=2e-5)
        dense_loss.backward()
        packed_loss.backward()
        dense_optimizer.step()
        packed_optimizer.step()

        dense_parameters = dict(dense.encoder.named_parameters())
        packed_parameters = dict(packed.encoder.named_parameters())
        self.assertEqual(dense_parameters.keys(), packed_parameters.keys())
        for name in dense_parameters:
            left_parameter = dense_parameters[name]
            right_parameter = packed_parameters[name]
            torch.testing.assert_close(
                right_parameter, left_parameter, atol=2e-6, rtol=2e-5, msg=name
            )
            left_state = dense_optimizer.state[left_parameter]
            right_state = packed_optimizer.state[right_parameter]
            self.assertEqual(left_state.keys(), right_state.keys(), name)
            for state_name in left_state:
                left = left_state[state_name]
                right = right_state[state_name]
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(
                        right,
                        left,
                        atol=2e-6,
                        rtol=2e-5,
                        msg=f"{name} optimizer state {state_name}",
                    )
                else:
                    self.assertEqual(right, left)


if __name__ == "__main__":
    unittest.main()
