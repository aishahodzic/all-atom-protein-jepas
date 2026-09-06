"""Independent adversarial tests for canonical packed-view equivalence.

These tests were designed from the scientific packed-view contract and the
production implementation before inspecting the repository's canonical tests.
They intentionally cover only gaps found after that independent plan was
frozen.
"""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from omegaconf import OmegaConf
import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CanonicalLeJEPAModel,
    canonical_invariance_loss,
    mask_batch_for_canonical_view,
    mean_pool_visible_residues,
    pack_batch_for_canonical_view,
)
from proteinfoundation.nn.feature_factory import SequenceSeparationPairFeat
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer


ATOL = 2e-6
RTOL = 2e-5


def _config(*, packed: bool):
    """Small configuration reproducing the exact packed-safe feature path."""
    return OmegaConf.create(
        {
            "nn_ae": {
                "latent_z_dim": 8,
                "encoder": {
                    "nlayers": 1,
                    "token_dim": 16,
                    "nheads": 4,
                    "parallel_mha_transition": False,
                    "strict_feats": False,
                    "feats_seq": [
                        "chain_break_per_res",
                        "x1_aatype",
                        "x1_a37coors_nm",
                        "x1_a37coors_nm_rel",
                        "chain_idx_seq",
                    ],
                    "feats_cond_seq": [],
                    "dim_cond": 128,
                    "idx_emb_dim": 8,
                    "feats_pair_repr": [
                        "rel_seq_sep",
                        "x1_bb_pair_dists_nm",
                        "x1_bb_pair_orientation",
                        "chain_idx_pair",
                    ],
                    "seq_sep_dim": 15,
                    "pair_repr_dim": 8,
                    "update_pair_repr": False,
                    "update_pair_repr_every_n": 1,
                    "use_tri_mult": False,
                    "use_qkln": True,
                    "normalize_latent": False,
                    "latent_z_dim": 8,
                },
                "decoder": {"latent_z_dim": 8},
            },
            "views": {
                "num_local_views": 4,
                "global_mask_pct": 0.20,
                "local_mask_pct": 0.80,
                "cropping": False,
                "rotation": False,
                "physical_token_drop": False,
                "packed_views": packed,
            },
            "readout": {"type": "mean_visible_residues"},
            "sigreg": {
                "weight": 0.02,
                "knots": 5,
                "num_proj": 11,
                "normalize_by_n": False,
            },
            "optimizer": {
                "type": "AdamW",
                "lr": 4e-4,
                "weight_decay": 0.04,
                "betas": [0.9, 0.95],
                "amsgrad": False,
                "exclude_bias_norm": True,
            },
            "schedule": {
                "type": "LinearWarmupCosineAnnealing",
                "total_steps": 100,
                "peak_step": 12,
                "start_lr": 1e-4,
                "end_lr": 4e-4,
            },
        }
    )


def _batch() -> dict:
    generator = torch.Generator().manual_seed(731)
    batch_size, length = 3, 8
    valid_lengths = torch.tensor([8, 7, 5])
    valid = torch.arange(length)[None] < valid_lengths[:, None]
    coords_nm = torch.randn(batch_size, length, 37, 3, generator=generator)
    coords_nm = coords_nm.masked_fill(~valid[..., None, None], 0.0)
    coords = coords_nm * 10.0
    coord_mask = valid[..., None].expand(-1, -1, 37).clone()
    residue_type = (
        torch.arange(batch_size * length).reshape(batch_size, length) % 20
    ).long()
    residue_type.masked_fill_(~valid, -1)
    residue_pdb_idx = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [21, 22, 23, 80, 81, 82, 83, -1],
            [1, 2, 1, 2, 3, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    chains = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 1, 1, 1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    chain_breaks = torch.zeros(batch_size, length, dtype=torch.bool)
    chain_breaks[1, 2] = True
    chain_breaks[2, 1] = True
    tensors = {
        "coords": coords,
        "coords_nm": coords_nm,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "residue_pdb_idx": residue_pdb_idx,
        "chains": chains,
        "chain_breaks_per_residue": chain_breaks,
    }
    masks = {
        key: valid.reshape(batch_size, length, *([1] * (value.ndim - 2)))
        .expand_as(value)
        .clone()
        for key, value in tensors.items()
    }
    return {**tensors, "mask_dict": masks, "audit_sentinel": object()}


def _view_masks() -> torch.Tensor:
    selections = (
        ((0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4), (0, 1, 2, 3)),
        ((1, 4, 7), (0, 3), (1,)),
        ((0, 5), (2, 6), (0, 3)),
        ((2, 3, 6), (1,), (2, 4)),
        ((0, 7), (0, 4, 6), (3,)),
    )
    masks = torch.zeros(3, 5, 8, dtype=torch.bool)
    for view, rows in enumerate(selections):
        for row, indices in enumerate(rows):
            masks[row, view, list(indices)] = True
    return masks


def _assert_nested_batch_equal(test: unittest.TestCase, left: dict, right: dict):
    test.assertEqual(left.keys(), right.keys())
    for key in left:
        if isinstance(left[key], dict):
            _assert_nested_batch_equal(test, left[key], right[key])
        elif isinstance(left[key], torch.Tensor):
            test.assertTrue(torch.equal(left[key], right[key]), key)
        else:
            test.assertIs(left[key], right[key], key)


class ExactSparseIndexOracleTest(unittest.TestCase):
    def test_residues_1_through_8_retain_2_5_8_without_reindexing(self):
        batch = _batch()
        one = {
            key: ({nested: value[:1] for nested, value in item.items()}
                  if isinstance(item, dict) else item[:1]
                  if isinstance(item, torch.Tensor) else item)
            for key, item in batch.items()
        }
        visible = torch.tensor(
            [[False, True, False, False, True, False, False, True]]
        )
        packed = pack_batch_for_canonical_view(one, visible)

        self.assertTrue(
            torch.equal(
                packed.batch["residue_pdb_idx"], torch.tensor([[2, 5, 8]])
            )
        )
        self.assertTrue(
            torch.equal(packed.original_indices, torch.tensor([[1, 4, 7]]))
        )

        rel = SequenceSeparationPairFeat(seq_sep_dim=15)(packed.batch)
        # For 15 bins, integer separation d maps to bin d + 7 in this range.
        expected_bins = torch.tensor(
            [[[7, 4, 1], [10, 7, 4], [13, 10, 7]]]
        )
        packed_adjacency_bins = torch.tensor(
            [[[7, 6, 5], [8, 7, 6], [9, 8, 7]]]
        )
        actual_bins = rel.argmax(dim=-1)
        self.assertTrue(torch.equal(actual_bins, expected_bins))
        self.assertFalse(torch.equal(actual_bins, packed_adjacency_bins))


class PackedBoundaryContractTest(unittest.TestCase):
    def test_padding_sentinels_masking_and_source_immutability(self):
        batch = _batch()
        before = copy.deepcopy(batch)
        before["audit_sentinel"] = batch["audit_sentinel"]
        visible = torch.zeros(3, 8, dtype=torch.bool)
        visible[0, [1, 4, 7]] = True
        visible[1, [3, 6]] = True
        visible[2, [1]] = True
        packed = pack_batch_for_canonical_view(batch, visible)
        _assert_nested_batch_equal(self, batch, before)

        for row, count in enumerate((3, 2, 1)):
            padding = slice(count, 3)
            self.assertTrue((packed.original_indices[row, padding] == -1).all())
            self.assertTrue((packed.batch["residue_type"][row, padding] == -1).all())
            self.assertTrue((packed.batch["residue_pdb_idx"][row, padding] == -1).all())
            self.assertTrue((packed.batch["chains"][row, padding] == -1).all())
            self.assertFalse(packed.batch["chain_breaks_per_residue"][row, padding].any())
            self.assertTrue((packed.batch["coords"][row, padding] == 0).all())
            self.assertTrue((packed.batch["coords_nm"][row, padding] == 0).all())
            self.assertFalse(packed.batch["coord_mask"][row, padding].any())
            self.assertFalse(packed.valid_mask[row, padding].any())
            for mask in packed.batch["mask_dict"].values():
                self.assertFalse(mask[row, padding].any())

        encoder = EncoderTransformer(**_config(packed=False).nn_ae).eval()
        perturbed = copy.deepcopy(packed.batch)
        padding = ~packed.valid_mask
        perturbed["coords"][padding] = 1e6
        perturbed["coords_nm"][padding] = -1e5
        perturbed["residue_type"][padding] = 19
        perturbed["residue_pdb_idx"][padding] = 999
        perturbed["chains"][padding] = 77
        perturbed["chain_breaks_per_residue"][padding] = True
        with torch.no_grad():
            original_mean = encoder(packed.batch)["mean"]
            changed_mean = encoder(perturbed)["mean"]
        self.assertTrue(
            torch.allclose(
                original_mean[packed.valid_mask],
                changed_mean[packed.valid_mask],
                atol=ATOL,
                rtol=RTOL,
            )
        )
        self.assertTrue(
            torch.allclose(
                mean_pool_visible_residues(original_mean, packed.valid_mask),
                mean_pool_visible_residues(changed_mean, packed.valid_mask),
                atol=ATOL,
                rtol=RTOL,
            )
        )

    def test_wrong_visibility_shape_fails_before_gather(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            pack_batch_for_canonical_view(
                _batch(), torch.ones(3, 7, dtype=torch.bool)
            )

    def test_allowlist_removal_and_sequence_reordering_fail_closed(self):
        mutations = {
            "remove sequence feature": lambda cfg: cfg.nn_ae.encoder.feats_seq.pop(),
            "remove pair feature": lambda cfg: cfg.nn_ae.encoder.feats_pair_repr.pop(),
            "reorder sequence features": lambda cfg: cfg.nn_ae.encoder.feats_seq.reverse(),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                cfg = _config(packed=True)
                mutation(cfg)
                with self.assertRaisesRegex(ValueError, "exact audited"):
                    CanonicalLeJEPAModel(cfg)

    def test_historical_dense_mask_in_place_contract_is_unchanged(self):
        batch = _batch()
        before = copy.deepcopy(batch)
        before["audit_sentinel"] = batch["audit_sentinel"]
        valid = batch["mask_dict"]["coords"][..., 0, 0]
        visible = torch.zeros_like(valid)
        visible[0, [1, 4, 7]] = True
        visible[1, [0, 3]] = True
        visible[2, [1, 4]] = True

        masked = mask_batch_for_canonical_view(batch, visible)
        _assert_nested_batch_equal(self, batch, before)
        self.assertTrue(torch.equal(masked["mask"], valid))
        self.assertTrue(
            torch.equal(
                masked["mask_dict"]["coords"],
                batch["mask_dict"]["coords"] & visible[..., None, None],
            )
        )
        hidden = ~visible
        self.assertTrue((masked["coords"][hidden] == 0).all())
        self.assertTrue((masked["coords_nm"][hidden] == 0).all())
        self.assertTrue(torch.equal(masked["coords"][visible], batch["coords"][visible]))
        self.assertTrue(
            torch.equal(masked["coords_nm"][visible], batch["coords_nm"][visible])
        )
        self.assertIs(masked["coord_mask"], batch["coord_mask"])
        self.assertIs(masked["residue_pdb_idx"], batch["residue_pdb_idx"])
        self.assertIs(
            masked["mask_dict"]["residue_type"],
            batch["mask_dict"]["residue_type"],
        )
        self.assertIs(masked["audit_sentinel"], batch["audit_sentinel"])


class OneOptimizerStepParityTest(unittest.TestCase):
    def test_dense_and_packed_adamw_step_and_state_match(self):
        batch = _batch()
        fixed_masks = _view_masks()
        torch.manual_seed(1201)
        dense = CanonicalLeJEPAModel(_config(packed=False)).train()
        packed = CanonicalLeJEPAModel(_config(packed=True)).train()
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
        dense_loss.backward()
        packed_loss.backward()
        dense_optimizer.step()
        packed_optimizer.step()

        dense_parameters = dict(dense.encoder.named_parameters())
        packed_parameters = dict(packed.encoder.named_parameters())
        self.assertEqual(dense_parameters.keys(), packed_parameters.keys())
        for name in dense_parameters:
            dense_parameter = dense_parameters[name]
            packed_parameter = packed_parameters[name]
            self.assertTrue(
                torch.allclose(
                    dense_parameter,
                    packed_parameter,
                    atol=ATOL,
                    rtol=RTOL,
                ),
                name,
            )
            dense_state = dense_optimizer.state[dense_parameter]
            packed_state = packed_optimizer.state[packed_parameter]
            self.assertEqual(dense_state.keys(), packed_state.keys(), name)
            for key in dense_state:
                left, right = dense_state[key], packed_state[key]
                if isinstance(left, torch.Tensor):
                    self.assertTrue(
                        torch.allclose(left, right, atol=ATOL, rtol=RTOL),
                        f"{name} optimizer state {key}",
                    )
                else:
                    self.assertEqual(left, right, f"{name} optimizer state {key}")


if __name__ == "__main__":
    unittest.main()
