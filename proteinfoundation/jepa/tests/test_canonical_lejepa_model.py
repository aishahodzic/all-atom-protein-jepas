"""Projector-free canonical model and optimizer tests."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch
from torch import nn

from proteinfoundation.jepa.canonical_lejepa_model import (
    CanonicalLeJEPAModel,
    canonical_invariance_loss,
    canonical_residue_invariance_loss,
    residue_invariance_visibility_mse,
)
from proteinfoundation.jepa.tests.canonical_test_utils import (
    synthetic_encoder_batch,
    tiny_canonical_config,
)


class CoordinateDummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8, bias=True)

    def forward(self, batch):
        mask = batch["mask_dict"]["coords"][..., 0, 0]
        mean = self.projection(batch["coords_nm"][:, :, 1]) * mask[..., None]
        # Deliberately consume RNG like the production encoder's unused z path.
        _ = torch.randn_like(mean)
        return {"mean": mean}


class RecordingSIGReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_shape = None
        self.seen_values = None

    def forward(self, values):
        self.seen_shape = tuple(values.shape)
        self.seen_values = values.detach().clone()
        return values.square().mean()


class ScriptedResidueEncoder(nn.Module):
    """Return one prescribed dense residue tensor for each canonical view."""

    def __init__(self, residue_views: torch.Tensor):
        super().__init__()
        if residue_views.ndim != 4 or residue_views.shape[1] != 5:
            raise ValueError("residue_views must have shape [B, 5, L, D]")
        self.residue_views = nn.Parameter(residue_views.clone())
        self.next_view = 0
        self.seen_masks = []

    def forward(self, batch, *, query_mask=None, key_mask=None):
        del batch
        if self.next_view >= 5:
            raise RuntimeError("scripted encoder received more than five views")
        self.seen_masks.append((query_mask, key_mask))
        result = self.residue_views[:, self.next_view]
        self.next_view += 1
        return {"mean": result}


class CanonicalModelTest(unittest.TestCase):
    def test_alignment_level_helper_default_explicit_and_model_fallback(self):
        self.assertEqual(tiny_canonical_config().alignment_level, "protein")
        self.assertEqual(
            tiny_canonical_config(alignment_level="residue").alignment_level,
            "residue",
        )
        config_without_key = tiny_canonical_config(alignment_level=None)
        self.assertNotIn("alignment_level", config_without_key)
        self.assertEqual(
            CanonicalLeJEPAModel(config_without_key).alignment_level,
            "protein",
        )

    def test_omitted_alignment_level_is_identical_to_explicit_protein(self):
        implicit = CanonicalLeJEPAModel(tiny_canonical_config(alignment_level=None))
        explicit = CanonicalLeJEPAModel(
            tiny_canonical_config(alignment_level="protein")
        )
        encoder = CoordinateDummyEncoder()
        implicit.encoder = copy.deepcopy(encoder)
        explicit.encoder = copy.deepcopy(encoder)
        implicit.sigreg = RecordingSIGReg()
        explicit.sigreg = RecordingSIGReg()
        batch = synthetic_encoder_batch(batch_size=3, length=12)

        torch.manual_seed(991)
        implicit_output = implicit.compute_loss(batch)
        torch.manual_seed(991)
        explicit_output = explicit.compute_loss(batch)

        self.assertEqual(implicit_output.keys(), explicit_output.keys())
        for key in implicit_output:
            self.assertTrue(
                torch.equal(implicit_output[key], explicit_output[key]), key
            )

    def test_invariance_is_exact_and_has_global_and_local_gradients(self):
        embeddings = torch.randn(4, 5, 8, requires_grad=True)
        expected = (embeddings[:, :1] - embeddings).pow(2).mean()
        actual = canonical_invariance_loss(embeddings)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertGreater(float(embeddings.grad[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(embeddings.grad[:, 1:].abs().sum()), 0.0)

    def test_residue_invariance_hand_case_is_same_index_and_excludes_padding(self):
        # The two real residues are deliberately far apart in value, so a
        # cross-index comparison would not produce the hand-computed answer.
        embeddings = torch.tensor(
            [
                [
                    [[0.0], [10.0], [999.0]],
                    [[1.0], [11.0], [-999.0]],
                    [[2.0], [12.0], [-999.0]],
                    [[3.0], [13.0], [-999.0]],
                    [[4.0], [14.0], [-999.0]],
                ]
            ]
        )
        real_mask = torch.tensor([[True, True, False]])
        # Sum of real, same-index squared errors is
        # 0 + 2*(1 + 4 + 9 + 16) = 60. Divide by 5 views, 2 residues,
        # and 1 latent dimension.
        expected = torch.tensor(6.0)
        actual = canonical_residue_invariance_loss(embeddings, real_mask)
        self.assertTrue(torch.equal(actual, expected))

    def test_residue_invariance_includes_visible_and_masked_but_not_padding(self):
        real_mask = torch.tensor([[True, True, False]])
        view_masks = torch.tensor(
            [
                [
                    [True, False, False],
                    [True, False, False],
                    [False, True, False],
                    [True, True, False],
                    [False, False, False],
                ]
            ]
        )
        baseline = torch.zeros(1, 5, 3, 1)

        visible_error = baseline.clone()
        visible_error[0, 1, 0, 0] = 2.0
        self.assertTrue(view_masks[0, 1, 0])
        self.assertGreater(
            float(canonical_residue_invariance_loss(visible_error, real_mask)),
            0.0,
        )

        masked_error = baseline.clone()
        masked_error[0, 1, 1, 0] = 2.0
        self.assertFalse(view_masks[0, 1, 1])
        self.assertGreater(
            float(canonical_residue_invariance_loss(masked_error, real_mask)),
            0.0,
        )

        padding_error = baseline.clone()
        padding_error[0, 1:, 2, 0] = 1_000_000.0
        self.assertEqual(
            float(canonical_residue_invariance_loss(padding_error, real_mask)),
            0.0,
        )

    def test_residue_invariance_gradients_reach_every_view_and_masked_outputs(self):
        values = torch.zeros(1, 5, 4, 2)
        for view_index in range(1, 5):
            values[:, view_index] = float(view_index)
        values.requires_grad_()
        real_mask = torch.tensor([[True, True, True, False]])
        view_masks = torch.tensor(
            [
                [
                    [True, False, True, False],
                    [True, False, False, False],
                    [False, True, False, False],
                    [True, True, False, False],
                    [False, False, True, False],
                ]
            ]
        )

        canonical_residue_invariance_loss(values, real_mask).backward()
        for view_index in range(5):
            self.assertGreater(
                float(values.grad[:, view_index, real_mask[0]].abs().sum()),
                0.0,
                f"view {view_index}",
            )
            masked_real = real_mask & ~view_masks[:, view_index]
            self.assertTrue(masked_real.any(), f"view {view_index}")
            self.assertGreater(
                float(values.grad[:, view_index][masked_real].abs().sum()),
                0.0,
                f"masked outputs in view {view_index}",
            )
        self.assertEqual(float(values.grad[:, :, 3].abs().sum()), 0.0)

    def test_residue_visibility_diagnostics_aggregate_the_four_local_views(self):
        embeddings = torch.zeros(1, 5, 4, 1)
        embeddings[:, 1:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        real_mask = torch.ones(1, 4, dtype=torch.bool)
        view_masks = torch.tensor(
            [
                [
                    [True, True, False, False],
                    [True, False, True, False],
                    [True, False, True, False],
                    [True, False, True, False],
                    [True, False, True, False],
                ]
            ]
        )
        actual = residue_invariance_visibility_mse(embeddings, real_mask, view_masks)
        expected = {
            "visible_visible": 1.0,
            "visible_masked": 4.0,
            "masked_visible": 9.0,
            "masked_masked": 16.0,
        }
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            self.assertEqual(float(actual[name]), value, name)

    def test_residue_logging_adds_only_the_six_requested_diagnostics(self):
        batch = synthetic_encoder_batch(batch_size=2, length=7)
        scripted = torch.randn(2, 5, 7, 8)
        model = CanonicalLeJEPAModel(tiny_canonical_config(alignment_level="residue"))
        model.encoder = ScriptedResidueEncoder(scripted)
        model.sigreg = RecordingSIGReg()
        model._first_batch_sanity_reported = True
        model.log_dict = mock.Mock()
        loss = model.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))

        metrics = model.log_dict.call_args.args[0]
        existing = {
            "train/loss",
            "train/inv_loss",
            "train/sigreg_loss",
            "train/global_visible_fraction",
            "train/local_visible_fraction",
            "train/embedding_std",
            "train/effective_rank",
            "train/global_local_cosine",
        }
        residue_only = {
            "train/residue_effective_rank",
            "train/residue_embedding_std",
            "train/inv_visible_visible",
            "train/inv_visible_masked",
            "train/inv_masked_visible",
            "train/inv_masked_masked",
        }
        self.assertEqual(set(metrics), existing | residue_only)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertTrue(all(not metrics[name].requires_grad for name in residue_only))

    def test_residue_mode_shape_all_real_pool_sigreg_axes_and_total_formula(self):
        torch.manual_seed(23)
        batch = synthetic_encoder_batch(batch_size=2, length=7)
        real_mask = batch["mask_dict"]["coords"][..., 0, 0]
        scripted = torch.arange(2 * 5 * 7 * 8, dtype=torch.float32).reshape(2, 5, 7, 8)
        # Make padding overwhelmingly different so its accidental inclusion is
        # unambiguous. Row 1 has one padded residue in this fixture.
        scripted[1, :, ~real_mask[1]] = 1_000_000.0

        model = CanonicalLeJEPAModel(tiny_canonical_config(alignment_level="residue"))
        model.encoder = ScriptedResidueEncoder(scripted)
        recorder = RecordingSIGReg()
        model.sigreg = recorder
        output = model.compute_loss(batch)

        self.assertEqual(tuple(output["residue_embeddings"].shape), (2, 5, 7, 8))
        expected_pooled = torch.stack(
            [
                torch.stack(
                    [scripted[b, view, real_mask[b]].mean(dim=0) for view in range(5)]
                )
                for b in range(2)
            ]
        )
        self.assertTrue(torch.equal(output["embeddings"], expected_pooled))
        self.assertEqual(recorder.seen_shape, (5, 2, 8))
        self.assertTrue(
            torch.equal(recorder.seen_values, expected_pooled.permute(1, 0, 2))
        )
        self.assertTrue(
            torch.equal(
                output["loss"],
                output["inv_loss"] + 0.02 * output["sigreg_loss"],
            )
        )
        output["loss"].backward()
        self.assertIsNotNone(model.encoder.residue_views.grad)
        self.assertGreater(float(model.encoder.residue_views.grad.abs().sum()), 0.0)

    def test_residue_alignment_rejects_packed_views(self):
        with self.assertRaisesRegex(
            ValueError, "permits views.packed_views=true only"
        ):
            CanonicalLeJEPAModel(
                tiny_canonical_config(alignment_level="residue", packed_views=True)
            )

    def test_full_projector_free_forward_backward_and_total_formula(self):
        torch.manual_seed(11)
        model = CanonicalLeJEPAModel(tiny_canonical_config())
        model.encoder = CoordinateDummyEncoder()
        batch = synthetic_encoder_batch(batch_size=4, length=16)
        output = model.compute_loss(batch)

        self.assertEqual(tuple(output["embeddings"].shape), (4, 5, 8))
        self.assertEqual(tuple(output["view_masks"].shape), (4, 5, 16))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(
            torch.equal(
                output["loss"],
                output["inv_loss"] + 0.02 * output["sigreg_loss"],
            )
        )
        output["loss"].backward()
        for parameter in model.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.encoder.parameters()
            ),
            0.0,
        )

    def test_architecture_contains_only_encoder_and_parameterless_sigreg(self):
        model = CanonicalLeJEPAModel(tiny_canonical_config())
        for forbidden in (
            "projector",
            "predictor",
            "target_encoder",
            "teacher_encoder",
            "ema_encoder",
            "mutation_embedding",
            "target_queries",
            "pos_emb",
        ):
            self.assertFalse(hasattr(model, forbidden), forbidden)
        self.assertEqual(sum(p.numel() for p in model.sigreg.parameters()), 0)
        self.assertEqual(
            set(name.split(".", 1)[0] for name, _ in model.named_parameters()),
            {"encoder"},
        )

    def test_sigreg_receives_view_protein_latent_axes(self):
        model = CanonicalLeJEPAModel(tiny_canonical_config())
        model.encoder = CoordinateDummyEncoder()
        recorder = RecordingSIGReg()
        model.sigreg = recorder
        model.compute_loss(synthetic_encoder_batch(batch_size=3, length=10))
        self.assertEqual(recorder.seen_shape, (5, 3, 8))

    def test_non_eight_dimensional_config_fails_loudly(self):
        cfg = tiny_canonical_config()
        cfg.nn_ae.latent_z_dim = 64
        with self.assertRaisesRegex(ValueError, "latent_z_dim == 8"):
            CanonicalLeJEPAModel(cfg)
        cfg = tiny_canonical_config()
        cfg.nn_ae.encoder.latent_z_dim = 64
        with self.assertRaisesRegex(ValueError, "latent_z_dim == 8"):
            CanonicalLeJEPAModel(cfg)

    def test_optimizer_groups_and_scheduler_are_exact(self):
        model = CanonicalLeJEPAModel(tiny_canonical_config())
        configured = model.configure_optimizers()
        optimizer = configured["optimizer"]
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 4e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertFalse(optimizer.defaults["amsgrad"])
        self.assertEqual(
            [group["weight_decay"] for group in optimizer.param_groups],
            [0.04, 0.0],
        )

        names_by_id = {
            id(parameter): name for name, parameter in model.encoder.named_parameters()
        }
        decayed = {
            names_by_id[id(parameter)]
            for parameter in optimizer.param_groups[0]["params"]
        }
        excluded = {
            names_by_id[id(parameter)]
            for parameter in optimizer.param_groups[1]["params"]
        }
        self.assertTrue(decayed)
        self.assertTrue(excluded)
        for name, parameter in model.encoder.named_parameters():
            should_exclude = (
                "bias" in name.lower() or "norm" in name.lower() or parameter.ndim == 1
            )
            self.assertEqual(name in excluded, should_exclude, name)
            self.assertEqual(name in decayed, not should_exclude, name)

        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        for _ in range(12_000):
            optimizer.step()
            scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 4e-4)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 4e-4)


if __name__ == "__main__":
    unittest.main()
