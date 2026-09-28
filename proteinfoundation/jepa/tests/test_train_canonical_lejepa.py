"""Resolved configuration and Lightning checkpoint integration tests."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hydra import compose, initialize_config_dir
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import RunningStage
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import torch

from proteinfoundation.utils.cluster_utils import ClusterSampler
from proteinfoundation.jepa.canonical_lejepa_ema import CanonicalEncoderEMA
from proteinfoundation.jepa.canonical_lejepa_model import CanonicalLeJEPAModel
from proteinfoundation.jepa.tests.canonical_test_utils import (
    synthetic_encoder_batch,
    tiny_canonical_config,
)
from proteinfoundation.jepa.train_canonical_lejepa import (
    build_callbacks,
    canonical_trainer_kwargs,
)
from proteinfoundation.vendor.stable_pretraining import (
    LinearWarmupCosineAnnealing,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _contains_key(value, target):
    if hasattr(value, "keys"):
        if target in value:
            return True
        return any(_contains_key(value[key], target) for key in value.keys())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, target) for item in value)
    return False


class CanonicalTrainingConfigTest(unittest.TestCase):
    def _resolved_config(self):
        with mock.patch.dict(os.environ, {"DATA_PATH": "/tmp/canonical-data"}):
            with initialize_config_dir(
                version_base=None, config_dir=str(REPO_ROOT / "configs")
            ):
                cfg = compose(config_name="training_canonical_lejepa")
                OmegaConf.resolve(cfg)
                return cfg

    def test_scientific_recipe_is_locked_and_projector_absent(self):
        cfg = self._resolved_config()
        self.assertEqual(cfg.alignment_level, "protein")
        self.assertEqual(cfg.nn_ae.latent_z_dim, 8)
        self.assertEqual(cfg.views.num_local_views, 4)
        self.assertEqual(cfg.views.global_mask_pct, 0.20)
        self.assertEqual(cfg.views.local_mask_pct, 0.80)
        self.assertFalse(cfg.views.cropping)
        self.assertFalse(cfg.views.rotation)
        self.assertFalse(cfg.views.physical_token_drop)
        self.assertTrue(cfg.views.packed_views)
        self.assertFalse(cfg.residue_compact_visible_kv)
        self.assertEqual(cfg.readout.type, "mean_visible_residues")
        self.assertEqual(cfg.sigreg.input, "protein_mean")
        self.assertEqual(cfg.sigreg.weight, 0.02)
        self.assertEqual(cfg.sigreg.knots, 17)
        self.assertEqual(cfg.sigreg.num_proj, 1024)
        self.assertFalse(cfg.sigreg.normalize_by_n)
        self.assertFalse(cfg.sigreg.joint_views)
        self.assertEqual(cfg.sigreg.residue_samples_per_protein, 1)
        self.assertFalse(cfg.sigreg.independent_residue_samples)
        self.assertFalse(_contains_key(cfg, "projector"))

        self.assertEqual(cfg.dataset.datamodule.batch_size, 8)
        self.assertEqual(cfg.dataset.datamodule.num_workers, 16)
        self.assertTrue(cfg.dataset.datamodule.skip_unprocessed)
        transform_targets = [
            entry._target_ for entry in cfg.dataset.datamodule.transforms
        ]
        self.assertEqual(
            transform_targets,
            [
                "proteinfoundation.datasets.transforms.CoordsToNanometers",
                "proteinfoundation.datasets.transforms.CenterStructureTransform",
                "proteinfoundation.datasets.transforms.ChainBreakPerResidueTransform",
            ],
        )
        self.assertFalse(
            any(
                "Rotation" in target or "Crop" in target for target in transform_targets
            )
        )

        self.assertEqual(cfg.optimizer.type, "AdamW")
        self.assertEqual(cfg.optimizer.lr, 4e-4)
        self.assertEqual(cfg.optimizer.weight_decay, 0.04)
        self.assertEqual(tuple(cfg.optimizer.betas), (0.9, 0.95))
        self.assertFalse(cfg.optimizer.amsgrad)
        self.assertEqual(cfg.schedule.total_steps, 100_000)
        self.assertEqual(cfg.schedule.peak_step, 12_000)
        self.assertEqual(cfg.schedule.start_lr, 1e-4)
        self.assertEqual(cfg.schedule.end_lr, 4e-4)
        self.assertEqual(cfg.ema.decay, 0.9999)
        self.assertEqual(cfg.ema.update_every, 32)
        self.assertEqual(list(cfg.ema.include), ["encoder"])
        self.assertFalse(cfg.pair_only_position_information)
        self.assertFalse(cfg.pair_only_evolving_pair_repr)
        self.assertFalse(cfg.sigreg.vicreg_rescue)
        self.assertEqual(cfg.checkpoint.every_n_train_steps, 10_000)
        self.assertIsNone(cfg.checkpoint.additional_every_n_train_steps)
        self.assertIsNone(cfg.checkpoint.additional_train_steps)
        self.assertEqual(cfg.checkpoint.last_every_n_train_steps, 1_500)

    def test_numbered_and_rolling_checkpoint_cadences_are_separate(self):
        cfg = self._resolved_config()
        cfg.wandb.enabled = False
        with tempfile.TemporaryDirectory() as directory:
            callbacks = build_callbacks(cfg, Path(directory))
        checkpoints = [
            callback
            for callback in callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        self.assertEqual(len(checkpoints), 2)
        numbered, rolling = checkpoints
        self.assertEqual(numbered._every_n_train_steps, 10_000)
        self.assertEqual(numbered.save_top_k, -1)
        self.assertFalse(numbered.save_last)
        self.assertEqual(rolling._every_n_train_steps, 1_500)
        self.assertEqual(rolling.save_top_k, 0)
        self.assertTrue(rolling.save_last)

    def test_additional_numbered_checkpoint_cadence_preserves_exposure_match(self):
        cfg = self._resolved_config()
        cfg.wandb.enabled = False
        cfg.checkpoint.additional_every_n_train_steps = 16_667
        with tempfile.TemporaryDirectory() as directory:
            callbacks = build_callbacks(cfg, Path(directory))
        checkpoints = [
            callback
            for callback in callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        self.assertEqual(len(checkpoints), 3)
        self.assertEqual(
            [callback._every_n_train_steps for callback in checkpoints],
            [10_000, 16_667, 1_500],
        )
        exposure_match = checkpoints[1]
        self.assertEqual(exposure_match.filename, "step-{step:08d}")
        self.assertEqual(exposure_match.save_top_k, -1)
        self.assertFalse(exposure_match.save_last)

    def test_exact_additional_checkpoint_milestones(self):
        cfg = self._resolved_config()
        cfg.wandb.enabled = False
        cfg.checkpoint.additional_train_steps = [15_000, 16_000, 17_000]
        with tempfile.TemporaryDirectory() as directory:
            callbacks = build_callbacks(cfg, Path(directory))
        checkpoints = [
            callback
            for callback in callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        self.assertEqual(
            [callback._every_n_train_steps for callback in checkpoints],
            [10_000, 15_000, 16_000, 17_000, 1_500],
        )
        for milestone in checkpoints[1:4]:
            self.assertEqual(milestone.filename, "step-{step:08d}")
            self.assertEqual(milestone.save_top_k, -1)
            self.assertFalse(milestone.save_last)

    def test_trainer_has_no_validation_or_gradient_clipping(self):
        cfg = self._resolved_config()
        kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)
        self.assertEqual(kwargs["max_steps"], 100_000)
        self.assertEqual(kwargs["devices"], 2)
        self.assertEqual(kwargs["num_nodes"], 1)
        self.assertEqual(kwargs["strategy"], "ddp")
        self.assertFalse(kwargs["use_distributed_sampler"])
        self.assertEqual(kwargs["precision"], "bf16-mixed")
        self.assertEqual(kwargs["accumulate_grad_batches"], 1)
        self.assertEqual(kwargs["limit_val_batches"], 0)
        self.assertEqual(kwargs["num_sanity_val_steps"], 0)
        self.assertIsNone(kwargs["gradient_clip_val"])

    def test_trainer_accepts_single_gpu_with_the_same_global_batch_size(self):
        cfg = self._resolved_config()
        cfg.dataset.datamodule.batch_size = 16
        cfg.trainer.devices = 1
        cfg.trainer.strategy = "auto"

        kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)

        self.assertEqual(kwargs["devices"], 1)
        self.assertEqual(kwargs["strategy"], "auto")
        self.assertEqual(cfg.dataset.datamodule.batch_size, 16)

    def test_trainer_accepts_single_a100_global_batch_96_ablation(self):
        cfg = self._resolved_config()
        cfg.dataset.datamodule.batch_size = 96
        cfg.trainer.devices = 1
        cfg.trainer.strategy = "auto"

        kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)

        self.assertEqual(kwargs["devices"], 1)
        self.assertEqual(kwargs["strategy"], "auto")
        self.assertEqual(cfg.dataset.datamodule.batch_size, 96)

    def test_trainer_accepts_two_gpu_global_batch_32_ablation(self):
        cfg = self._resolved_config()
        cfg.dataset.datamodule.batch_size = 16

        kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)

        self.assertEqual(kwargs["devices"], 2)
        self.assertEqual(cfg.dataset.datamodule.batch_size, 16)

    def test_trainer_accepts_two_gpu_global_batch_64_ablation(self):
        cfg = self._resolved_config()
        cfg.dataset.datamodule.batch_size = 32

        kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)

        self.assertEqual(kwargs["devices"], 2)
        self.assertEqual(cfg.dataset.datamodule.batch_size, 32)

    def test_trainer_accepts_audited_two_gpu_memory_probe_batches(self):
        cfg = self._resolved_config()
        for local_batch_size in (48, 64, 96):
            with self.subTest(local_batch_size=local_batch_size):
                probe_cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=False)
                )
                probe_cfg.dataset.datamodule.batch_size = local_batch_size
                kwargs = canonical_trainer_kwargs(
                    probe_cfg, [], Path("run"), None
                )
                self.assertEqual(kwargs["devices"], 2)

    def test_trainer_rejects_noncanonical_distributed_topology(self):
        cfg = self._resolved_config()

        cases = (
            ("dataset.datamodule.batch_size", 12, "requires devices=2"),
            ("trainer.devices", 1, "requires devices=2"),
            ("trainer.devices", 3, "requires devices=2"),
            ("trainer.num_nodes", 2, "requires one node"),
            ("trainer.strategy", "auto", "requires devices=2"),
            (
                "trainer.use_distributed_sampler",
                True,
                "ClusterSampler already shards",
            ),
            ("trainer.accumulate_grad_batches", 2, "accumulate_grad_batches=1"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                invalid_cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=False)
                )
                OmegaConf.update(invalid_cfg, key, value)
                with self.assertRaisesRegex(ValueError, message):
                    canonical_trainer_kwargs(invalid_cfg, [], Path("run"), None)

    def test_trainer_allows_a_resumable_prefix_of_the_100k_schedule(self):
        cfg = self._resolved_config()
        cfg.trainer.max_steps = 10_000
        kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)
        self.assertEqual(kwargs["max_steps"], 10_000)
        self.assertEqual(cfg.schedule.total_steps, 100_000)
        self.assertEqual(cfg.schedule.peak_step, 12_000)

        cfg.trainer.max_steps = 100_001
        with self.assertRaisesRegex(ValueError, "no greater than"):
            canonical_trainer_kwargs(cfg, [], Path("run"), None)

    def test_10k_prefix_lr_state_continues_the_uninterrupted_100k_warmup(self):
        cfg = tiny_canonical_config(num_proj=4)

        def optimizer_and_scheduler():
            parameter = torch.nn.Parameter(torch.tensor(0.0))
            optimizer = torch.optim.AdamW(
                [parameter],
                lr=float(cfg.optimizer.lr),
                weight_decay=float(cfg.optimizer.weight_decay),
                betas=tuple(float(value) for value in cfg.optimizer.betas),
            )
            parameter.grad = torch.zeros_like(parameter)
            scheduler = LinearWarmupCosineAnnealing(
                optimizer,
                total_steps=int(cfg.schedule.total_steps),
                peak_step=int(cfg.schedule.peak_step),
                start_factor=float(cfg.schedule.start_lr) / float(cfg.optimizer.lr),
                end_lr=float(cfg.schedule.end_lr),
            )
            return optimizer, scheduler

        uninterrupted_optimizer, uninterrupted_scheduler = optimizer_and_scheduler()
        prefix_optimizer, prefix_scheduler = optimizer_and_scheduler()
        uninterrupted_lrs = []
        prefix_lrs = []
        for _ in range(10_000):
            uninterrupted_optimizer.step()
            uninterrupted_scheduler.step()
            uninterrupted_lrs.append(uninterrupted_optimizer.param_groups[0]["lr"])
            prefix_optimizer.step()
            prefix_scheduler.step()
            prefix_lrs.append(prefix_optimizer.param_groups[0]["lr"])

        self.assertEqual(prefix_lrs, uninterrupted_lrs)
        self.assertAlmostEqual(prefix_lrs[-1], 3.5e-4, places=12)
        self.assertEqual(prefix_scheduler.last_epoch, 10_000)

        resumed_optimizer, resumed_scheduler = optimizer_and_scheduler()
        resumed_optimizer.load_state_dict(prefix_optimizer.state_dict())
        resumed_scheduler.load_state_dict(prefix_scheduler.state_dict())
        self.assertEqual(resumed_scheduler.last_epoch, 10_000)

        uninterrupted_optimizer.step()
        uninterrupted_scheduler.step()
        resumed_optimizer.step()
        resumed_scheduler.step()
        self.assertEqual(
            resumed_optimizer.param_groups[0]["lr"],
            uninterrupted_optimizer.param_groups[0]["lr"],
        )
        self.assertEqual(resumed_scheduler.last_epoch, 10_001)

    def test_cluster_sampler_is_not_double_sharded_by_lightning(self):
        class TinyPDBDataset(Dataset):
            database = "pdb"
            file_names = [f"sequence-{index}.pt" for index in range(8)]

            def __len__(self):
                return len(self.file_names)

            def __getitem__(self, index):
                return index

        cfg = self._resolved_config()
        canonical_kwargs = canonical_trainer_kwargs(cfg, [], Path("run"), None)
        dataset = TinyPDBDataset()
        mapping = {f"sequence-{index}": [f"sequence-{index}"] for index in range(8)}
        sampler = ClusterSampler(dataset, mapping, sampling_mode="cluster-reps")
        dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            use_distributed_sampler=canonical_kwargs["use_distributed_sampler"],
        )

        connector = trainer._accelerator_connector
        with mock.patch.object(
            type(connector),
            "is_distributed",
            new_callable=mock.PropertyMock,
            return_value=True,
        ):
            prepared = trainer._data_connector._prepare_dataloader(
                dataloader,
                shuffle=True,
                mode=RunningStage.TRAINING,
            )

        self.assertIs(prepared, dataloader)
        self.assertIs(prepared.sampler, sampler)

    def test_full_lightning_checkpoint_resume_round_trip(self):
        L.seed_everything(123, workers=True)
        cfg = tiny_canonical_config(num_proj=4)
        batch = synthetic_encoder_batch(batch_size=2, length=8)
        loader = DataLoader([batch, batch], batch_size=None)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first_ema = CanonicalEncoderEMA(
                decay=0.9999,
                update_every=1,
                include=("encoder",),
                device="cpu",
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=directory,
                every_n_train_steps=1,
                save_top_k=-1,
                save_last=True,
                filename="step-{step}",
            )
            first_model = CanonicalLeJEPAModel(cfg)
            first_trainer = L.Trainer(
                max_steps=2,
                max_epochs=-1,
                accelerator="cpu",
                devices=1,
                precision="32-true",
                callbacks=[first_ema, checkpoint_callback],
                logger=False,
                enable_progress_bar=False,
                limit_val_batches=0,
                num_sanity_val_steps=0,
                gradient_clip_val=None,
                log_every_n_steps=1,
            )
            first_trainer.fit(first_model, train_dataloaders=loader)
            checkpoint_path = directory / "last.ckpt"
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            for key in (
                "state_dict",
                "state_dict_ema",
                "state_dict_ema_meta",
                "optimizer_states",
                "lr_schedulers",
                "global_step",
                "callbacks",
            ):
                self.assertIn(key, checkpoint)
            self.assertEqual(checkpoint["global_step"], 2)
            self.assertFalse(
                any("projector" in key for key in checkpoint["state_dict"])
            )

            resumed_ema = CanonicalEncoderEMA(
                decay=0.9999,
                update_every=1,
                include=("encoder",),
                device="cpu",
            )
            resumed_model = CanonicalLeJEPAModel(cfg)
            resumed_trainer = L.Trainer(
                max_steps=3,
                max_epochs=-1,
                accelerator="cpu",
                devices=1,
                precision="32-true",
                callbacks=[resumed_ema],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                limit_val_batches=0,
                num_sanity_val_steps=0,
                gradient_clip_val=None,
                log_every_n_steps=1,
            )
            resumed_trainer.fit(
                resumed_model, train_dataloaders=loader, ckpt_path=checkpoint_path
            )
            self.assertEqual(resumed_trainer.global_step, 3)
            self.assertEqual(resumed_ema._num_updates, 3)
            self.assertEqual(resumed_ema._last_ema_step, 3)


if __name__ == "__main__":
    unittest.main()
