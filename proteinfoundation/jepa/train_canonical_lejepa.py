"""Dedicated training entry point for projector-free canonical LeJEPA V1."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from proteinfoundation.jepa.canonical_lejepa_ema import CanonicalEncoderEMA
from proteinfoundation.jepa.canonical_lejepa_model import CanonicalLeJEPAModel


def get_run_directories(run_name: str) -> tuple[Path, Path]:
    root = Path("store") / run_name
    return root, root / "checkpoints"


class CanonicalCohortReport(Callback):
    """Report the already-resolved cohort without invoking data setup twice."""

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        datamodule = trainer.datamodule
        train_rows = len(datamodule.train_ds)
        selected_rows = (
            len(datamodule.df_data)
            if getattr(datamodule, "df_data", None) is not None
            else None
        )
        steps_per_epoch = len(trainer.train_dataloader)
        logger.info(
            "Canonical cohort: selected_rows={}, train_rows={}, batch_size={}, "
            "optimizer_steps_per_epoch={}; skip_unprocessed={}. The exact number "
            "dropped before a cached cohort CSV was written is only available in "
            "the data-preparation log.",
            selected_rows,
            train_rows,
            datamodule.batch_size,
            steps_per_epoch,
            getattr(datamodule, "skip_unprocessed", None),
        )


def build_callbacks(cfg: DictConfig, checkpoint_dir: Path) -> list[Any]:
    """Build only evaluation EMA, checkpointing, and optional LR logging."""
    callbacks: list[Any] = []
    if bool(cfg.ema.enabled):
        ema_cfg = OmegaConf.to_container(cfg.ema, resolve=True)
        ema_cfg.pop("enabled")
        callbacks.append(CanonicalEncoderEMA(**ema_cfg))
    callbacks.append(CanonicalCohortReport())

    checkpoint_cfg = OmegaConf.to_container(cfg.checkpoint, resolve=True)
    checkpoint_cfg["dirpath"] = str(checkpoint_dir)
    additional_every_n_train_steps = checkpoint_cfg.pop(
        "additional_every_n_train_steps", None
    )
    last_every_n_train_steps = checkpoint_cfg.pop(
        "last_every_n_train_steps", None
    )
    save_last = bool(checkpoint_cfg.get("save_last", False))
    if last_every_n_train_steps is not None:
        checkpoint_cfg["save_last"] = False
    callbacks.append(ModelCheckpoint(**checkpoint_cfg))
    if additional_every_n_train_steps is not None:
        additional_interval = int(additional_every_n_train_steps)
        if additional_interval < 1:
            raise ValueError(
                "checkpoint.additional_every_n_train_steps must be positive"
            )
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=str(checkpoint_cfg["filename"]),
                every_n_train_steps=additional_interval,
                every_n_epochs=0,
                save_top_k=-1,
                save_last=False,
                save_weights_only=bool(checkpoint_cfg["save_weights_only"]),
                auto_insert_metric_name=bool(
                    checkpoint_cfg["auto_insert_metric_name"]
                ),
            )
        )
    if last_every_n_train_steps is not None and save_last:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="recovery-{step:08d}",
                every_n_train_steps=int(last_every_n_train_steps),
                every_n_epochs=0,
                save_top_k=0,
                save_last=True,
                save_weights_only=bool(checkpoint_cfg["save_weights_only"]),
                auto_insert_metric_name=False,
            )
        )
    if cfg.wandb.enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def canonical_trainer_kwargs(
    cfg: DictConfig, callbacks: list[Any], run_root: Path, experiment_logger: Any
) -> dict[str, Any]:
    """Return auditable trainer settings with no validation or clipping."""
    trainer_cfg = cfg.trainer
    max_steps = int(trainer_cfg.max_steps)
    schedule_total_steps = int(cfg.schedule.total_steps)
    if max_steps < 1 or max_steps > schedule_total_steps:
        raise ValueError(
            "trainer.max_steps must be positive and no greater than "
            "schedule.total_steps"
        )
    if int(trainer_cfg.accumulate_grad_batches) != 1:
        raise ValueError("canonical V1 requires accumulate_grad_batches=1")
    local_batch_size = int(cfg.dataset.datamodule.batch_size)
    devices = int(trainer_cfg.devices)
    num_nodes = int(trainer_cfg.num_nodes)
    accumulate_grad_batches = int(trainer_cfg.accumulate_grad_batches)
    global_batch_size = (
        local_batch_size * devices * num_nodes * accumulate_grad_batches
    )
    if num_nodes != 1:
        raise ValueError("canonical V1 requires one node")
    topology = (devices, local_batch_size, str(trainer_cfg.strategy))
    allowed_topologies = {
        (2, 8, "ddp"),
        (2, 16, "ddp"),
        (2, 32, "ddp"),
        (2, 48, "ddp"),
        (2, 64, "ddp"),
        (2, 96, "ddp"),
        (1, 16, "auto"),
        (1, 96, "auto"),
    }
    if topology not in allowed_topologies:
        raise ValueError(
            "canonical V1 requires devices=2 with an audited local batch size "
            "and strategy=ddp, or devices=1 with local batch size 16 and "
            "strategy=auto"
        )
    if bool(trainer_cfg.use_distributed_sampler):
        raise ValueError(
            "canonical V1 requires use_distributed_sampler=false because "
            "ClusterSampler already shards by rank/world size"
        )
    if global_batch_size not in (16, 32, 64, 96, 128, 192):
        raise ValueError(
            "canonical V1 requires an audited global batch size in "
            "{16, 32, 64, 96, 128, 192}"
        )
    if int(trainer_cfg.limit_val_batches) != 0:
        raise ValueError("canonical V1 disables validation")
    if int(trainer_cfg.num_sanity_val_steps) != 0:
        raise ValueError("canonical V1 disables sanity validation")
    if trainer_cfg.gradient_clip_val is not None:
        raise ValueError("canonical V1 disables gradient clipping")

    plugins = (
        [SLURMEnvironment(auto_requeue=True)]
        if "SLURM_JOB_ID" in os.environ
        else []
    )
    return {
        "max_steps": max_steps,
        "max_epochs": int(trainer_cfg.max_epochs),
        "accelerator": trainer_cfg.accelerator,
        "devices": devices,
        "num_nodes": num_nodes,
        "strategy": trainer_cfg.strategy,
        "use_distributed_sampler": bool(trainer_cfg.use_distributed_sampler),
        "precision": trainer_cfg.precision,
        "accumulate_grad_batches": accumulate_grad_batches,
        "limit_val_batches": int(trainer_cfg.limit_val_batches),
        "num_sanity_val_steps": int(trainer_cfg.num_sanity_val_steps),
        "gradient_clip_val": trainer_cfg.gradient_clip_val,
        "callbacks": callbacks,
        "logger": experiment_logger,
        "log_every_n_steps": int(trainer_cfg.log_every_n_steps),
        "enable_checkpointing": bool(trainer_cfg.enable_checkpointing),
        "default_root_dir": str(run_root),
        "plugins": plugins,
    }


def store_resolved_config(cfg: DictConfig, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"exp_config_{cfg.run_name_}.json"
    path.write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training_canonical_lejepa",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    L.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    run_root, checkpoint_dir = get_run_directories(str(cfg.run_name_))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store_resolved_config(cfg, checkpoint_dir)

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    model = CanonicalLeJEPAModel(cfg)

    experiment_logger = None
    if cfg.wandb.enabled:
        experiment_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name_,
            id=cfg.run_name_,
            resume="allow",
        )
        experiment_logger.log_hyperparams(
            OmegaConf.to_container(cfg, resolve=True)
        )

    callbacks = build_callbacks(cfg, checkpoint_dir)
    trainer = L.Trainer(
        **canonical_trainer_kwargs(cfg, callbacks, run_root, experiment_logger)
    )

    resume_path = cfg.resume.ckpt_path
    if resume_path is not None:
        resume_path = str(Path(resume_path).expanduser().resolve())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        logger.info("Full-state resume from {}", resume_path)
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
