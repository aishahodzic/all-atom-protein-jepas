import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
from glob import glob
import hydra
import omegaconf

import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
from rich.text import Text

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluation_manifest import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    artifact_identity,
    canonical_json_sha256,
    file_identity,
    git_identity,
    runtime_identity,
    safe_path_component,
    scientific_runtime_versions,
    write_immutable_json,
)
from laproteina_eval_policy import (  # noqa: E402
    UPSTREAM_L600_PROTOCOL,
    resolve_laproteina_evaluation_policy,
)

# Env-var-gated, not a hardcoded call: this line reruns from scratch inside every
# spawn-started worker process (spawn re-executes top-level module code on import,
# unlike fork), including dataset/base.py's mp.get_context("spawn").Pool() used by
# cache_all_tokenized() -- so a runtime torch.multiprocessing.set_sharing_strategy()
# call made only in data_module.py's parent process never reaches those workers; they
# independently re-hit this exact line and reset themselves to whatever's hardcoded
# here. Env vars, unlike Python globals, ARE inherited by spawned children, so this is
# the one lever that actually reaches them. Default "file_system" preserves the
# original fix (avoids "received 0 items of ancdata" on the long multi-hour DataLoader);
# data_module.py temporarily sets TORCH_SHARING_STRATEGY=file_descriptor around the
# one-shot cache_all_tokenized() precompute, which needs the OTHER strategy instead
# (file_system's per-tensor mmap blew through vm.max_map_count for large splits).
torch.multiprocessing.set_sharing_strategy(os.environ.get("TORCH_SHARING_STRATEGY", "file_system"))

# Metrics we care most about for generalization (held-out CATH folds/superfamilies,
# not just i.i.d. validation): AUROC for classification tasks (bind/cat/con/rep/ept),
# spearmanr for regression tasks (flex*). Highlighted in the results table and
# pulled into the cross-run summary; absent (and skipped) for tasks with different
# split names (BindShake, Homo).
HIGHLIGHT_METRICS = ("auroc", "spearmanr")
HIGHLIGHT_SPLITS = ("fold_test", "superfamily_test")


def _json_safe(value, non_finite_fields=None, prefix=""):
    """Convert OmegaConf/metric values to strict JSON (no NaN/Infinity)."""

    if torch.is_tensor(value):
        value = value.detach().cpu()
        return _json_safe(value.item() if value.numel() == 1 else value.tolist(), non_finite_fields, prefix)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(
                item,
                non_finite_fields,
                f"{prefix}.{key}" if prefix else str(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, non_finite_fields, f"{prefix}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if non_finite_fields is not None:
            non_finite_fields.append(prefix)
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def validate_locked_protocol_config(cfg):
    """Fail early if a direct Hydra invocation contradicts the named lock."""

    protocol = getattr(cfg, "tokenizer_eval_protocol", None)
    if protocol in (None, ""):
        return None
    if cfg.tokenizer != "WrappedLaProteinaTokenizer":
        raise ValueError(f"{protocol} is only valid for WrappedLaProteinaTokenizer")

    policy = resolve_laproteina_evaluation_policy(
        protocol,
        encoder_source=getattr(cfg, "tokenizer_encoder_source", "online"),
        evaluation_max_length=getattr(cfg, "tokenizer_eval_max_length", None),
        cache_tag=getattr(cfg, "tokenizer_cache_tag", None),
    )
    if int(cfg.data.filter_length) != policy.stb_filter_length:
        raise ValueError(
            f"{protocol} locks data.filter_length={policy.stb_filter_length}, "
            f"got {cfg.data.filter_length}"
        )
    if not bool(getattr(cfg, "precompute_tokens", False)):
        raise ValueError(f"{protocol} requires precompute_tokens=true for cohort auditing")
    if bool(getattr(cfg, "validate_only", False)) or bool(
        getattr(cfg, "test_only", False)
    ):
        raise ValueError(f"{protocol} requires a fresh train-and-evaluate probe run")

    expected_micro_batch = (
        64 if cfg.data.data_name == "TapeRemoteHomologyDataset" else 8
    )
    locked_values = {
        "optimization.seed": (int(cfg.optimization.seed), 1234),
        "optimization.micro_batch_size": (
            int(cfg.optimization.micro_batch_size),
            expected_micro_batch,
        ),
        "optimization.optimizer.lr": (
            float(cfg.optimization.optimizer.lr),
            0.001,
        ),
        "model.dropout": (float(cfg.model.dropout), 0.1),
        "model.num_layer": (int(cfg.model.num_layer), 1),
        "model.hidden_size": (int(cfg.model.hidden_size), 512),
        "model.use_sequence": (bool(cfg.model.use_sequence), False),
        "model.sequence_only": (bool(cfg.model.sequence_only), False),
        "trainer.precision": (str(cfg.trainer.precision), "32-true"),
        "trainer.deterministic": (bool(cfg.trainer.deterministic), True),
        "trainer.reload_dataloaders_every_n_epochs": (
            int(cfg.trainer.reload_dataloaders_every_n_epochs),
            1,
        ),
        "trainer.accumulate_grad_batches": (
            int(cfg.trainer.accumulate_grad_batches),
            1,
        ),
        "trainer.gradient_clip_val": (float(cfg.trainer.gradient_clip_val), 1.0),
        "trainer.val_check_interval": (float(cfg.trainer.val_check_interval), 1.0),
        "data.truncation_length": (int(cfg.data.truncation_length), 5000),
        "data.use_sequence": (bool(cfg.data.use_sequence), False),
        "data.use_continuous": (bool(cfg.data.use_continuous), False),
    }
    for name, (actual, expected) in locked_values.items():
        if actual != expected:
            raise ValueError(f"{protocol} locks {name}={expected!r}, got {actual!r}")

    smoke = os.environ.get("STB_SMOKE", "0") == "1"
    expected_validation_cadence = 1 if smoke else 2
    if int(cfg.trainer.check_val_every_n_epoch) != expected_validation_cadence:
        raise ValueError(
            f"{protocol} locks validation to every "
            f"{expected_validation_cadence} epoch(s) for "
            f"{'smoke' if smoke else 'full'} runs"
        )
    expected_max_steps = 2 if smoke else 10000
    if int(cfg.max_steps) != expected_max_steps:
        raise ValueError(
            f"{protocol} locks max_steps={expected_max_steps} "
            f"for {'smoke' if smoke else 'full'} runs, got {cfg.max_steps}"
        )
    if int(cfg.lightning.callbacks.early_stopping.patience) != 10:
        raise ValueError(f"{protocol} locks early-stopping patience=10")
    expected_checkpoint_cadence = 1
    if int(cfg.lightning.callbacks.checkpoint.every_n_epochs) != expected_checkpoint_cadence:
        raise ValueError(f"{protocol} locks checkpoint saving at every validation event")
    expected_limits = (1, 2, 2) if smoke else (None, None, None)
    actual_limits = (
        cfg.trainer.max_epochs,
        cfg.trainer.limit_train_batches,
        cfg.trainer.limit_val_batches,
    )
    if actual_limits != expected_limits:
        raise ValueError(
            f"{protocol} locks (max_epochs, limit_train_batches, "
            f"limit_val_batches)={expected_limits!r} for this run mode, "
            f"got {actual_limits!r}"
        )
    expected_monitor = (
        "validation_spearmanr" if bool(cfg.model.regression) else "validation_f1_score"
    )
    checkpoint_callback = cfg.lightning.callbacks.checkpoint
    early_stopping = cfg.lightning.callbacks.early_stopping
    if (
        str(checkpoint_callback.monitor) != expected_monitor
        or str(early_stopping.monitor) != expected_monitor
        or str(checkpoint_callback.mode) != "max"
        or str(early_stopping.mode) != "max"
    ):
        raise ValueError(
            f"{protocol} locks checkpoint/early-stop monitoring to "
            f"{expected_monitor!r} in max mode"
        )
    aggregation = getattr(cfg.model, "classification_metric_aggregation", None)
    if aggregation is not None and aggregation != "legacy_batch_mean":
        raise ValueError(f"{protocol} requires upstream legacy_batch_mean metrics")
    weighting = getattr(cfg.model, "binary_class_weighting", None)
    if weighting is not None and weighting != "legacy_batch":
        raise ValueError(f"{protocol} requires upstream legacy_batch class weighting")
    return policy


def _source_file_identities(repository_dir):
    candidates = {
        "feature_factory": repository_dir / "proteinfoundation" / "nn" / "feature_factory.py",
        "tokenizer": _SRC_DIR / "tokenizer.py",
        "dataset_base": _SRC_DIR / "dataset" / "base.py",
        "data_module": _SRC_DIR / "data_module.py",
        "model_module": _SRC_DIR / "model_module.py",
        "metric_aggregation": _SRC_DIR / "metric_aggregation.py",
        "run_supervised_task": Path(__file__).resolve(),
        "shared_embedding_cache": _SRC_DIR / "shared_embedding_cache.py",
    }
    return {
        name: file_identity(path)
        for name, path in candidates.items()
        if path.exists()
    }


def _scientific_probe_config(cfg):
    """Path-independent settings that can change the downstream result."""

    model = _json_safe(omegaconf.OmegaConf.to_container(cfg.model, resolve=True))
    for operational_key in ("pretrained_ckpt_path", "ckpt_path"):
        model.pop(operational_key, None)
    data = _json_safe(omegaconf.OmegaConf.to_container(cfg.data, resolve=True))
    for operational_key in ("data_path", "pdb_data_dir", "num_workers", "prefetch_factor"):
        data.pop(operational_key, None)
    return {
        "model": model,
        "data": data,
        "optimization": _json_safe(
            omegaconf.OmegaConf.to_container(cfg.optimization, resolve=True)
        ),
        "trainer": {
            "precision": str(cfg.trainer.precision),
            "gradient_clip_val": _json_safe(cfg.trainer.gradient_clip_val),
            "accumulate_grad_batches": int(cfg.trainer.accumulate_grad_batches),
            "deterministic": bool(cfg.trainer.deterministic),
            "reload_dataloaders_every_n_epochs": int(
                cfg.trainer.reload_dataloaders_every_n_epochs
            ),
            "val_check_interval": _json_safe(cfg.trainer.val_check_interval),
            "limit_train_batches": _json_safe(cfg.trainer.limit_train_batches),
            "limit_val_batches": _json_safe(cfg.trainer.limit_val_batches),
        },
        "stopping": {
            "max_steps": int(cfg.max_steps),
            "early_stopping_monitor": str(
                cfg.lightning.callbacks.early_stopping.monitor
            ),
            "early_stopping_mode": str(cfg.lightning.callbacks.early_stopping.mode),
            "early_stopping_patience": int(
                cfg.lightning.callbacks.early_stopping.patience
            ),
            "checkpoint_every_n_epochs": int(
                cfg.lightning.callbacks.checkpoint.every_n_epochs
            ),
        },
    }


def write_locked_result_manifest(trainer, cfg, datamodule, metrics):
    protocol = getattr(cfg, "tokenizer_eval_protocol", None)
    if protocol in (None, ""):
        return None

    tokenizer = datamodule.get_tokenizer()
    run_id = os.environ.get("EVALUATION_RUN_ID")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-p{os.getpid()}"
        os.environ["EVALUATION_RUN_ID"] = run_id
    branch = os.environ.get("EXPERIMENT_LABEL", "unknown-branch")
    checkpoint_label = os.environ.get("CHECKPOINT_LABEL", "unknown-checkpoint")
    task_label = os.environ.get("TASK_LABEL", cfg.experiment_name)

    non_finite_fields = []
    safe_metrics = _json_safe(metrics, non_finite_fields, "metrics")
    resolved_config = _json_safe(
        omegaconf.OmegaConf.to_container(cfg, resolve=True),
        non_finite_fields,
        "config",
    )
    repository_dir = Path(__file__).resolve().parents[3]
    best_probe_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    best_probe = (
        artifact_identity(best_probe_path)
        if best_probe_path and os.path.exists(best_probe_path)
        else None
    )
    checkpoint = file_identity(tokenizer.pretrained_ckpt_path)
    if checkpoint["sha256"] != tokenizer.checkpoint_sha256:
        raise RuntimeError("Checkpoint bytes changed during the evaluation run")
    encoder_config = (
        file_identity(tokenizer.encoder_cfg_path)
        if tokenizer.encoder_cfg_path is not None
        else None
    )
    if (
        encoder_config is not None
        and encoder_config["sha256"] != tokenizer.encoder_cfg_sha256
    ):
        raise RuntimeError("Encoder config bytes changed during the evaluation run")

    protocol_record = {
        "name": protocol,
        "encoder_source": tokenizer.encoder_source,
        "stb_selected_residue_filter_length": int(cfg.data.filter_length),
        "whole_chain_max_length_filter": None,
        "checkpoint_training_filters_applied": False,
        "cache_tag": tokenizer.evaluation_cache_tag,
        "single_class_present_sample_weight": 0.5,
        "classification_metric_aggregation": getattr(
            cfg.model, "classification_metric_aggregation", None
        ),
        "binary_class_weighting": getattr(
            cfg.model, "binary_class_weighting", None
        ),
        "unknown_residue_adapter": "zero-aa-identity-v1",
        # Preserve STB's own optimal-alignment adapter. In the canonical
        # BindInt cohort, 2fmd_A has two unresolved X positions; those two
        # labels are dropped by upstream alignment and are counted per split.
        "local_alignment_policy": "upstream-optimal-alignment-report-drops-v1",
    }
    cohorts = datamodule.evaluation_cohort_reports()
    expected_splits = {"train", *datamodule.all_split_names}
    if set(cohorts) != expected_splits:
        raise RuntimeError(
            f"Cannot publish an incomplete cohort manifest: expected "
            f"{sorted(expected_splits)}, got {sorted(cohorts)}"
        )
    for split, report in cohorts.items():
        conservation = report.get("row_conservation", {})
        if conservation.get("holds") is not True:
            raise RuntimeError(f"Row conservation is not proven for split {split!r}")
        for field in (
            "resolve_or_parse_failure_rows",
            "checkpoint_filter_exclusion_rows",
            "encoder_failure_rows",
        ):
            if int(report.get(field, -1)) != 0:
                raise RuntimeError(
                    f"Locked cohort split {split!r} has non-zero or missing {field}"
                )
        cache_artifact = report.get("token_cache_artifact", {})
        if not cache_artifact.get("sha256") or cache_artifact.get("size_bytes") is None:
            raise RuntimeError(
                f"Locked cohort split {split!r} lacks token-cache byte identity"
            )
    scientific_probe_config = _scientific_probe_config(cfg)
    scientific_versions = scientific_runtime_versions()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": protocol_record,
        "run": {
            "id": run_id,
            "started_at_utc": os.environ.get("EVALUATION_STARTED_AT_UTC"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_label": checkpoint_label,
            "experiment": branch,
            "task": task_label,
            "smoke": os.environ.get("STB_SMOKE", "0") == "1",
            **runtime_identity(),
        },
        "encoder": {
            **checkpoint,
            "checkpoint_type": tokenizer.checkpoint_type,
            "state_prefix": tokenizer.encoder_state_prefix,
            "latent_z_dim": tokenizer.latent_z_dim,
            "encoder_config": (
                encoder_config
            ),
            "checkpoint_training_selector_recorded_but_not_applied": (
                tokenizer.checkpoint_training_selector
            ),
        },
        "shared_embedding_cache": shared_embedding_cache_manifest_record(tokenizer),
        "probe": {
            "seed": int(cfg.optimization.seed),
            "micro_batch_size": int(cfg.optimization.micro_batch_size),
            "max_steps": int(cfg.max_steps),
            "precision": str(cfg.trainer.precision),
            "early_stopping_patience": int(
                cfg.lightning.callbacks.early_stopping.patience
            ),
            "checkpoint_every_n_validations": int(
                cfg.lightning.callbacks.checkpoint.every_n_epochs
            ),
            "resolved_config": resolved_config,
            "resolved_config_sha256": canonical_json_sha256(resolved_config),
            "scientific_config": scientific_probe_config,
            "scientific_config_sha256": canonical_json_sha256(
                scientific_probe_config
            ),
            "best_checkpoint": best_probe,
        },
        "cohorts": cohorts,
        "metrics": safe_metrics,
        "non_finite_fields_serialized_as_null": sorted(set(non_finite_fields)),
        "source_files": _source_file_identities(repository_dir),
        "git": git_identity(repository_dir),
        "scientific_runtime_versions": scientific_versions,
    }
    scientific_identity = {
        "protocol": protocol_record,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_type": tokenizer.checkpoint_type,
        "encoder_state_prefix": tokenizer.encoder_state_prefix,
        "latent_z_dim": tokenizer.latent_z_dim,
        "encoder_config_sha256": tokenizer.encoder_cfg_sha256,
        "probe_config_sha256": manifest["probe"]["scientific_config_sha256"],
        "scientific_runtime_versions": scientific_versions,
        "cohort_sha256_by_split": {
            split: report.get("cohort_sha256") for split, report in cohorts.items()
        },
        "token_cache_sha256_by_split": {
            split: report["token_cache_artifact"]["sha256"]
            for split, report in cohorts.items()
        },
        "metrics": safe_metrics,
    }
    manifest["scientific_identity_sha256"] = canonical_json_sha256(
        scientific_identity
    )

    log_root = Path(os.environ.get("STB_LOG_ROOT", cfg.trainer.default_root_dir))
    manifest_path = (
        log_root
        / "manifests"
        / safe_path_component(run_id)
        / (
            f"{safe_path_component(checkpoint_label)}__"
            f"{safe_path_component(branch)}__{safe_path_component(task_label)}.json"
        )
    )
    write_immutable_json(manifest_path, manifest)
    print(f"Immutable result manifest written to {manifest_path}")
    return str(manifest_path)


def print_and_record_results(trainer, cfg, datamodule):
    """Replaces Lightning's built-in results table (which -- because our
    metrics already have unique per-split names via add_dataloader_idx=False
    -- prints the exact same full merged metrics dict identically under each
    'DataLoader N' column) with a single table sourced from
    trainer.callback_metrics, and appends the fold/superfamily generalization
    metrics to a cross-run summary CSV.
    """
    metrics = {k: (v.item() if torch.is_tensor(v) else v) for k, v in trainer.callback_metrics.items()}

    branch = os.environ.get("EXPERIMENT_LABEL", "unknown-branch")
    checkpoint_label = os.environ.get("CHECKPOINT_LABEL", "unknown-checkpoint")
    task_label = os.environ.get("TASK_LABEL", cfg.experiment_name)

    table = Table(
        title=f"Branch: {branch}  |  Checkpoint: {checkpoint_label}  |  Task: {task_label}",
        expand=True,
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    highlighted = {}  # split -> (metric_name, value)
    for key in sorted(metrics.keys()):
        value = metrics[key]
        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)

        is_highlight = any(
            key == f"{split}_{metric}"
            for split in HIGHLIGHT_SPLITS for metric in HIGHLIGHT_METRICS
        )
        if is_highlight:
            table.add_row(Text(key, style="bold cyan"), Text(value_str, style="bold cyan"))
            for split in HIGHLIGHT_SPLITS:
                for metric in HIGHLIGHT_METRICS:
                    if key == f"{split}_{metric}":
                        highlighted[split] = (metric, value_str)
        else:
            table.add_row(key, value_str)

    console = Console(width=120, record=True)
    console.print(table)

    # The immutable publication record is authoritative and is created before
    # either mutable convenience artifact below. A collision therefore cannot
    # append a duplicate/ambiguous summary row first.
    write_locked_result_manifest(trainer, cfg, datamodule, metrics)

    # Full per-run results (every metric, not just the fold/superfamily
    # highlights the summary CSV keeps below) -- one file per (checkpoint,
    # branch, task), so results persist beyond scrolled-away terminal output.
    summary_file = os.environ.get("SUMMARY_FILE")
    if summary_file:
        results_dir = os.path.join(os.path.dirname(summary_file), "results")
        os.makedirs(results_dir, exist_ok=True)
        safe = safe_path_component
        protocol = getattr(cfg, "tokenizer_eval_protocol", None)
        run_suffix = (
            f"_{safe(os.environ.get('EVALUATION_RUN_ID', 'run'))}"
            if protocol not in (None, "")
            else ""
        )
        result_file = os.path.join(
            results_dir,
            f"{safe(checkpoint_label)}_{safe(branch)}_{safe(task_label)}{run_suffix}.txt",
        )
        with open(result_file, "x" if protocol not in (None, "") else "w") as f:
            f.write(console.export_text(styles=False))
        print(f"Full results written to {result_file}")

    if summary_file and ("fold_test" in highlighted or "superfamily_test" in highlighted):
        metric_name = (highlighted.get("fold_test") or highlighted.get("superfamily_test"))[0]
        fold_val = highlighted.get("fold_test", (metric_name, ""))[1]
        superfamily_val = highlighted.get("superfamily_test", (metric_name, ""))[1]

        write_header = not os.path.exists(summary_file)
        with open(summary_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["checkpoint", "branch", "task", "metric", "fold_test", "superfamily_test"])
            writer.writerow([checkpoint_label, branch, task_label, metric_name, fold_val, superfamily_val])

# local imports
exc_dir = os.path.dirname(os.path.dirname(__file__)) # "src/"
sys.path.append(exc_dir)
exc_dir_baseline = os.path.join(os.path.abspath(exc_dir), "baselines")
all_baseline_names = glob(exc_dir_baseline + "/*")
for name in all_baseline_names:
    if name != "cheap_proteins":
        sys.path.append(os.path.join(exc_dir_baseline, name))
sys.path.append(exc_dir_baseline)

import data_module
from shared_embedding_cache import (
    install_shared_embedding_cache,
    shared_embedding_cache_manifest_record,
)
from util import setup_loggings

install_shared_embedding_cache()

def setup_trainer(cfg):
    trainer_logger = hydra.utils.instantiate(cfg.lightning.logger)

    # strategy determins distributed training
    if cfg.deepspeed_path:
        strategy = hydra.utils.instantiate(cfg.lightning.strategy)
    else:
        strategy = "ddp" # distributed data parallel
    
    # callbacks
    callbacks = [
        hydra.utils.instantiate(cfg.lightning.callbacks.checkpoint),
        hydra.utils.instantiate(cfg.lightning.callbacks.early_stopping),
        hydra.utils.instantiate(cfg.lightning.callbacks.lr_monitor),
        hydra.utils.instantiate(cfg.lightning.callbacks.progress_bar)
    ]
    
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=[],
        strategy=strategy,
        logger=trainer_logger,
    )

    return trainer


@hydra.main(version_base=None, config_path="config")
def main(cfg):
    """
    Launch supervised fine-tuning using a hydra config, for protein classification
    """
    omegaconf.OmegaConf.resolve(cfg)
    validate_locked_protocol_config(cfg)

    # set up python loggings
    logger = setup_loggings(cfg)

    # forbidden restart model
    assert cfg.model.ckpt_path is None
    # set seed before initializing models
    pl.seed_everything(cfg.optimization.seed)

    # set up trainer
    trainer = setup_trainer(cfg)

    # set up data module
    cfg.data.multi_label = cfg.model.multi_label
    cfg.data.is_global_or_local = cfg.model.is_global_or_local
    if cfg.tokenizer == "WrappedOurPretrainedTokenizer":
        assert cfg.tokenizer_pretrained_ckpt_path is not None
        assert cfg.tokenizer_ckpt_name is not None
        tmp_cfg = omegaconf.OmegaConf.load(os.path.join(exc_dir, "./script/config/pretrain.yaml"))["model"]
        tmp_cfg.quantizer.freeze_codebook = True
        tmp_cfg.quantizer._need_init = False
        tmp_cfg.quantizer.use_linear_project = cfg.quantizer_use_linear_project
        tmp_cfg.encoder.d_model = cfg.model_encoder_dmodel
        tmp_cfg.encoder.n_layers = cfg.model_encoder_nlayers
        tmp_cfg.encoder.v_heads = cfg.model_encoder_vheads
        tmp_cfg.quantizer.codebook_size = cfg.quantizer_codebook_size 
        tmp_cfg.quantizer.codebook_embed_size = cfg.quantizer_codebook_embed_size
        tmp_cfg.encoder.d_out = cfg.model_encoder_dout

        pretrained_model_cfg = {
            "model_cfg": tmp_cfg,
            "pretrained_ckpt_path": cfg.tokenizer_pretrained_ckpt_path,
            "ckpt_name": cfg.tokenizer_ckpt_name,
        }

    elif cfg.tokenizer == "WrappedLaProteinaTokenizer":
        assert cfg.tokenizer_pretrained_ckpt_path is not None
        checkpoint_type = getattr(cfg, "tokenizer_checkpoint_type", "ae")
        encoder_cfg_path = getattr(cfg, "tokenizer_encoder_cfg_path", None)
        if checkpoint_type != "ae":
            assert encoder_cfg_path is not None, \
                "tokenizer_encoder_cfg_path is required when tokenizer_checkpoint_type != 'ae'"
        pretrained_model_cfg = {
            "pretrained_ckpt_path": cfg.tokenizer_pretrained_ckpt_path,
            "checkpoint_type": checkpoint_type,
            "encoder_cfg_path": encoder_cfg_path,
            "force_remove_non_standard_residues": getattr(cfg, "tokenizer_force_remove_non_standard_residues", None),
            "encoder_source": getattr(cfg, "tokenizer_encoder_source", "online"),
            "evaluation_max_length": getattr(cfg, "tokenizer_eval_max_length", None),
            "evaluation_cache_tag": getattr(cfg, "tokenizer_cache_tag", None),
            "evaluation_protocol": getattr(cfg, "tokenizer_eval_protocol", None),
        }
    else:
        pretrained_model_cfg = {}

    datamodule = data_module.ProteinDataModule(
        tokenizer_name=cfg.tokenizer,
        tokenizer_device=getattr(cfg, "tokenizer_device", "cpu"),
        seed=cfg.optimization.seed,
        micro_batch_size=cfg.optimization.micro_batch_size,
        data_args=cfg.data,
        py_logger=logger,
        test_only=getattr(cfg, "test_only", False),
        precompute_tokens=getattr(cfg, "precompute_tokens", False),
        tokenizer_kwargs=pretrained_model_cfg
    )
    datamodule.setup()
    
    if not cfg.data.use_continuous:
        # num_tokens is needed only when using single MLP classifier w/ LMs
        cfg.model.num_tokens = datamodule.get_tokenizer().get_num_tokens()
    cfg.model.use_sequence = cfg.data.use_sequence

    # set up module module
    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        model_cfg=cfg.model,
        trainer=trainer,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        all_split_names=datamodule.all_split_names,
        codebook_embedding=None if cfg.data.use_continuous else datamodule.get_codebook_embedding()
    )

    # training pipeline
    
    if not getattr(cfg, "validate_only", False) and not getattr(cfg, "test_only", False):
        logger.info("*********** start training ***********\n\n")
        
        trainer.fit(
            model=model, datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        logger.info(f"Saving final model weights to {cfg.save_dir_path}")
        trainer.save_checkpoint(cfg.save_dir_path, weights_only=True)
        logger.info("Finished saving final model")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        trainer.validate(
            model=model, datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
        print_and_record_results(trainer, cfg, datamodule)
    else:
        logger.info("*********** start validation ***********\n\n")
        trainer.validate(
            model=model, datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path,
            verbose=False,
        )
        print_and_record_results(trainer, cfg, datamodule)

if __name__ == "__main__":
    main()
