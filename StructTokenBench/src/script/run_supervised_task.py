import csv
import os
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

torch.set_float32_matmul_precision("high")  # A30 has Tensor Cores; default "highest" leaves them unused

# Metrics we care most about for generalization (held-out CATH folds/superfamilies,
# not just i.i.d. validation): AUROC for classification tasks (bind/cat/con/rep/ept),
# spearmanr for regression tasks (flex*). Highlighted in the results table and
# pulled into the cross-run summary; absent (and skipped) for tasks with different
# split names (BindShake, Homo).
HIGHLIGHT_METRICS = ("auroc", "spearmanr")
HIGHLIGHT_SPLITS = ("fold_test", "superfamily_test")


def print_and_record_results(trainer, cfg):
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

    # Full per-run results (every metric, not just the fold/superfamily
    # highlights the summary CSV keeps below) -- one file per (checkpoint,
    # branch, task), so results persist beyond scrolled-away terminal output.
    summary_file = os.environ.get("SUMMARY_FILE")
    if summary_file:
        results_dir = os.path.join(os.path.dirname(summary_file), "results")
        os.makedirs(results_dir, exist_ok=True)
        safe = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
        result_file = os.path.join(
            results_dir, f"{safe(checkpoint_label)}_{safe(branch)}_{safe(task_label)}.txt")
        with open(result_file, "w") as f:
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
from util import setup_loggings

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
        print_and_record_results(trainer, cfg)
    else:
        logger.info("*********** start validation ***********\n\n")
        trainer.validate(
            model=model, datamodule=datamodule,
            ckpt_path=cfg.model.ckpt_path,
            verbose=False,
        )
        print_and_record_results(trainer, cfg)

if __name__ == "__main__":
    main()