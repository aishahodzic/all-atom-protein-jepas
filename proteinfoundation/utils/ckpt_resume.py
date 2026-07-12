import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import torch
import yaml
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

REGISTRY_PATH = Path(__file__).parent / "ckpt_registry.yaml"


@rank_zero_only
def _log_info(msg):
    logger.info(msg)


@rank_zero_only
def _create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _lookup_hf_checkpoint(run_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Looks up (repo, filename, ema_filename) for run_name in ckpt_registry.yaml.
    Add a new experiment there, not here."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    entry = registry.get("runs", {}).get(run_name)
    if entry is None:
        return None, None, None
    repo = entry.get("hf_repo", registry.get("default_hf_repo"))
    return repo, entry.get("filename"), entry.get("ema_filename")


def resolve_resume_checkpoint(model, cfg_exp, ckpt_path_store) -> Optional[str]:
    """
    Always prompts for which checkpoint to resume from; never auto-selects one.
    Loads pretrain weights into `model` in-place for the "pretrain" choice. For
    "last"/"download" the returned path is passed to trainer.fit(ckpt_path=...),
    which restores full trainer state (optimizer, step count, RNG), not just weights.
    """
    from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt

    last_ckpt_name = fetch_last_ckpt(ckpt_path_store)
    last_ckpt_path = (
        os.path.join(ckpt_path_store, last_ckpt_name)
        if last_ckpt_name is not None
        else None
    )
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    resume_hf_repo, resume_hf_filename, resume_hf_ema_filename = _lookup_hf_checkpoint(
        cfg_exp.run_name_
    )

    _log_info(
        f"Last checkpoint found for run '{cfg_exp.run_name_}': {last_ckpt_path or 'none'}"
    )
    _log_info(f"pretrain_ckpt_path in config: {pretrain_ckpt_path or 'unset'}")

    options = ["scratch"]
    if last_ckpt_path is not None:
        options.append("last")
    if pretrain_ckpt_path is not None:
        options.append("pretrain")
    if last_ckpt_path is None and resume_hf_repo and resume_hf_filename:
        options.append("download")

    if "LOCAL_RANK" in os.environ:
        choice = os.environ["RESUME_CHOICE"]
    else:
        choice = input(f"Start training from checkpoint? [{'/'.join(options)}]: ").strip().lower()
        while choice not in options:
            choice = input(f"Please enter one of {options}: ").strip().lower()
        os.environ["RESUME_CHOICE"] = choice

    resume_ckpt_path = None
    if choice == "last":
        _log_info(f"Resuming from last checkpoint: {last_ckpt_path}")
        resume_ckpt_path = last_ckpt_path
    elif choice == "download":
        # Download only if LOCAL_RANK unset (spawned DDP ranks reuse the same file).
        resume_ckpt_path = os.path.join(ckpt_path_store, "last.ckpt")
        if "LOCAL_RANK" not in os.environ:
            _log_info(f"Downloading '{resume_hf_filename}' from '{resume_hf_repo}'")
            from getpass import getpass

            from huggingface_hub import hf_hub_download

            hf_token = os.environ.get("HF_TOKEN") or getpass("Please enter your HF read token: ")
            downloaded_path = hf_hub_download(
                repo_id=resume_hf_repo,
                filename=resume_hf_filename,
                repo_type="dataset",
                token=hf_token,
            )
            _create_dir(ckpt_path_store)
            shutil.copyfile(downloaded_path, resume_ckpt_path)

            if resume_hf_ema_filename:
                downloaded_ema_path = hf_hub_download(
                    repo_id=resume_hf_repo,
                    filename=resume_hf_ema_filename,
                    repo_type="dataset",
                    token=hf_token,
                )
                shutil.copyfile(
                    downloaded_ema_path,
                    os.path.join(ckpt_path_store, "last-EMA.ckpt"),
                )
        _log_info(f"Resuming from downloaded checkpoint: {resume_ckpt_path}")
    elif choice == "pretrain":
        _log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        _log_info("Training from scratch.")

    # If not resuming from `last`/`download` (which restore RNG/optimizer state), set seed
    if resume_ckpt_path is None:
        _log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    return resume_ckpt_path
