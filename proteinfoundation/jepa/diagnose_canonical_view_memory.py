"""Measure one canonical masked view at the required actual batch size of 32.

This diagnostic does not alter the canonical model or loss implementation.  It
isolates one encoder view and reports CUDA memory after its forward and after a
minimal scalar backward through that pooled view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import hydra
from hydra import compose, initialize_config_dir
import lightning as L
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import OmegaConf, open_dict
import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CanonicalLeJEPAModel,
    mask_batch_for_canonical_view,
    mean_pool_visible_residues,
    random_visible_residue_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def _memory(device: torch.device) -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated(device) / GIB,
        torch.cuda.max_memory_reserved(device) / GIB,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("global", "local"), required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    masked_pct = 0.20 if args.view == "global" else 0.80

    load_dotenv(REPO_ROOT / ".env")
    if not torch.cuda.is_available():
        raise RuntimeError("canonical view-memory diagnostic requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("allocated GPU does not support bf16")

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(config_name="training_canonical_lejepa")
    with open_dict(cfg):
        cfg.dataset.datamodule.num_workers = int(args.num_workers)
        cfg.dataset.datamodule.pin_memory = False
        cfg.wandb.enabled = False
    OmegaConf.resolve(cfg)
    L.seed_everything(int(cfg.seed), workers=True)

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    actual_batch_size = int(batch["coords"].shape[0])
    if actual_batch_size != 32:
        raise RuntimeError(f"expected actual batch size 32, got {actual_batch_size}")

    device = torch.device("cuda", torch.cuda.current_device())
    batch = move_data_to_device(batch, device)
    model = CanonicalLeJEPAModel(cfg).to(device).train()
    model.zero_grad(set_to_none=True)

    valid = batch["mask_dict"]["coords"][..., 0, 0].bool()
    visible = random_visible_residue_mask(valid, masked_pct=masked_pct)
    masked_batch = mask_batch_for_canonical_view(batch, visible)

    print(f"view={args.view}", flush=True)
    print(f"masked_pct={masked_pct:.2f}", flush=True)
    print(f"gpu_model={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"gpu_total_gib={torch.cuda.get_device_properties(device).total_memory / GIB:.4f}",
        flush=True,
    )
    print(f"actual_batch_size={actual_batch_size}", flush=True)
    print("precision=bf16-mixed-equivalent CUDA autocast", flush=True)
    print(f"batch_shape={tuple(batch['coords'].shape)}", flush=True)
    print(
        f"mean_visible_fraction={float((visible.sum(1) / valid.sum(1)).mean()):.6f}",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"baseline_allocated_gib={torch.cuda.memory_allocated(device) / GIB:.4f}",
        flush=True,
    )
    print(
        f"baseline_reserved_gib={torch.cuda.memory_reserved(device) / GIB:.4f}",
        flush=True,
    )

    phase = "forward"
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            residue_latents = model.encoder(masked_batch)["mean"]
            pooled = mean_pool_visible_residues(residue_latents, visible)
            diagnostic_loss = pooled.square().mean()
        torch.cuda.synchronize(device)
        forward_allocated, forward_reserved = _memory(device)
        print(f"forward_peak_allocated_gib={forward_allocated:.4f}", flush=True)
        print(f"forward_peak_reserved_gib={forward_reserved:.4f}", flush=True)
        print("forward_success=true", flush=True)

        phase = "backward"
        diagnostic_loss.backward()
        torch.cuda.synchronize(device)
        total_allocated, total_reserved = _memory(device)
        print(
            f"forward_backward_peak_allocated_gib={total_allocated:.4f}",
            flush=True,
        )
        print(
            f"forward_backward_peak_reserved_gib={total_reserved:.4f}",
            flush=True,
        )
        print("backward_success=true", flush=True)
    except torch.cuda.OutOfMemoryError:
        allocated, reserved = _memory(device)
        print(f"oom_phase={phase}", flush=True)
        print(f"oom_peak_allocated_gib={allocated:.4f}", flush=True)
        print(f"oom_peak_reserved_gib={reserved:.4f}", flush=True)
        raise


if __name__ == "__main__":
    main()
