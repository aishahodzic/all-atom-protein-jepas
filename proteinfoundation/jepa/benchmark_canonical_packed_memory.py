"""Packed-view B=32 CUDA benchmark for canonical protein LeJEPA.

This benchmark is deliberately separate from the dense reference diagnostics.
Each invocation measures exactly one mode (global, local, or the complete
five-view objective) in a fresh process using the production canonical config,
a real training batch, and bf16 CUDA autocast.  It performs no optimizer step
and does not start a training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import hydra
from hydra import compose, initialize_config_dir
import lightning as L
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import OmegaConf, open_dict
import torch

from proteinfoundation.jepa.canonical_lejepa_model import (
    CanonicalLeJEPAModel,
    mean_pool_visible_residues,
    pack_batch_for_canonical_view,
    random_visible_residue_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def _memory_gib(device: torch.device) -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated(device) / GIB,
        torch.cuda.max_memory_reserved(device) / GIB,
    )


def _gradient_health(model: CanonicalLeJEPAModel) -> tuple[int, int, bool]:
    parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    gradients = [parameter.grad for parameter in parameters]
    present = [gradient for gradient in gradients if gradient is not None]
    finite = len(present) == len(parameters) and all(
        bool(torch.isfinite(gradient).all()) for gradient in present
    )
    return len(parameters), len(present), finite


def _print_view_statistics(
    *,
    label: str,
    visible_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    counts = visible_mask.sum(dim=-1)
    valid_counts = valid_mask.sum(dim=-1)
    fractions = counts.float() / valid_counts.clamp_min(1).float()
    print(f"{label}_packed_k={int(counts.max().item())}", flush=True)
    print(f"{label}_visible_count_min={int(counts.min().item())}", flush=True)
    print(f"{label}_visible_count_mean={float(counts.float().mean()):.6f}", flush=True)
    print(f"{label}_visible_count_max={int(counts.max().item())}", flush=True)
    print(f"{label}_visible_fraction_mean={float(fractions.mean()):.6f}", flush=True)


def _print_scalar_losses(output: dict[str, Any]) -> None:
    for name in ("loss", "inv_loss", "sigreg_loss"):
        value = output.get(name)
        if isinstance(value, torch.Tensor):
            print(f"{name}={float(value.detach().float()):.8f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("global", "local", "full"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    load_dotenv(REPO_ROOT / ".env")
    if not torch.cuda.is_available():
        raise RuntimeError("packed canonical benchmark requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("allocated GPU does not support bf16")

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(config_name="training_canonical_lejepa")
    with open_dict(cfg):
        # This is the only scientific-model execution override. The masks and
        # objective remain those of the production canonical config.
        cfg.views.packed_views = True
        cfg.dataset.datamodule.batch_size = int(args.batch_size)
        cfg.dataset.datamodule.num_workers = int(args.num_workers)
        cfg.dataset.datamodule.pin_memory = False
        cfg.wandb.enabled = False
    OmegaConf.resolve(cfg)
    L.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    actual_batch_size = int(batch["coords"].shape[0])
    if actual_batch_size != args.batch_size:
        raise RuntimeError(
            f"expected actual batch size {args.batch_size}, got {actual_batch_size}"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    batch = move_data_to_device(batch, device)
    model = CanonicalLeJEPAModel(cfg).to(device).train()
    if not model.packed_views:
        raise RuntimeError("packed benchmark constructed a dense canonical model")
    model.zero_grad(set_to_none=True)

    valid_mask = batch["mask_dict"]["coords"][..., 0, 0].bool()
    valid_counts = valid_mask.sum(dim=1)
    print(f"mode={args.mode}", flush=True)
    print("packed_views=true", flush=True)
    print(f"gpu_model={torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"gpu_total_gib={torch.cuda.get_device_properties(device).total_memory / GIB:.4f}",
        flush=True,
    )
    print("precision=bf16-mixed-equivalent CUDA autocast", flush=True)
    print("timing=cold_single_batch_cuda_events", flush=True)
    print(f"actual_batch_size={actual_batch_size}", flush=True)
    print(f"dense_batch_shape={tuple(batch['coords'].shape)}", flush=True)
    print(f"valid_length_min={int(valid_counts.min().item())}", flush=True)
    print(f"valid_length_mean={float(valid_counts.float().mean()):.6f}", flush=True)
    print(f"valid_length_max={int(valid_counts.max().item())}", flush=True)
    print(f"train_rows={len(datamodule.train_ds)}", flush=True)

    fixed_visible_mask = None
    if args.mode in ("global", "local"):
        masked_pct = 0.20 if args.mode == "global" else 0.80
        fixed_visible_mask = random_visible_residue_mask(
            valid_mask, masked_pct=masked_pct
        )
        print(f"masked_pct={masked_pct:.2f}", flush=True)
        _print_view_statistics(
            label=args.mode,
            visible_mask=fixed_visible_mask,
            valid_mask=valid_mask,
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

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    phase = "forward"
    try:
        torch.cuda.synchronize(device)
        forward_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.mode == "full":
                output = model.compute_loss(batch)
                loss = output["loss"]
            else:
                assert fixed_visible_mask is not None
                packed_view = pack_batch_for_canonical_view(
                    batch, fixed_visible_mask
                )
                residue_latents = model.encoder(packed_view.batch)["mean"]
                pooled = mean_pool_visible_residues(
                    residue_latents, packed_view.valid_mask
                )
                loss = pooled.square().mean()
                output = {"loss": loss}
        forward_end.record()
        forward_end.synchronize()
        forward_ms = forward_start.elapsed_time(forward_end)
        forward_allocated, forward_reserved = _memory_gib(device)
        print(f"forward_runtime_ms={forward_ms:.3f}", flush=True)
        print(f"forward_peak_allocated_gib={forward_allocated:.4f}", flush=True)
        print(f"forward_peak_reserved_gib={forward_reserved:.4f}", flush=True)
        print("forward_success=true", flush=True)

        if args.mode == "full":
            view_masks = output["view_masks"]
            for view_index, label in enumerate(
                ("global", "local_1", "local_2", "local_3", "local_4")
            ):
                _print_view_statistics(
                    label=label,
                    visible_mask=view_masks[:, view_index],
                    valid_mask=valid_mask,
                )

        scalar_losses = {
            name: value
            for name, value in output.items()
            if name in ("loss", "inv_loss", "sigreg_loss")
            and isinstance(value, torch.Tensor)
        }
        if not scalar_losses or not all(
            bool(torch.isfinite(value.detach())) for value in scalar_losses.values()
        ):
            raise RuntimeError("packed canonical benchmark produced a non-finite loss")
        _print_scalar_losses(output)

        phase = "backward"
        torch.cuda.synchronize(device)
        backward_start.record()
        loss.backward()
        backward_end.record()
        backward_end.synchronize()
        backward_ms = backward_start.elapsed_time(backward_end)
        total_allocated, total_reserved = _memory_gib(device)
        print(f"backward_runtime_ms={backward_ms:.3f}", flush=True)
        print(f"forward_backward_runtime_ms={forward_ms + backward_ms:.3f}", flush=True)
        print(
            f"forward_backward_peak_allocated_gib={total_allocated:.4f}",
            flush=True,
        )
        print(
            f"forward_backward_peak_reserved_gib={total_reserved:.4f}",
            flush=True,
        )

        parameter_count, gradient_count, gradients_finite = _gradient_health(model)
        print(f"encoder_trainable_parameter_tensors={parameter_count}", flush=True)
        print(f"encoder_parameter_tensors_with_grad={gradient_count}", flush=True)
        print(f"encoder_gradients_finite={str(gradients_finite).lower()}", flush=True)
        if not gradients_finite:
            raise RuntimeError("packed benchmark produced missing or non-finite gradients")
        print("backward_success=true", flush=True)
    except torch.cuda.OutOfMemoryError:
        allocated, reserved = _memory_gib(device)
        print(f"oom_phase={phase}", flush=True)
        print(f"oom_peak_allocated_gib={allocated:.4f}", flush=True)
        print(f"oom_peak_reserved_gib={reserved:.4f}", flush=True)
        raise


if __name__ == "__main__":
    main()
