"""One-batch GPU memory smoke for projector-free canonical LeJEPA V1.

This intentionally performs no optimizer step and starts no training run.  It
loads one real batch of 32 proteins, executes the five-view loss under CUDA
bf16 autocast, and backpropagates once while reporting peak device memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import hydra
from hydra import compose, initialize_config_dir
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import OmegaConf, open_dict
import torch

from proteinfoundation.jepa.canonical_lejepa_model import CanonicalLeJEPAModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def _finite_encoder_gradients(model: CanonicalLeJEPAModel) -> tuple[int, bool]:
    gradients = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    return len(gradients), bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not torch.cuda.is_available():
        raise RuntimeError("the canonical memory smoke requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated GPU does not support bf16")

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(config_name="training_canonical_lejepa")
    with open_dict(cfg):
        cfg.dataset.datamodule.num_workers = int(args.num_workers)
        cfg.dataset.datamodule.pin_memory = False
        cfg.wandb.enabled = False
    OmegaConf.resolve(cfg)

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    actual_batch_size = int(batch["coords"].shape[0])
    if actual_batch_size != 32:
        raise RuntimeError(
            f"canonical smoke requires actual batch size 32, got {actual_batch_size}"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    batch = move_data_to_device(batch, device)
    model = CanonicalLeJEPAModel(cfg).to(device).train()
    model.zero_grad(set_to_none=True)

    print(f"gpu_model={torch.cuda.get_device_name(device)}", flush=True)
    print(f"gpu_total_gib={torch.cuda.get_device_properties(device).total_memory / 1024**3:.4f}", flush=True)
    print(f"actual_batch_size={actual_batch_size}", flush=True)
    print("num_views=5", flush=True)
    print("precision=bf16-mixed-equivalent CUDA autocast", flush=True)
    print(f"batch_shape={tuple(batch['coords'].shape)}", flush=True)
    print(f"train_rows={len(datamodule.train_ds)}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.compute_loss(batch)
    if not all(
        bool(torch.isfinite(output[name]))
        for name in ("loss", "inv_loss", "sigreg_loss")
    ):
        raise RuntimeError("canonical smoke produced a non-finite loss")
    output["loss"].backward()
    torch.cuda.synchronize(device)

    gradient_count, gradients_finite = _finite_encoder_gradients(model)
    if not gradients_finite:
        raise RuntimeError("canonical smoke produced missing or non-finite gradients")

    gib = 1024**3
    print(f"encoder_parameters_with_grad={gradient_count}", flush=True)
    print(f"loss={float(output['loss'].detach().float()):.8f}", flush=True)
    print(f"inv_loss={float(output['inv_loss'].detach().float()):.8f}", flush=True)
    print(
        f"sigreg_loss={float(output['sigreg_loss'].detach().float()):.8f}",
        flush=True,
    )
    print(
        f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / gib:.4f}",
        flush=True,
    )
    print(
        f"peak_reserved_gib={torch.cuda.max_memory_reserved(device) / gib:.4f}",
        flush=True,
    )
    print("forward_success=true", flush=True)
    print("backward_success=true", flush=True)


if __name__ == "__main__":
    main()
