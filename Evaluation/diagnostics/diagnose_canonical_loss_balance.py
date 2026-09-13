"""Measure canonical invariance/SIGReg gradient balance on a fixed DDP batch."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from proteinfoundation.jepa.canonical_lejepa_model import CanonicalLeJEPAModel


class _LossAdapter(nn.Module):
    def __init__(self, model: CanonicalLeJEPAModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.model.compute_loss(batch)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _component_backward(
    ddp_model: DistributedDataParallel,
    batch: dict[str, Any],
    *,
    seed: int,
    component: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    ddp_model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = ddp_model(batch)
        if component == "invariance":
            loss = output["inv_loss"]
        elif component == "weighted_sigreg":
            loss = ddp_model.module.model.sigreg_weight * output["sigreg_loss"]
        else:
            raise ValueError(component)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in ddp_model.module.model.encoder.named_parameters()
        if parameter.grad is not None
    }
    detached = {
        key: value.detach().float()
        for key, value in output.items()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    }
    detached["embeddings"] = output["embeddings"].detach().float()
    return gradients, detached


def _effective_rank(values: torch.Tensor) -> float:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()

    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    cfg = OmegaConf.create(json.loads(args.config.read_text(encoding="utf-8")))
    cfg.dataset.datamodule.num_workers = 4
    cfg.dataset.datamodule.pin_memory = True
    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.setup("fit")
    batch = _to_device(next(iter(datamodule.train_dataloader())), device)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if int(checkpoint["global_step"]) != 10_000:
        raise RuntimeError(
            f"expected step 10000, got {checkpoint['global_step']}"
        )
    model = CanonicalLeJEPAModel(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).train()
    ddp_model = DistributedDataParallel(
        _LossAdapter(model), device_ids=[local_rank], broadcast_buffers=False
    )

    inv_gradients, inv_output = _component_backward(
        ddp_model, batch, seed=args.seed, component="invariance"
    )
    sig_gradients, sig_output = _component_backward(
        ddp_model, batch, seed=args.seed, component="weighted_sigreg"
    )
    if inv_gradients.keys() != sig_gradients.keys():
        raise RuntimeError("loss components reached different encoder parameters")

    inv_sq = 0.0
    sig_sq = 0.0
    dot = 0.0
    combined_sq = 0.0
    for name in inv_gradients:
        inv_gradient = inv_gradients[name].double()
        sig_gradient = sig_gradients[name].double()
        inv_sq += float(inv_gradient.square().sum())
        sig_sq += float(sig_gradient.square().sum())
        dot += float((inv_gradient * sig_gradient).sum())
        combined_sq += float((inv_gradient + sig_gradient).square().sum())
    inv_norm = math.sqrt(inv_sq)
    sig_norm = math.sqrt(sig_sq)

    embeddings = inv_output.pop("embeddings")
    gathered = [torch.empty_like(embeddings) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, embeddings)
    all_embeddings = torch.cat(gathered, dim=0).cpu()

    scalar_names = ("loss", "inv_loss", "sigreg_loss")
    scalar_values = torch.stack([inv_output[name] for name in scalar_names])
    dist.all_reduce(scalar_values, op=dist.ReduceOp.AVG)

    if rank == 0:
        per_view_rank = [
            _effective_rank(all_embeddings[:, view]) for view in range(5)
        ]
        per_local_mse = [
            float(
                (all_embeddings[:, 0] - all_embeddings[:, view])
                .square()
                .mean()
            )
            for view in range(1, 5)
        ]
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "global_step": int(checkpoint["global_step"]),
            "global_batch_size": int(all_embeddings.shape[0]),
            "latent_dim": int(all_embeddings.shape[-1]),
            "loss": float(scalar_values[0]),
            "invariance_loss": float(scalar_values[1]),
            "sigreg_loss_unweighted": float(scalar_values[2]),
            "sigreg_weight": float(model.sigreg_weight),
            "sigreg_loss_weighted": float(model.sigreg_weight * scalar_values[2]),
            "invariance_gradient_norm": inv_norm,
            "weighted_sigreg_gradient_norm": sig_norm,
            "sigreg_to_invariance_gradient_ratio": sig_norm / max(inv_norm, 1e-30),
            "gradient_cosine": dot / max(inv_norm * sig_norm, 1e-30),
            "combined_gradient_norm": math.sqrt(combined_sq),
            "per_view_effective_rank": per_view_rank,
            "mean_effective_rank": sum(per_view_rank) / len(per_view_rank),
            "global_local_mse_by_local": per_local_mse,
            "mean_global_local_mse": sum(per_local_mse) / len(per_local_mse),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
