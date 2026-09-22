"""Two-GPU global-B16 test for protein-centred residue SIGReg.

Run with ``torchrun --standalone --nproc-per-node=2``. The preprocessing of
each local protein is rank-local; CanonicalSIGReg then synchronizes its random
projection bank and aggregates the two B=8 empirical characteristic functions.
"""

from __future__ import annotations

import os
from unittest import mock

import torch
import torch.distributed as dist

from proteinfoundation.jepa.canonical_lejepa_model import (
    protein_centered_residue_sigreg_input,
)
from proteinfoundation.jepa.canonical_sigreg import CanonicalSIGReg


KNOTS = 17
NUM_PROJ = 1024
VIEWS = 5
GLOBAL_BATCH = 16
LOCAL_BATCH = 8
LENGTH = 7
INPUT_DIM = 6
LATENT_DIM = 8


def _direct_sigreg_reference(
    embeddings: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    t = torch.linspace(0, 3, KNOTS, device=embeddings.device)
    spacing = 3 / (KNOTS - 1)
    quadrature = torch.full((KNOTS,), 2 * spacing, device=embeddings.device)
    quadrature[[0, -1]] = spacing
    gaussian_cf = torch.exp(-t.square() / 2)
    phase = (embeddings @ directions).unsqueeze(-1) * t
    cosine_mean = phase.cos().mean(dim=1)
    sine_mean = phase.sin().mean(dim=1)
    error = (cosine_mean - gaussian_cf).square() + sine_mean.square()
    return ((error @ (quadrature * gaussian_cf)) * GLOBAL_BATCH).mean()


def _global_mask_and_indices() -> tuple[torch.Tensor, torch.Tensor]:
    patterns = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    real_mask = patterns.repeat(GLOBAL_BATCH // len(patterns), 1)
    sampled_indices = torch.tensor([6, 3, 4, 1] * (GLOBAL_BATCH // 4))
    if not real_mask.gather(1, sampled_indices[:, None]).all():
        raise AssertionError("test fixture selected a padded residue")
    return real_mask, sampled_indices


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL centered-residue SIGReg test requires CUDA")
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"expected two NCCL ranks, got {world_size}")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        generator = torch.Generator().manual_seed(3401)
        global_inputs = torch.randn(
            GLOBAL_BATCH,
            VIEWS,
            LENGTH,
            INPUT_DIM,
            generator=generator,
        )
        initial_weight = torch.randn(
            INPUT_DIM,
            LATENT_DIM,
            generator=torch.Generator().manual_seed(991),
        )
        global_mask, global_indices = _global_mask_and_indices()
        local_inputs = global_inputs.chunk(world_size, dim=0)[rank].to(device)
        local_mask = global_mask.chunk(world_size, dim=0)[rank].to(device)
        local_indices = global_indices.chunk(world_size, dim=0)[rank].to(device)

        parameter = initial_weight.to(device).requires_grad_(True)
        local_residues = local_inputs @ parameter
        local_sigreg_input, returned_indices = (
            protein_centered_residue_sigreg_input(
                local_residues,
                local_mask,
                sampled_indices=local_indices,
            )
        )
        if not torch.equal(returned_indices, local_indices):
            raise AssertionError("fixed local sampled indices changed")
        if tuple(local_sigreg_input.shape) != (VIEWS, LOCAL_BATCH, LATENT_DIM):
            raise AssertionError(f"wrong local input shape: {local_sigreg_input.shape}")

        torch.cuda.manual_seed(50_000 + rank)
        original_randn = torch.randn
        captured: dict[str, torch.Tensor] = {}

        def capture_directions(*shape, **kwargs):
            sampled = original_randn(*shape, **kwargs)
            if tuple(shape) == (LATENT_DIM, NUM_PROJ):
                captured["raw"] = sampled.detach().clone()
                captured["live"] = sampled
            return sampled

        sigreg = CanonicalSIGReg(
            knots=KNOTS,
            num_proj=NUM_PROJ,
            normalize_by_n=False,
        ).to(device)
        with mock.patch(
            "proteinfoundation.jepa.canonical_sigreg.torch.randn",
            side_effect=capture_directions,
        ):
            distributed_loss = sigreg(local_sigreg_input)

        raw_bank = captured["raw"]
        synchronized_bank = captured["live"].detach().clone()
        raw_banks = [torch.empty_like(raw_bank) for _ in range(world_size)]
        synchronized_banks = [
            torch.empty_like(synchronized_bank) for _ in range(world_size)
        ]
        dist.all_gather(raw_banks, raw_bank)
        dist.all_gather(synchronized_banks, synchronized_bank)
        if torch.equal(raw_banks[0], raw_banks[1]):
            raise AssertionError("rank RNG states unexpectedly sampled equal banks")
        if not torch.equal(synchronized_banks[0], synchronized_banks[1]):
            raise AssertionError("projection banks differ after rank-0 broadcast")

        direct_parameter = initial_weight.to(device).requires_grad_(True)
        global_residues = global_inputs.to(device) @ direct_parameter
        global_sigreg_input, _ = protein_centered_residue_sigreg_input(
            global_residues,
            global_mask.to(device),
            sampled_indices=global_indices.to(device),
        )
        reference_loss = _direct_sigreg_reference(
            global_sigreg_input,
            synchronized_banks[0],
        )
        torch.testing.assert_close(
            distributed_loss,
            reference_loss,
            rtol=2e-5,
            atol=2e-5,
        )

        distributed_loss.backward()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError("local centered-residue gradient missing/non-finite")
        ddp_averaged_gradient = parameter.grad.detach().clone()
        dist.all_reduce(ddp_averaged_gradient, op=dist.ReduceOp.AVG)

        reference_loss.backward()
        if direct_parameter.grad is None:
            raise AssertionError("direct global reference gradient is missing")
        torch.testing.assert_close(
            ddp_averaged_gradient,
            direct_parameter.grad,
            rtol=3e-5,
            atol=3e-5,
        )

        if rank == 0:
            print("backend=nccl", flush=True)
            print("world_size=2", flush=True)
            print(f"gpu_model={torch.cuda.get_device_name(device)}", flush=True)
            print("local_batch=8", flush=True)
            print("global_batch=16", flush=True)
            print("sigreg_input=protein_centered_residue", flush=True)
            print("same_residue_row_across_views=true", flush=True)
            print("projection_bank_sync=true", flush=True)
            print(f"distributed_loss={float(distributed_loss):.8f}", flush=True)
            print(f"reference_loss={float(reference_loss):.8f}", flush=True)
            print("forward_parity=true", flush=True)
            print("ddp_averaged_gradient_parity=true", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
