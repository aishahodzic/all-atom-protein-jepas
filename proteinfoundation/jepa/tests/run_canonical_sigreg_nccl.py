"""Two-GPU NCCL verification of canonical SIGReg synchronization and AVG.

Run with ``torchrun --standalone --nproc-per-node=2``.  This is deliberately a
separate GPU integration test: the ordinary unit suite stays CPU-capable, and
the scientific implementation continues to use the official ``ReduceOp.AVG``.
The ranks start from different RNG states; rank 0's projection bank is then
broadcast before the global ECF is formed.
"""

from __future__ import annotations

import os
from unittest import mock

import torch
import torch.distributed as dist

from proteinfoundation.jepa.canonical_sigreg import CanonicalSIGReg


KNOTS = 17
NUM_PROJ = 1024
VIEWS = 5
GLOBAL_BATCH = 16
LATENT_DIM = 8
INPUT_DIM = 6


def _direct_global_reference(
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
    statistic = error @ (quadrature * gaussian_cf)
    return (statistic * GLOBAL_BATCH).mean()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL SIGReg integration test requires CUDA")
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"expected exactly two NCCL ranks, got {world_size}")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        global_inputs = torch.randn(
            VIEWS,
            GLOBAL_BATCH,
            INPUT_DIM,
            generator=torch.Generator().manual_seed(18),
        )
        initial_weight = torch.randn(
            INPUT_DIM,
            LATENT_DIM,
            generator=torch.Generator().manual_seed(212),
        )
        local_inputs = global_inputs.chunk(world_size, dim=1)[rank].to(device)
        parameter = initial_weight.to(device).requires_grad_(True)
        local_embeddings = local_inputs @ parameter

        torch.cuda.manual_seed(10_000 + rank)
        original_randn = torch.randn
        captured = {}

        def capture_directions(*shape, **kwargs):
            sampled = original_randn(*shape, **kwargs)
            if tuple(shape) == (LATENT_DIM, NUM_PROJ):
                captured["raw"] = sampled.detach().clone()
                captured["synchronized"] = sampled
            return sampled

        sigreg = CanonicalSIGReg(
            knots=KNOTS, num_proj=NUM_PROJ, normalize_by_n=False
        ).to(device)
        with mock.patch(
            "proteinfoundation.jepa.canonical_sigreg.torch.randn",
            side_effect=capture_directions,
        ):
            distributed_loss = sigreg(local_embeddings)

        raw_bank = captured["raw"]
        synchronized_bank = captured["synchronized"].detach().clone()
        raw_banks = [torch.empty_like(raw_bank) for _ in range(world_size)]
        synchronized_banks = [
            torch.empty_like(synchronized_bank) for _ in range(world_size)
        ]
        dist.all_gather(raw_banks, raw_bank)
        dist.all_gather(synchronized_banks, synchronized_bank)
        if torch.equal(raw_banks[0], raw_banks[1]):
            raise AssertionError("rank RNG states unexpectedly sampled identical banks")
        if not torch.equal(synchronized_banks[0], synchronized_banks[1]):
            raise AssertionError("SIGReg projection banks differ after rank-0 broadcast")

        direct_parameter = initial_weight.to(device).requires_grad_(True)
        expected = _direct_global_reference(
            global_inputs.to(device) @ direct_parameter,
            synchronized_banks[0],
        )
        torch.testing.assert_close(distributed_loss, expected, rtol=2e-5, atol=2e-5)

        distributed_loss.backward()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError("distributed SIGReg parameter gradient is missing/non-finite")
        ddp_averaged_gradient = parameter.grad.detach().clone()
        dist.all_reduce(ddp_averaged_gradient, op=dist.ReduceOp.AVG)

        expected.backward()
        if direct_parameter.grad is None:
            raise AssertionError("direct-reference parameter gradient is missing")
        torch.testing.assert_close(
            ddp_averaged_gradient,
            direct_parameter.grad,
            rtol=3e-5,
            atol=3e-5,
        )

        rank_losses = [torch.zeros_like(distributed_loss) for _ in range(world_size)]
        dist.all_gather(rank_losses, distributed_loss.detach())
        for other_loss in rank_losses:
            torch.testing.assert_close(other_loss, distributed_loss)

        if rank == 0:
            print("backend=nccl", flush=True)
            print(f"world_size={world_size}", flush=True)
            print(f"gpu_model={torch.cuda.get_device_name(device)}", flush=True)
            print(f"local_batch={GLOBAL_BATCH // world_size}", flush=True)
            print(f"global_batch={GLOBAL_BATCH}", flush=True)
            print(f"projection_bank={LATENT_DIM}x{NUM_PROJ}", flush=True)
            print("rank_rng_states=different", flush=True)
            print("projection_bank_sync=true", flush=True)
            print("collective=ReduceOp.AVG", flush=True)
            print(f"distributed_loss={float(distributed_loss):.8f}", flush=True)
            print(f"reference_loss={float(expected):.8f}", flush=True)
            print("forward_parity=true", flush=True)
            print("ddp_averaged_parameter_gradient_parity=true", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
