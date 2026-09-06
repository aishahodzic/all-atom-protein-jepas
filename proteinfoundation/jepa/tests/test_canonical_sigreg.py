"""Independent equation and distributed tests for canonical SIGReg."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from proteinfoundation.jepa.canonical_sigreg import CanonicalSIGReg


def _direct_reference(
    embeddings: torch.Tensor,
    *,
    knots: int,
    num_proj: int,
    normalize_by_n: bool,
) -> torch.Tensor:
    directions = torch.randn(embeddings.shape[-1], num_proj)
    directions = directions / torch.linalg.vector_norm(directions, dim=0)
    t = torch.linspace(0, 3, knots)
    spacing = 3 / (knots - 1)
    quadrature = torch.full((knots,), 2 * spacing)
    quadrature[[0, -1]] = spacing
    gaussian_cf = torch.exp(-t.square() / 2)
    weighted_quadrature = quadrature * gaussian_cf

    projected = embeddings @ directions
    phase = projected.unsqueeze(-1) * t
    cosine_mean = phase.cos().mean(dim=1)
    sine_mean = phase.sin().mean(dim=1)
    error = (cosine_mean - gaussian_cf).square() + sine_mean.square()
    statistic = error @ weighted_quadrature
    if not normalize_by_n:
        statistic = statistic * embeddings.shape[1]
    return statistic.mean()


def _distributed_worker(rank, world_size, init_file, embeddings, queue):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    try:
        torch.manual_seed(991)
        shard = embeddings.chunk(world_size, dim=1)[rank]
        loss = CanonicalSIGReg(knots=7, num_proj=11, normalize_by_n=False)(shard)
        if rank == 0:
            queue.put(float(loss))
    finally:
        dist.destroy_process_group()


class CanonicalSIGRegTest(unittest.TestCase):
    def test_exact_independent_equation_reference(self):
        embeddings = torch.randn(5, 6, 8, generator=torch.Generator().manual_seed(3))
        torch.manual_seed(41)
        expected = _direct_reference(
            embeddings, knots=7, num_proj=13, normalize_by_n=False
        )
        torch.manual_seed(41)
        actual = CanonicalSIGReg(knots=7, num_proj=13, normalize_by_n=False)(
            embeddings
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_n_scaling_and_protein_permutation(self):
        embeddings = torch.randn(5, 8, 4, generator=torch.Generator().manual_seed(4))
        torch.manual_seed(55)
        normalized = CanonicalSIGReg(5, 9, normalize_by_n=True)(embeddings)
        torch.manual_seed(55)
        scaled = CanonicalSIGReg(5, 9, normalize_by_n=False)(embeddings)
        self.assertTrue(torch.allclose(scaled, normalized * 8, atol=1e-6))

        permutation = torch.randperm(8, generator=torch.Generator().manual_seed(9))
        torch.manual_seed(55)
        permuted = CanonicalSIGReg(5, 9, normalize_by_n=False)(
            embeddings[:, permutation]
        )
        self.assertTrue(torch.allclose(scaled, permuted, atol=1e-6))

    def test_distributed_collective_is_avg_and_uses_global_n(self):
        embeddings = torch.randn(5, 4, 4, generator=torch.Generator().manual_seed(18))
        all_reduce_calls = []
        broadcast_calls = []

        def identical_rank_average(tensor, op):
            all_reduce_calls.append(op)
            # Simulate a second rank with an identical ECF: AVG leaves it unchanged.
            return tensor

        def rank_zero_projection_bank(tensor, src):
            broadcast_calls.append((tensor.shape, src))
            return tensor

        torch.manual_seed(212)
        with (
            mock.patch("proteinfoundation.jepa.canonical_sigreg.dist.is_available", return_value=True),
            mock.patch("proteinfoundation.jepa.canonical_sigreg.dist.is_initialized", return_value=True),
            mock.patch("proteinfoundation.jepa.canonical_sigreg.dist.get_world_size", return_value=2),
            mock.patch(
                "proteinfoundation.jepa.canonical_sigreg.dist.broadcast",
                side_effect=rank_zero_projection_bank,
            ),
            mock.patch(
                "proteinfoundation.jepa.canonical_sigreg.dist.all_reduce",
                side_effect=identical_rank_average,
            ),
        ):
            distributed = CanonicalSIGReg(7, 11, normalize_by_n=False)(embeddings)

        torch.manual_seed(212)
        per_sample = CanonicalSIGReg(7, 11, normalize_by_n=True)(embeddings)
        self.assertEqual(broadcast_calls, [(torch.Size([4, 11]), 0)])
        self.assertEqual(all_reduce_calls, [dist.ReduceOp.AVG])
        self.assertTrue(
            torch.allclose(distributed, per_sample * embeddings.shape[1] * 2)
        )

    @unittest.skip(
        "PyTorch Gloo rejects ReduceOp.AVG; exact LeVJEPA DDP behavior requires "
        "an AVG-capable backend (e.g. NCCL) and is covered above with a mocked collective"
    )
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "gloo unavailable")
    def test_two_process_global_ecf_matches_single_process(self):
        embeddings = torch.randn(5, 8, 4, generator=torch.Generator().manual_seed(8))
        torch.manual_seed(991)
        expected = CanonicalSIGReg(knots=7, num_proj=11, normalize_by_n=False)(
            embeddings
        )
        context = mp.get_context("spawn")
        queue = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "gloo-init")
            mp.spawn(
                _distributed_worker,
                args=(2, init_file, embeddings, queue),
                nprocs=2,
                join=True,
            )
            actual = queue.get(timeout=10)
        self.assertAlmostEqual(actual, float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
