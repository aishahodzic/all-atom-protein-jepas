"""SIGReg for the projector-free canonical protein LeJEPA experiment.

The implementation in this file is independently written under this
repository's Apache-2.0 license.  Its mathematical behavior is matched to the
official LeVJEPA training implementation supplied by the researcher (LeVJEPA
revision ``941526c428aa513a5bdfa38697724fda794d8496``), without copying the
CC-BY-NC source from LeVJEPA's ``module.py``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class CanonicalSIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularization with LeVJEPA semantics.

    Input axes are ``[view, protein, latent]``.  Each view is regularized
    independently across proteins.  Under DDP, empirical characteristic
    functions are averaged across ranks and the statistic uses the resulting
    global protein count.

    [LEVJEPA EXACT] Quadrature, random unit directions, ECF construction,
    distributed averaging, sample-count scaling, and final averaging match the
    supplied official LeVJEPA implementation.
    """

    def __init__(
        self,
        knots: int = 17,
        num_proj: int = 1024,
        normalize_by_n: bool = False,
    ) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError(f"knots must be at least 2, got {knots}")
        if num_proj < 1:
            raise ValueError(f"num_proj must be positive, got {num_proj}")

        self.num_proj = int(num_proj)
        self.normalize_by_n = bool(normalize_by_n)

        quadrature_points = torch.linspace(0, 3, knots, dtype=torch.float32)
        spacing = 3 / (knots - 1)
        trapezoid_weights = torch.full(
            (knots,), 2 * spacing, dtype=torch.float32
        )
        trapezoid_weights[[0, -1]] = spacing
        gaussian_cf = torch.exp(-quadrature_points.square() / 2.0)

        self.register_buffer("t", quadrature_points)
        self.register_buffer("phi", gaussian_cf)
        self.register_buffer("weights", trapezoid_weights * gaussian_cf)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return SIGReg for ``embeddings`` shaped ``[V, B, D]``."""
        if embeddings.ndim != 3:
            raise ValueError(
                "CanonicalSIGReg expects [views, proteins, latent_dim], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[1] < 1:
            raise ValueError("CanonicalSIGReg requires at least one protein")

        # [LEVJEPA EXACT] One shared bank of random unit directions is used for
        # all views in this invocation.
        directions = torch.randn(
            embeddings.size(-1), self.num_proj, device=embeddings.device
        )
        directions.div_(directions.norm(p=2, dim=0))

        # Every rank samples and normalizes normally, then rank 0's bank is
        # authoritative so the distributed ECF describes the same projections.
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(directions, src=0)

        projected_t = (embeddings @ directions).unsqueeze(-1) * self.t
        empirical_cf = torch.stack(
            (
                projected_t.cos().mean(dim=-3),
                projected_t.sin().mean(dim=-3),
            )
        )

        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(empirical_cf, op=dist.ReduceOp.AVG)
            world_size = dist.get_world_size()

        error = (empirical_cf[0] - self.phi).square() + empirical_cf[1].square()
        statistic = error @ self.weights
        if not self.normalize_by_n:
            statistic = statistic * embeddings.size(-2) * world_size
        return statistic.mean()
