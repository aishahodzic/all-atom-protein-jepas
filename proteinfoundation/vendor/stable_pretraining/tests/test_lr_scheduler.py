"""Parity tests for the scheduler vendored from stable-pretraining.

The local reference factory below is the exact scheduler composition at pinned
revision 3001802f6c8afcfaae99a5a3760edd952508d1e0. The optional checkout test
also compares directly against that upstream source when a pinned checkout is
provided through ``STABLE_PRETRAINING_PINNED_CHECKOUT``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import unittest

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from proteinfoundation.vendor.stable_pretraining.lr_scheduler import (
    UPSTREAM_REVISION,
    UPSTREAM_SOURCE,
    LinearWarmupCosineAnnealing,
)


EXPECTED_REVISION = "3001802f6c8afcfaae99a5a3760edd952508d1e0"


def _pinned_upstream_reference(
    optimizer, total_steps, start_factor=0.01, end_lr=0.0, peak_step=0.01
):
    """Exact factory body from the pinned upstream revision."""
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    anneal = CosineAnnealingLR(
        optimizer, T_max=total_steps - peak_step, eta_min=end_lr
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup, anneal],
        milestones=[peak_step],
    )
    return scheduler


def _make_optimizer(lr: float) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.SGD([parameter], lr=lr)


def _learning_rate_trace(factory, *, steps: int, **kwargs) -> list[float]:
    optimizer = _make_optimizer(kwargs.pop("lr"))
    scheduler = factory(optimizer, **kwargs)
    trace = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        trace.append(optimizer.param_groups[0]["lr"])
    return trace


class LinearWarmupCosineAnnealingParityTest(unittest.TestCase):
    def assert_trace_matches_reference(self, **kwargs) -> None:
        actual = _learning_rate_trace(LinearWarmupCosineAnnealing, **kwargs)
        expected = _learning_rate_trace(_pinned_upstream_reference, **kwargs)
        self.assertEqual(actual, expected)

    def test_canonical_lejepa_100k_flat_after_warmup_recipe(self):
        self.assert_trace_matches_reference(
            lr=4e-4,
            steps=100_000,
            total_steps=100_000,
            start_factor=0.25,
            end_lr=4e-4,
            peak_step=12_000,
        )

    def test_canonical_recipe_step_index_values(self):
        trace = _learning_rate_trace(
            LinearWarmupCosineAnnealing,
            lr=4e-4,
            steps=100_000,
            total_steps=100_000,
            start_factor=0.25,
            end_lr=4e-4,
            peak_step=12_000,
        )
        self.assertEqual(trace[0], 1e-4)
        self.assertAlmostEqual(trace[6_000], 2.5e-4, places=15)
        self.assertAlmostEqual(trace[11_999], 3.99975e-4, places=15)
        self.assertEqual(trace[12_000], 4e-4)
        self.assertEqual(trace[12_001], 4e-4)
        self.assertEqual(trace[100_000], 4e-4)

    def test_fractional_peak_step_and_cosine_decay(self):
        self.assert_trace_matches_reference(
            lr=1e-3,
            steps=40,
            total_steps=40,
            start_factor=0.1,
            end_lr=1e-5,
            peak_step=0.2,
        )

    def test_provenance_and_adjacent_mit_license(self):
        self.assertEqual(UPSTREAM_REVISION, EXPECTED_REVISION)
        self.assertIn(EXPECTED_REVISION, UPSTREAM_SOURCE)
        license_path = Path(__file__).resolve().parents[1] / "LICENSE"
        license_text = license_path.read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2024 rbalestr-lab", license_text)

    def test_direct_parity_with_pinned_checkout_when_available(self):
        checkout_value = os.environ.get("STABLE_PRETRAINING_PINNED_CHECKOUT")
        if checkout_value is None:
            self.skipTest("STABLE_PRETRAINING_PINNED_CHECKOUT is not set")
        checkout = Path(checkout_value)
        source = checkout / "stable_pretraining" / "optim" / "lr_scheduler.py"
        if not source.is_file():
            self.skipTest("pinned stable-pretraining checkout is not available")

        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(revision, EXPECTED_REVISION)

        spec = importlib.util.spec_from_file_location(
            "_pinned_stable_pretraining_lr_scheduler", source
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)

        actual = _learning_rate_trace(
            LinearWarmupCosineAnnealing,
            lr=4e-4,
            steps=100_000,
            total_steps=100_000,
            start_factor=0.25,
            end_lr=4e-4,
            peak_step=12_000,
        )
        expected = _learning_rate_trace(
            upstream.LinearWarmupCosineAnnealing,
            lr=4e-4,
            steps=100_000,
            total_steps=100_000,
            start_factor=0.25,
            end_lr=4e-4,
            peak_step=12_000,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
