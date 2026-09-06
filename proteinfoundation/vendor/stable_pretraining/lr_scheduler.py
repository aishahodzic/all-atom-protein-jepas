"""Vendored stable-pretraining learning-rate scheduler.

This file contains only the scheduler required by canonical LeJEPA. It is
copied from ``stable-pretraining`` revision
``3001802f6c8afcfaae99a5a3760edd952508d1e0``:

https://github.com/galilai-group/stable-pretraining/blob/3001802f6c8afcfaae99a5a3760edd952508d1e0/stable_pretraining/optim/lr_scheduler.py

The implementation is intentionally kept behavior-identical to that pinned
source. See the adjacent ``LICENSE`` file for the upstream MIT license.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2024 rbalestr-lab

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


UPSTREAM_REVISION = "3001802f6c8afcfaae99a5a3760edd952508d1e0"
UPSTREAM_SOURCE = (
    "https://github.com/galilai-group/stable-pretraining/blob/"
    f"{UPSTREAM_REVISION}/stable_pretraining/optim/lr_scheduler.py"
)


def LinearWarmupCosineAnnealing(
    optimizer, total_steps, start_factor=0.01, end_lr=0.0, peak_step=0.01
):
    """Combine linear warmup with cosine annealing decay.

    This function creates a scheduler that first linearly warms up the learning rate,
    then applies cosine annealing decay. This is commonly used in self-supervised
    learning to achieve better convergence.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): Total number of training steps.
        start_factor (float, optional): Initial learning rate factor for warmup. Defaults to 0.01.
        end_lr (float, optional): Final learning rate after annealing. Defaults to 0.0.
        peak_step (float, optional): Step at which warmup ends (as fraction of total_steps).
                                   Defaults to 0.01.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Combined warmup and annealing scheduler.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = LinearWarmupCosineAnnealing(optimizer, total_steps=1000)
    """
    if peak_step < 1:
        peak_step = int(peak_step * total_steps)
    warmup = LinearLR(optimizer, start_factor, total_iters=peak_step)
    anneal = CosineAnnealingLR(optimizer, T_max=total_steps - peak_step, eta_min=end_lr)
    scheduler = SequentialLR(
        optimizer,
        [warmup, anneal],
        milestones=[peak_step],
    )
    return scheduler
