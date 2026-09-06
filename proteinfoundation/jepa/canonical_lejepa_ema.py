"""Evaluation-only encoder EMA for canonical LeJEPA.

Adapted from the official LeVJEPA ``WeightEMA`` callback at repository
revision ``941526c428aa513a5bdfa38697724fda794d8496``.  The official LeVJEPA
repository root is MIT licensed; the upstream copyright and license notice are
retained below.  This callback is deliberately isolated from La-Proteina's
historical optimizer-wrapping EMA implementation.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 GalilAI-group

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback


LEVJEPA_UPSTREAM_REVISION = "941526c428aa513a5bdfa38697724fda794d8496"
LEVJEPA_WEIGHT_EMA_SOURCE = (
    "https://github.com/MLO-lab/LeVJEPA/blob/"
    f"{LEVJEPA_UPSTREAM_REVISION}/callbacks.py"
)


class CanonicalEncoderEMA(Callback):
    """Maintain a float32 EMA of selected encoder parameters for evaluation.

    [LEVJEPA EXACT] Decay is interpreted per optimizer step.  An update made
    every ``update_every`` steps uses ``decay ** elapsed_steps``.  Training
    weights remain untouched and the EMA mapping is stored separately under
    ``save_key`` in each checkpoint.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        update_every: int = 32,
        start_step: int = 0,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        save_key: str = "state_dict_ema",
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")

        self.decay = float(decay)
        self.update_every = int(update_every)
        self.start_step = int(start_step)
        self.include = tuple(include or ())
        self.exclude = tuple(exclude or ())
        self.save_key = save_key
        self.device = device
        self.dtype = dtype

        self._ema_state: dict[str, torch.Tensor] = {}
        self._loaded_ema_state: dict[str, torch.Tensor] | None = None
        self._last_global_step = -1
        self._last_ema_step = -1
        self._num_updates = 0

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"[save_key={self.save_key},include={self.include}]"
        )
    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._last_global_step = int(trainer.global_step)
        if self._last_ema_step < 0:
            self._last_ema_step = int(trainer.global_step)
        if self._loaded_ema_state is not None:
            self._ema_state = {
                name: tensor.to(self._target_device(pl_module), dtype=self._dtype)
                for name, tensor in self._loaded_ema_state.items()
            }
            self._loaded_ema_state = None
        elif not self._ema_state:
            self._init_from_module(pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        current_step = int(trainer.global_step)
        if current_step == self._last_global_step:
            return
        self._last_global_step = current_step

        if current_step < self.start_step or current_step % self.update_every != 0:
            return
        if not self._ema_state:
            self._init_from_module(pl_module)
            return

        elapsed_steps = max(current_step - self._last_ema_step, 1)
        update_decay = self.decay**elapsed_steps
        with torch.no_grad():
            current_state = pl_module.state_dict()
            for name in self._tracked_names(pl_module):
                value = current_state[name]
                if not value.is_floating_point():
                    continue
                ema_value = self._ema_state[name]
                value = value.detach().to(
                    device=ema_value.device, dtype=ema_value.dtype
                )
                ema_value.mul_(update_decay).add_(
                    value, alpha=1.0 - update_decay
                )
        self._last_ema_step = current_step
        self._num_updates += 1

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del trainer, pl_module
        checkpoint[self.save_key] = {
            name: tensor.detach().cpu()
            for name, tensor in sorted(self._ema_state.items())
        }
        checkpoint[f"{self.save_key}_meta"] = {
            "decay": self.decay,
            "update_every": self.update_every,
            "start_step": self.start_step,
            "include": self.include,
            "exclude": self.exclude,
            "effective_decay": self.decay**self.update_every,
            "num_updates": self._num_updates,
            "last_ema_step": self._last_ema_step,
            "upstream_revision": LEVJEPA_UPSTREAM_REVISION,
        }

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del trainer, pl_module
        ema_state = checkpoint.get(self.save_key)
        if ema_state is not None:
            self._loaded_ema_state = {
                name: tensor.detach().cpu() for name, tensor in ema_state.items()
            }
        metadata = checkpoint.get(f"{self.save_key}_meta") or {}
        self._num_updates = int(metadata.get("num_updates", 0))
        self._last_ema_step = int(metadata.get("last_ema_step", -1))

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_updates": self._num_updates,
            "last_global_step": self._last_global_step,
            "last_ema_step": self._last_ema_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._num_updates = int(state_dict.get("num_updates", 0))
        self._last_global_step = int(state_dict.get("last_global_step", -1))
        self._last_ema_step = int(state_dict.get("last_ema_step", -1))

    @property
    def _dtype(self) -> torch.dtype:
        dtype = getattr(torch, self.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown torch dtype: {self.dtype}")
        return dtype

    def _target_device(self, pl_module: pl.LightningModule) -> torch.device:
        if self.device == "cuda":
            return pl_module.device
        return torch.device(self.device)

    def _init_from_module(self, pl_module: pl.LightningModule) -> None:
        target_device = self._target_device(pl_module)
        current_state = pl_module.state_dict()
        self._ema_state = {}
        with torch.no_grad():
            for name in self._tracked_names(pl_module):
                value = current_state[name]
                if value.is_floating_point():
                    self._ema_state[name] = (
                        value.detach()
                        .to(device=target_device, dtype=self._dtype)
                        .clone()
                    )

    def _tracked_names(self, pl_module: pl.LightningModule) -> tuple[str, ...]:
        try:
            named_parameters = pl_module.named_parameters(with_callbacks=False)
        except TypeError:
            named_parameters = pl_module.named_parameters()

        state_keys = set(pl_module.state_dict().keys())
        names = []
        for name, parameter in named_parameters:
            if name not in state_keys or not parameter.is_floating_point():
                continue
            if self.include and not self._matches_prefix(name, self.include):
                continue
            if self.exclude and self._matches_prefix(name, self.exclude):
                continue
            names.append(name)
        return tuple(names)

    @staticmethod
    def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
        return any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes
        )
