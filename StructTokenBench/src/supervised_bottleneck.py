"""Opt-in supervised bottleneck for continuous StructTokenBench readouts.

The frozen tokenizer continues to emit its native-width representation.  This
module replaces only the downstream ``SequenceClassificationModel`` with a
variant that learns

    LayerNorm(input_dim) -> Linear(input_dim, bottleneck_dim, bias=False)

jointly with the ordinary task head.  It intentionally does not alter token
caches or the frozen encoder and is installed only by the dedicated experiment
entrypoint.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

import model_module


_ORIGINAL_MODEL = model_module.SequenceClassificationModel


class SupervisedBottleneckSequenceClassificationModel(_ORIGINAL_MODEL):
    """The standard STB model preceded by a task-supervised linear bottleneck."""

    def __init__(self, model_cfg, codebook_embedding=None):
        input_dim = int(model_cfg.d_model)
        bottleneck_dim = int(model_cfg.supervised_bottleneck_dim)
        if input_dim <= bottleneck_dim:
            raise ValueError(
                "supervised bottleneck requires input_dim > bottleneck_dim, got "
                f"{input_dim} -> {bottleneck_dim}"
            )
        if model_cfg.num_tokens is not None:
            raise ValueError("supervised bottleneck requires continuous tokenizer outputs")
        if bool(model_cfg.use_sequence) or bool(model_cfg.sequence_only):
            raise ValueError("supervised bottleneck does not support sequence features")

        downstream_cfg = copy.deepcopy(model_cfg)
        downstream_cfg.d_model = bottleneck_dim
        super().__init__(downstream_cfg, codebook_embedding=codebook_embedding)

        self.supervised_bottleneck_input_dim = input_dim
        self.supervised_bottleneck_dim = bottleneck_dim
        self.supervised_bottleneck = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck_dim, bias=False),
        )

    def inference_feature(self, input_list):
        input_ids, input_mask, seqs = input_list
        if input_ids.ndim != 3:
            raise ValueError(
                "supervised bottleneck expected continuous [batch, length, width] "
                f"features, got shape {tuple(input_ids.shape)}"
            )
        if input_ids.shape[-1] != self.supervised_bottleneck_input_dim:
            raise ValueError(
                "supervised bottleneck input width mismatch: expected "
                f"{self.supervised_bottleneck_input_dim}, got {input_ids.shape[-1]}"
            )
        projected = self.supervised_bottleneck(input_ids)
        return super().inference_feature((projected, input_mask, seqs))


def install_supervised_bottleneck() -> None:
    """Install the experiment model into STB's deferred model factory."""

    current = model_module.SequenceClassificationModel
    if current is SupervisedBottleneckSequenceClassificationModel:
        return
    if current is not _ORIGINAL_MODEL:
        raise RuntimeError(
            "SequenceClassificationModel was already replaced by an unknown experiment"
        )
    model_module.SequenceClassificationModel = (
        SupervisedBottleneckSequenceClassificationModel
    )

