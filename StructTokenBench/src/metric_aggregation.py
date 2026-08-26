"""Epoch-level metric aggregation helpers.

Classification metrics such as AUROC, F1, and MCC are not additive.  Computing
them independently on every minibatch and averaging the results changes the
answer when the same examples are rebatched.  The helpers in this module keep
the operation explicit: pool predictions first, then compute each metric once.
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torchmetrics.classification import (
    AUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    ConfusionMatrix,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
)


CLASSIFICATION_METRIC_AGGREGATIONS = {"global", "legacy_batch_mean"}
BINARY_CLASS_WEIGHTING_MODES = {"none", "fixed", "legacy_batch"}


def _concatenate_if_needed(value):
    if isinstance(value, Sequence):
        if not value:
            raise ValueError("Cannot aggregate an empty sequence of tensors")
        return torch.cat(list(value), dim=0)
    return value


def calculate_epoch_classification_metrics(logits, targets, task):
    """Calculate classification metrics once over all supplied examples.

    ``logits`` and ``targets`` may be complete tensors or sequences of batch
    tensors.  Supporting both forms makes batch-invariance directly testable.
    """

    logits = _concatenate_if_needed(logits)
    targets = _concatenate_if_needed(targets)
    device = logits.device

    if task == "binary":
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        # Pass probabilities explicitly.  TorchMetrics otherwise guesses
        # whether an input is logits from its numeric range; an all-[0, 1]
        # logit tensor would therefore get the wrong decision threshold.
        probabilities = torch.sigmoid(logits.float())

        accuracy = BinaryAccuracy().to(device)(probabilities, targets)
        auroc = AUROC(task="binary").to(device)(probabilities, targets)
        f1_score = BinaryF1Score().to(device)(probabilities, targets)
        mcc = BinaryMatthewsCorrCoef().to(device)(probabilities, targets)

        confusion_by_class = ConfusionMatrix(
            task="binary", num_classes=2, normalize="true"
        ).to(device)(probabilities, targets)
        confusion_overall = ConfusionMatrix(
            task="binary", num_classes=2, normalize="all"
        ).to(device)(probabilities, targets)

        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "f1_score": f1_score,
            "mcc": mcc,
            "true_neg": confusion_by_class[0, 0],
            "false_pos": confusion_by_class[0, 1],
            "false_neg": confusion_by_class[1, 0],
            "true_pos": confusion_by_class[1, 1],
            "true_neg_toall": confusion_overall[0, 0],
            "false_pos_toall": confusion_overall[0, 1],
            "false_neg_toall": confusion_overall[1, 0],
            "true_pos_toall": confusion_overall[1, 1],
            "unweighted_loss": nn.BCEWithLogitsLoss()(logits, targets.float()),
        }

    if task == "multiclass":
        num_classes = logits.shape[-1]
        targets = targets.long()
        probabilities = torch.softmax(logits.float(), dim=-1)
        return {
            "accuracy": MulticlassAccuracy(num_classes=num_classes).to(device)(
                probabilities, targets
            ).float(),
            "auroc": AUROC(task="multiclass", num_classes=num_classes).to(device)(
                probabilities, targets
            ).float(),
            "f1_score": MulticlassF1Score(num_classes=num_classes).to(device)(
                probabilities, targets
            ).float(),
            "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)(
                probabilities, targets
            ).float(),
        }

    raise ValueError(f"Unknown classification task: {task!r}")


def gather_variable_length_tensor(tensor):
    """All-gather a tensor whose first dimension may differ between ranks.

    Validation datasets are sharded before their DataLoaders are constructed,
    so ranks can retain different numbers of residues.  Padding only for the
    collective and then trimming prevents a mean-of-rank-metrics bug from
    replacing the original mean-of-batch-metrics bug.
    """

    if not dist.is_available() or not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    tensor = tensor.contiguous()
    local_length = torch.tensor(
        [tensor.shape[0]], device=tensor.device, dtype=torch.long
    )
    gathered_lengths = [torch.zeros_like(local_length) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, local_length)
    lengths = [int(length.item()) for length in gathered_lengths]
    max_length = max(lengths)

    if tensor.shape[0] < max_length:
        padding = torch.zeros(
            (max_length - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding), dim=0)

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(
        [
            rank_tensor[:rank_length]
            for rank_tensor, rank_length in zip(gathered, lengths)
        ],
        dim=0,
    )


def balanced_binary_class_weights(positive_count, negative_count):
    """Return fixed inverse-frequency sample weights from training-set counts."""

    if positive_count <= 0 or negative_count <= 0:
        raise ValueError(
            "Balanced binary weighting requires at least one example of each class; "
            f"got positive_count={positive_count}, negative_count={negative_count}"
        )
    total = positive_count + negative_count
    return total / (2.0 * positive_count), total / (2.0 * negative_count)


def binary_classification_loss(
    logits,
    targets,
    weighting_mode,
    positive_weight=1.0,
    negative_weight=1.0,
):
    """Binary loss with fixed configured weights or an explicit legacy mode."""

    if weighting_mode not in BINARY_CLASS_WEIGHTING_MODES:
        raise ValueError(
            f"binary_class_weighting must be one of "
            f"{sorted(BINARY_CLASS_WEIGHTING_MODES)}, got {weighting_mode!r}"
        )

    if weighting_mode == "legacy_batch":
        positive_count = targets.sum()
        negative_count = targets.numel() - positive_count
        if positive_count <= 0:
            # Literal upstream semantics: the absent class reports an infinite
            # inverse-frequency weight, while every actually used (negative)
            # sample has weight 0.5. The infinite value is never assigned to a
            # sample, so loss and gradients remain finite.
            positive_weight = float("inf")
            negative_weight = 0.5
        elif negative_count <= 0:
            positive_weight = 0.5
            negative_weight = float("inf")
        else:
            total = targets.numel()
            # Preserve upstream's arithmetic order as well as its algebra:
            # ``(N / count) * 0.5`` is not always bit-identical in float32 to
            # ``N / (2 * count)``.
            positive_weight = (total / positive_count) * 0.5
            negative_weight = (total / negative_count) * 0.5
    elif weighting_mode == "none":
        positive_weight = negative_weight = 1.0

    sample_weights = torch.full_like(targets, float(negative_weight))
    sample_weights[targets.long() == 1] = float(positive_weight)
    loss = nn.BCEWithLogitsLoss(weight=sample_weights)(logits, targets)
    return loss, float(positive_weight), float(negative_weight)
