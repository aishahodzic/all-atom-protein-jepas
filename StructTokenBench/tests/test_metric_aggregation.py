import sys
import importlib.machinery
import types
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metric_aggregation import (  # noqa: E402
    balanced_binary_class_weights,
    binary_classification_loss,
    calculate_epoch_classification_metrics,
    gather_variable_length_tensor,
)


def test_binary_auroc_is_computed_after_pooling_not_meaned_per_batch():
    logits = [torch.tensor([0.8, 0.9]), torch.tensor([0.1, 0.2])]
    targets = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]

    per_batch_aurocs = [
        calculate_epoch_classification_metrics(x, y, task="binary")["auroc"]
        for x, y in zip(logits, targets)
    ]
    pooled = calculate_epoch_classification_metrics(logits, targets, task="binary")

    torch.testing.assert_close(torch.stack(per_batch_aurocs).mean(), torch.tensor(1.0))
    torch.testing.assert_close(pooled["auroc"], torch.tensor(0.75))


def test_binary_metrics_are_invariant_to_rebatching():
    logits = torch.tensor([-2.0, 1.0, 0.5, -0.5, 2.0, -1.0, 1.5, -1.5])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    batches_a = ([logits[:2], logits[2:]], [targets[:2], targets[2:]])
    batches_b = (
        [logits[:1], logits[1:5], logits[5:]],
        [targets[:1], targets[1:5], targets[5:]],
    )
    metrics_a = calculate_epoch_classification_metrics(*batches_a, task="binary")
    metrics_b = calculate_epoch_classification_metrics(*batches_b, task="binary")

    assert metrics_a.keys() == metrics_b.keys()
    for metric_name in metrics_a:
        torch.testing.assert_close(metrics_a[metric_name], metrics_b[metric_name])


def test_binary_logits_use_zero_decision_threshold_even_when_all_are_in_zero_one():
    metrics = calculate_epoch_classification_metrics(
        torch.tensor([0.1, 0.2]), torch.tensor([0.0, 1.0]), task="binary"
    )

    # Both logits are positive, hence both sigmoid probabilities exceed 0.5:
    # TP=1, FP=1, giving F1=2/3.  Passing these raw values to TorchMetrics
    # would misidentify them as probabilities and incorrectly return F1=0.
    torch.testing.assert_close(metrics["f1_score"], torch.tensor(2.0 / 3.0))


def test_multiclass_metrics_are_invariant_to_rebatching():
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 3.0],
            [1.5, 2.0, 0.0],
            [0.5, 0.0, 2.5],
            [2.0, 1.5, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 0, 2, 1])

    metrics_a = calculate_epoch_classification_metrics(
        [logits[:2], logits[2:]], [targets[:2], targets[2:]], task="multiclass"
    )
    metrics_b = calculate_epoch_classification_metrics(
        [logits[:1], logits[1:4], logits[4:]],
        [targets[:1], targets[1:4], targets[4:]],
        task="multiclass",
    )

    assert metrics_a.keys() == metrics_b.keys()
    for metric_name in metrics_a:
        torch.testing.assert_close(metrics_a[metric_name], metrics_b[metric_name])


def test_fixed_binary_weights_are_derived_once_from_global_counts():
    positive_weight, negative_weight = balanced_binary_class_weights(2, 8)
    assert positive_weight == 2.5
    assert negative_weight == 0.625

    logits = torch.zeros(10)
    targets = torch.tensor([1.0, 1.0] + [0.0] * 8)
    loss, used_positive_weight, used_negative_weight = binary_classification_loss(
        logits,
        targets,
        weighting_mode="fixed",
        positive_weight=positive_weight,
        negative_weight=negative_weight,
    )

    assert torch.isfinite(loss)
    assert used_positive_weight == positive_weight
    assert used_negative_weight == negative_weight


def test_legacy_weighting_no_longer_divides_by_zero_on_single_class_batch():
    for targets in (torch.zeros(4), torch.ones(4)):
        logits = torch.zeros(4)
        loss, positive_weight, negative_weight = binary_classification_loss(
            logits, targets, weighting_mode="legacy_batch"
        )

        assert torch.isfinite(loss)
        if targets.sum() == 0:
            assert positive_weight == float("inf")
            assert negative_weight == 0.5
        else:
            assert positive_weight == 0.5
            assert negative_weight == float("inf")
        expected = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )
        torch.testing.assert_close(loss, expected)


def test_legacy_mixed_batch_matches_upstream_float32_arithmetic_order():
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    logits = torch.linspace(-1.0, 1.0, targets.numel())

    loss, positive_weight, negative_weight = binary_classification_loss(
        logits, targets, weighting_mode="legacy_batch"
    )

    upstream_positive = (targets.numel() / targets.sum()) * 0.5
    upstream_negative = (
        targets.numel() / (targets.numel() - targets.sum())
    ) * 0.5
    upstream_sample_weights = torch.ones_like(targets) * upstream_negative
    upstream_sample_weights[targets.long() == 1] = upstream_positive
    upstream_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=upstream_sample_weights
    )

    assert positive_weight == float(upstream_positive)
    assert negative_weight == float(upstream_negative)
    torch.testing.assert_close(loss, upstream_loss, rtol=0, atol=0)


def test_gather_is_identity_without_distributed_initialization():
    value = torch.tensor([[1.0], [2.0]])
    assert gather_variable_length_tensor(value) is value


def _load_pl_model_class():
    # DeepSpeed's import initializes Triton and requires a GPU driver even for
    # this CPU-only aggregation test.  Model code only needs ``deepspeed.utils``
    # after optimizer steps, so a small import stub keeps this test focused.
    previous_deepspeed = sys.modules.get("deepspeed")
    installed_stub = "model_module" not in sys.modules
    if installed_stub:
        deepspeed_stub = types.ModuleType("deepspeed")
        deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec(
            "deepspeed", loader=None
        )
        deepspeed_stub.__version__ = "0.0.0"
        deepspeed_stub.utils = types.SimpleNamespace(
            safe_get_full_grad=lambda parameter: parameter.grad
        )
        sys.modules["deepspeed"] = deepspeed_stub
    try:
        from model_module import PlModel
    finally:
        if installed_stub:
            if previous_deepspeed is None:
                sys.modules.pop("deepspeed", None)
            else:
                sys.modules["deepspeed"] = previous_deepspeed

    return PlModel


def _make_pl_model():
    pl_model_class = _load_pl_model_class()
    model_cfg = types.SimpleNamespace(
        class_name="SequenceClassificationModel",
        regression=False,
        multi_label=False,
        classification_metric_aggregation="global",
        is_global_or_local="local",
    )
    optimizer_cfg = types.SimpleNamespace(micro_batch_size=2)
    return pl_model_class(
        model_cfg=model_cfg,
        trainer=types.SimpleNamespace(),
        py_logger=types.SimpleNamespace(),
        optimizer_cfg=optimizer_cfg,
        all_split_names=["validation"],
        codebook_embedding=None,
    )


def _binary_batch_output(logits, targets):
    zero = torch.tensor(0.0)
    return {
        "validation_loss": zero,
        "validation_accuracy": zero,
        "validation_auroc": torch.tensor(1.0),
        "validation_f1_score": zero,
        "validation_mcc": zero,
        "validation_true_neg": zero,
        "validation_false_pos": zero,
        "validation_false_neg": zero,
        "validation_true_pos": zero,
        "validation_true_neg_toall": zero,
        "validation_false_pos_toall": zero,
        "validation_false_neg_toall": zero,
        "validation_true_pos_toall": zero,
        "validation_unweighted_loss": zero,
        "validation_pos_weight": torch.tensor(1.0),
        "validation_neg_weight": torch.tensor(1.0),
        "logits": logits,
        "targets": targets,
    }


def test_pl_model_wires_pooled_metric_into_checkpoint_metric_name():
    module = _make_pl_model()
    logged = {}
    module.log_dict = lambda values, **kwargs: logged.update(values)

    module._valid_or_test_epoch_end(
        [
            _binary_batch_output(torch.tensor([0.8, 0.9]), torch.tensor([0.0, 1.0])),
            _binary_batch_output(torch.tensor([0.1, 0.2]), torch.tensor([0.0, 1.0])),
        ],
        split="validation",
    )

    # The callback still monitors validation_f1_score, while AUROC makes the
    # difference between pooled and batch-mean aggregation especially clear.
    torch.testing.assert_close(logged["validation_auroc"], torch.tensor(0.75))


def test_pl_model_does_not_step_log_names_reserved_for_global_metrics():
    module = _make_pl_model()
    module.model = lambda input_list, targets: (
        (
            torch.tensor(0.25),
            {"auroc": torch.tensor(1.0), "f1_score": torch.tensor(1.0)},
        ),
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.0, 1.0]),
    )
    logged = {}
    module.log_dict = lambda values, **kwargs: logged.update(values)

    module._valid_or_test_step(
        {
            "input_list": (torch.empty(0),),
            "targets": torch.tensor([0.0, 1.0]),
        },
        batch_idx=0,
        split="validation",
    )

    assert set(logged) == {"validation_loss"}
