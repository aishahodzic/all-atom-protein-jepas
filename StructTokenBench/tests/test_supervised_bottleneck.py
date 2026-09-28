import importlib.machinery
import sys
from pathlib import Path
import types
from types import SimpleNamespace
import unittest

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# DeepSpeed initializes Triton/GPU drivers at import time.  These CPU-only
# tests only exercise the downstream PyTorch module.
deepspeed_stub = types.ModuleType("deepspeed")
deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
deepspeed_stub.__version__ = "0.0.0"
deepspeed_stub.utils = types.SimpleNamespace(
    safe_get_full_grad=lambda parameter: parameter.grad
)
sys.modules.setdefault("deepspeed", deepspeed_stub)

from supervised_bottleneck import (  # noqa: E402
    SupervisedBottleneckSequenceClassificationModel,
)


def config(**overrides):
    values = {
        "pretrained_ckpt_path": "",
        "num_labels": 1,
        "dropout": 0.1,
        "num_layer": 1,
        "hidden_size": 16,
        "num_tokens": None,
        "d_model": 12,
        "supervised_bottleneck_dim": 3,
        "is_global_or_local": "local",
        "multi_label": False,
        "regression": False,
        "classification_metric_aggregation": "legacy_batch_mean",
        "binary_class_weighting": "legacy_batch",
        "binary_positive_weight": 1.0,
        "binary_negative_weight": 1.0,
        "use_sequence": False,
        "sequence_only": False,
        "add_noise": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SupervisedBottleneckTest(unittest.TestCase):
    def test_projects_before_standard_downstream_stack(self):
        torch.manual_seed(1234)
        model = SupervisedBottleneckSequenceClassificationModel(config())
        features = torch.randn(2, 5, 12)
        mask = torch.tensor(
            [[False, False, False, True, True], [False, False, False, False, True]]
        )
        seqs = torch.zeros(2, 5, dtype=torch.long)
        targets = torch.zeros(2, 5)
        output = model((features, mask, seqs), targets)

        self.assertEqual(model.supervised_bottleneck[1].weight.shape, (3, 12))
        self.assertEqual(model.position_embed.embedding_dim, 3)
        self.assertEqual(model.classify.classify[0].in_features, 3)
        self.assertTrue(torch.isfinite(output[0][0]))
        output[0][0].backward()
        self.assertIsNotNone(model.supervised_bottleneck[1].weight.grad)
        self.assertTrue(
            torch.isfinite(model.supervised_bottleneck[1].weight.grad).all()
        )

    def test_rejects_non_compressive_width(self):
        with self.assertRaisesRegex(ValueError, "input_dim > bottleneck_dim"):
            SupervisedBottleneckSequenceClassificationModel(
                config(d_model=8, supervised_bottleneck_dim=8)
            )


if __name__ == "__main__":
    unittest.main()
