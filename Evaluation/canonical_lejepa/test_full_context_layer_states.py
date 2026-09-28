import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from Evaluation.canonical_lejepa.extract_full_context_layer_states import (
    _atomic_torch_save,
    _assert_nested_equal,
    select_pool_indices,
)


class FullContextLayerStateUtilityTests(unittest.TestCase):
    def test_selection_is_deterministic_and_preserves_existing_split(self):
        protein_ids = [f"train_{i:03d}" for i in range(80)] + [
            f"test_{i:03d}" for i in range(20)
        ]
        probe_split = {
            protein_id: ("probe_train" if protein_id.startswith("train") else "probe_test")
            for protein_id in protein_ids
        }
        pool = SimpleNamespace(protein_ids=protein_ids, probe_split=probe_split)
        first = select_pool_indices(pool, 10, 42)
        second = select_pool_indices(pool, 10, 42)
        self.assertEqual(first, second)
        selected_splits = [probe_split[protein_ids[index]] for index in first]
        self.assertEqual(selected_splits.count("probe_train"), 8)
        self.assertEqual(selected_splits.count("probe_test"), 2)

    def test_atomic_save_round_trip_is_exact(self):
        value = [{"protein_id": "x", "state": torch.randn(3, 8), "labels": ["A", "G", "V"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.pt"
            _atomic_torch_save(value, path)
            reloaded = torch.load(path, map_location="cpu", weights_only=False)
            _assert_nested_equal(value, reloaded)


if __name__ == "__main__":
    unittest.main()
