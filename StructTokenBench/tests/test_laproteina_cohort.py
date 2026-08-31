import importlib.machinery
import sys
import types
from pathlib import Path
import unittest
from unittest import mock

import torch


REPOSITORY = Path(__file__).resolve().parents[2]
SRC = REPOSITORY / "StructTokenBench" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPOSITORY))

# DeepSpeed initializes Triton/GPU drivers at import time. These CPU-only tests
# only need its deferred gradient helper to exist, not a runtime accelerator.
deepspeed_stub = types.ModuleType("deepspeed")
deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
deepspeed_stub.__version__ = "0.0.0"
deepspeed_stub.utils = types.SimpleNamespace(
    safe_get_full_grad=lambda parameter: parameter.grad
)
sys.modules.setdefault("deepspeed", deepspeed_stub)

from dataset.base import (  # noqa: E402
    BaseDataset,
    _detach_shared_tensors,
    _resolve_and_parse_worker,
    cohort_summary,
)
from proteinfoundation.nn.feature_factory import ResidueTypeSeqFeat  # noqa: E402
from torch_geometric.data import Data  # noqa: E402


class LaProteinaCohortTest(unittest.TestCase):
    def test_homo_label_mapping_is_idempotent(self):
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.data_name = "TapeRemoteHomologyDataset"
        dataset.target_field = "fold_label"
        dataset.data = [
            {"fold_label": 22},
            {"fold_label": 295},
            {"fold_label": 246},
            {"fold_label": 999},
        ]

        dataset.additional_label_filtering_for_TAPE_homo("laproteina")
        once = [dict(item) for item in dataset.data]
        dataset.additional_label_filtering_for_TAPE_homo("laproteina")

        self.assertEqual(dataset.data, once)
        self.assertEqual(len(dataset.data), 3)
        self.assertTrue(
            all(0 <= item["fold_label"] < 45 for item in dataset.data)
        )

    def test_fresh_low_only_homo_labels_are_not_mistaken_for_mapped_cache(self):
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.data_name = "TapeRemoteHomologyDataset"
        dataset.target_field = "fold_label"
        dataset.data = [
            {"fold_label": 0},
            {"fold_label": 3},
            {"fold_label": 22},
            {"fold_label": 36},
        ]

        dataset.additional_label_filtering_for_TAPE_homo("laproteina")

        # Sorted raw IDs map 0->0, 3->1, 22->6, 36->8. The raw 22/36 must
        # not survive merely because every input happened to be below 45.
        self.assertEqual(
            [item["fold_label"] for item in dataset.data], [0, 1, 6, 8]
        )

    def test_unmarked_legacy_homo_cache_is_recognized_as_already_mapped(self):
        dataset = BaseDataset.__new__(BaseDataset)
        dataset.data_name = "TapeRemoteHomologyDataset"
        dataset.target_field = "fold_label"
        dataset.data = [{"fold_label": value} for value in (0, 3, 22, 36)]

        dataset.additional_label_filtering_for_TAPE_homo(
            "laproteina", from_token_cache=True
        )

        self.assertEqual(
            [item["fold_label"] for item in dataset.data], [0, 3, 22, 36]
        )
        self.assertTrue(
            all(item["_stb_homo_labels_mapped_v1"] for item in dataset.data)
        )

    def test_pyg_graph_tensors_are_detached_from_worker_shared_memory(self):
        original = torch.arange(12, dtype=torch.float32)
        try:
            original.share_memory_()
        except RuntimeError:
            # Some CI/container profiles forbid torch_shm_manager. The pointer
            # assertion below still proves PyG's internal _store was traversed.
            pass
        graph = Data(x=original.reshape(3, 4))

        detached = _detach_shared_tensors(graph)

        self.assertIs(detached, graph)
        self.assertNotEqual(detached.x.data_ptr(), original.data_ptr())
        self.assertFalse(detached.x.is_shared())
        torch.testing.assert_close(detached.x, original.reshape(3, 4))

    def test_publication_worker_bypasses_checkpoint_selector(self):
        graph = types.SimpleNamespace(coords=torch.zeros(3, 37, 3))
        with (
            mock.patch("dataset.base.resolve_chain_id_cached", return_value="A"),
            mock.patch(
                "dataset.base.WrappedLaProteinaTokenizer.matches_training_distribution_metadata",
                side_effect=AssertionError("training selector must not be called"),
            ),
            mock.patch(
                "dataset.base.WrappedLaProteinaTokenizer._in_pretrained_cache",
                return_value=False,
            ),
            mock.patch(
                "dataset.base.WrappedLaProteinaTokenizer._parse_graph",
                return_value=graph,
            ),
        ):
            result = _resolve_and_parse_worker(
                "/tmp/1abc.cif",
                "A",
                False,
                "/tmp",
                apply_checkpoint_training_filters=False,
            )

        resolved, returned_graph, matches, pretrained_hit, error = result
        self.assertEqual(resolved, "A")
        self.assertIs(returned_graph, graph)
        self.assertTrue(matches)
        self.assertFalse(pretrained_hit)
        self.assertIsNone(error)

    def test_worker_preserves_parse_failure_reason(self):
        with (
            mock.patch("dataset.base.resolve_chain_id_cached", return_value="A"),
            mock.patch(
                "dataset.base.WrappedLaProteinaTokenizer._in_pretrained_cache",
                return_value=False,
            ),
            mock.patch(
                "dataset.base.WrappedLaProteinaTokenizer._parse_graph",
                side_effect=RuntimeError("synthetic parse failure"),
            ),
        ):
            result = _resolve_and_parse_worker(
                "/tmp/1abc.cif",
                "A",
                False,
                "/tmp",
                apply_checkpoint_training_filters=False,
            )

        self.assertEqual(result[0], "A")
        self.assertIn("RuntimeError: synthetic parse failure", result[-1])

    def test_cohort_hash_depends_on_order_and_labels_not_root_path(self):
        first = {
            "pdb_id": "1abc",
            "chain_id": "A",
            "residue_range": [""],
            "binding_label": [0, 1],
            "token_ids": torch.zeros(2, 8),
        }
        second = {
            "pdb_id": "2def",
            "chain_id": "B",
            "residue_range": [""],
            "binding_label": [1],
            "token_ids": torch.zeros(1, 8),
        }
        original = cohort_summary([first, second], "binding_label")
        copied = cohort_summary([dict(first), dict(second)], "binding_label")
        reordered = cohort_summary([second, first], "binding_label")

        self.assertEqual(original["cohort_sha256"], copied["cohort_sha256"])
        self.assertNotEqual(original["cohort_sha256"], reordered["cohort_sha256"])

    def test_unknown_residue_gets_zero_identity_without_dropping_position(self):
        feature = ResidueTypeSeqFeat()
        batch = {
            "residue_type": torch.tensor([[0, 20, 19]]),
            "mask_dict": {"residue_type": torch.tensor([[True, True, True]])},
        }
        encoded = feature(batch)

        self.assertEqual(tuple(encoded.shape), (1, 3, 20))
        self.assertEqual(encoded[0, 0].sum().item(), 1.0)
        self.assertEqual(encoded[0, 1].sum().item(), 0.0)
        self.assertEqual(encoded[0, 2].sum().item(), 1.0)

    def test_nonstandard_residue_name_is_preserved_as_openfold_unknown(self):
        from openfold.np.residue_constants import resname_to_idx
        from proteinfoundation.datasets.pdb_data import (
            residue_names_to_openfold_indices,
        )

        indices = residue_names_to_openfold_indices(["ALA", "SEC", "TYR"])

        self.assertEqual(
            indices,
            [resname_to_idx["ALA"], resname_to_idx["UNK"], resname_to_idx["TYR"]],
        )


if __name__ == "__main__":
    unittest.main()
