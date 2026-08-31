import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

import shared_embedding_cache as shared


class DummyTokenizer:
    checkpoint_sha256 = "checkpoint-a"
    checkpoint_type = "jepa"
    encoder_source = "online"
    encoder_state_prefix = "online_encoder"
    encoder_cfg_sha256 = "config-a"
    latent_z_dim = 8
    evaluation_protocol = "upstream-l600-v1"


def expected_result(value=1.0):
    return (
        torch.tensor([[value, 2.0], [3.0, 4.0]], dtype=torch.float32),
        np.array([10, 11]),
        ["ALA", "GLY"],
    )


def _concurrent_worker(structure_path, counter_path):
    tokenizer = DummyTokenizer()

    def compute():
        with open(counter_path, "a", encoding="utf-8") as handle:
            handle.write("computed\n")
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(0.2)
        return expected_result()

    shared.get_or_compute_shared_embedding(
        tokenizer, structure_path, "A", False, compute
    )


class SharedEmbeddingCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.structure = self.root / "1abc.cif"
        self.structure.write_text("structure-v1\n", encoding="utf-8")
        self.environment = mock.patch.dict(
            os.environ,
            {
                "STB_SHARED_EMBEDDING_CACHE": "1",
                "STB_SHARED_EMBEDDING_CACHE_ROOT": str(self.root / "cache"),
            },
            clear=False,
        )
        self.environment.start()
        self.implementation = mock.patch.object(
            shared,
            "_encoder_implementation_identity",
            return_value={"files": {}, "sha256": "implementation-a"},
        )
        self.implementation.start()
        self.versions = mock.patch.object(
            shared, "scientific_runtime_versions", return_value={"torch": "test"}
        )
        self.versions.start()
        shared._sha256_for_unchanged_file.cache_clear()

    def tearDown(self):
        self.versions.stop()
        self.implementation.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def test_second_task_reuses_bit_identical_embedding(self):
        calls = []

        def compute():
            calls.append(1)
            return expected_result()

        first = shared.get_or_compute_shared_embedding(
            DummyTokenizer(), str(self.structure), "A", False, compute
        )
        second = shared.get_or_compute_shared_embedding(
            DummyTokenizer(), str(self.structure), "A", False,
            lambda: self.fail("cache hit unexpectedly recomputed the encoder"),
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_structure_content_and_checkpoint_invalidate_independently(self):
        calls = []

        def compute(value):
            def inner():
                calls.append(value)
                return expected_result(value)
            return inner

        tokenizer = DummyTokenizer()
        shared.get_or_compute_shared_embedding(
            tokenizer, str(self.structure), "A", False, compute(1.0)
        )

        self.structure.write_text("structure-v2-longer\n", encoding="utf-8")
        shared.get_or_compute_shared_embedding(
            tokenizer, str(self.structure), "A", False, compute(2.0)
        )

        changed_checkpoint = DummyTokenizer()
        changed_checkpoint.checkpoint_sha256 = "checkpoint-b"
        shared.get_or_compute_shared_embedding(
            changed_checkpoint, str(self.structure), "A", False, compute(3.0)
        )
        self.assertEqual(calls, [1.0, 2.0, 3.0])

    def test_concurrent_tasks_compute_once(self):
        counter = self.root / "counter.txt"
        context = mp.get_context("fork")
        processes = [
            context.Process(
                target=_concurrent_worker,
                args=(str(self.structure), str(counter)),
            )
            for _ in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), ["computed"])

    def test_disabled_cache_has_no_files_or_reuse(self):
        os.environ["STB_SHARED_EMBEDDING_CACHE"] = "0"
        calls = []

        def compute():
            calls.append(1)
            return expected_result()

        shared.get_or_compute_shared_embedding(
            DummyTokenizer(), str(self.structure), "A", False, compute
        )
        shared.get_or_compute_shared_embedding(
            DummyTokenizer(), str(self.structure), "A", False, compute
        )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
