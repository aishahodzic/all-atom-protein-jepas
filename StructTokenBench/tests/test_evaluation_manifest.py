import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluation_manifest import (  # noqa: E402
    artifact_identity,
    canonical_json_sha256,
    file_identity,
    write_immutable_json,
)


class EvaluationManifestTest(unittest.TestCase):
    def test_file_identity_follows_bytes_not_staging_path(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.ckpt"
            second = Path(directory) / "nested" / "second.ckpt"
            second.parent.mkdir()
            first.write_bytes(b"same checkpoint bytes")
            second.write_bytes(b"same checkpoint bytes")

            first_identity = file_identity(first)
            second_identity = file_identity(second)
            self.assertNotEqual(first_identity["path"], second_identity["path"])
            self.assertEqual(first_identity["sha256"], second_identity["sha256"])

            second.write_bytes(b"changed checkpoint bytes")
            self.assertNotEqual(
                first_identity["sha256"], file_identity(second)["sha256"]
            )

    def test_immutable_writer_refuses_overwrite_and_preserves_first_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.json"
            write_immutable_json(destination, {"run": 1})
            original = destination.read_bytes()

            with self.assertRaises(FileExistsError):
                write_immutable_json(destination, {"run": 2})

            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(json.loads(original), {"run": 1})
            self.assertEqual(os.stat(destination).st_mode & 0o222, 0)

    def test_non_finite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_immutable_json(
                    Path(directory) / "manifest.json", {"bad": float("nan")}
                )

    def test_canonical_hash_ignores_dictionary_insertion_order(self):
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": 2}),
            canonical_json_sha256({"b": 2, "a": 1}),
        )

    def test_directory_artifact_identity_covers_every_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "probe.ckpt"
            (checkpoint / "nested").mkdir(parents=True)
            (checkpoint / "model.pt").write_bytes(b"weights")
            (checkpoint / "nested" / "state.json").write_text("{}")

            first = artifact_identity(checkpoint)
            (checkpoint / "nested" / "state.json").write_text('{"step": 1}')
            second = artifact_identity(checkpoint)

            self.assertEqual(first["kind"], "directory")
            self.assertEqual(first["file_count"], 2)
            self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
