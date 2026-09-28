"""Dedicated STB entrypoint for the supervised 768D-to-8D bottleneck study."""

from __future__ import annotations

import os
from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from supervised_bottleneck import install_supervised_bottleneck  # noqa: E402


dimension = os.environ.get("STB_SUPERVISED_BOTTLENECK_DIM")
if dimension != "8":
    raise RuntimeError(
        "run_supervised_bottleneck_task.py requires "
        "STB_SUPERVISED_BOTTLENECK_DIM=8"
    )

install_supervised_bottleneck()

import run_supervised_task as task_runner  # noqa: E402


_original_source_file_identities = task_runner._source_file_identities


def _source_file_identities(repository_dir):
    identities = _original_source_file_identities(repository_dir)
    identities["supervised_bottleneck"] = task_runner.file_identity(
        _SRC_DIR / "supervised_bottleneck.py"
    )
    identities["supervised_bottleneck_entrypoint"] = task_runner.file_identity(
        Path(__file__).resolve()
    )
    return identities


task_runner._source_file_identities = _source_file_identities

branch = os.environ.get("EXPERIMENT_LABEL")
if branch:
    suffix = "-supervised-bottleneck8"
    if not branch.endswith(suffix):
        os.environ["EXPERIMENT_LABEL"] = f"{branch}{suffix}"


if __name__ == "__main__":
    task_runner.main()

