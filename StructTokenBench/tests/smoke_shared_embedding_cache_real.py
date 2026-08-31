"""One-structure real-encoder equivalence smoke for the shared STB cache.

This is intentionally separate from the fast unit suite because it loads the
full 130M LaProteina encoder checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import types

import numpy as np
import torch

# The login node has no active Triton/CUDA driver.  DeepSpeed is only an
# incidental import of StructTokenBench's unused VQ-VAE baseline here; marking
# Triton unavailable keeps this CPU-only encoder smoke from initializing it.
sys.modules["triton"] = None
# pstbench deliberately omits the training-only W&B client; AutoEncoder is
# imported for checkpoint compatibility although this smoke loads JEPA only.
_wandb_stub = types.ModuleType("wandb")
_wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
sys.modules.setdefault("wandb", _wandb_stub)

import dataset.base as dataset_base
import shared_embedding_cache as shared
from shared_embedding_cache import install_shared_embedding_cache
from tokenizer import WrappedLaProteinaTokenizer

# Keep this at module scope: spawn-started STB workers re-import their main
# module, so production's run_supervised_task.py installs the hook the same way.
install_shared_embedding_cache()


def assert_equal(reference, candidate):
    assert torch.equal(reference[0].cpu(), candidate[0].cpu())
    assert np.array_equal(np.asarray(reference[1]), np.asarray(candidate[1]))
    assert list(reference[2]) == list(candidate[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--chain", default="A")
    args = parser.parse_args()

    tokenizer = WrappedLaProteinaTokenizer(
        device=torch.device("cpu"),
        pretrained_ckpt_path=args.checkpoint,
        checkpoint_type="jepa",
        encoder_cfg_path=args.encoder_config,
        encoder_source="online",
        evaluation_max_length=None,
        evaluation_cache_tag="upstream-l600-v1",
        evaluation_protocol="upstream-l600-v1",
    )

    os.environ["STB_SHARED_EMBEDDING_CACHE"] = "0"
    start = time.monotonic()
    reference = tokenizer.encode_structure(args.structure, args.chain, True)
    uncached_seconds = time.monotonic() - start

    with tempfile.TemporaryDirectory(prefix="stb-shared-real-") as cache_root:
        os.environ["STB_SHARED_EMBEDDING_CACHE"] = "1"
        os.environ["STB_SHARED_EMBEDDING_CACHE_ROOT"] = cache_root

        start = time.monotonic()
        first = tokenizer.encode_structure(args.structure, args.chain, True)
        first_seconds = time.monotonic() - start
        assert_equal(reference, first)

        start = time.monotonic()
        second = tokenizer.encode_structure(args.structure, args.chain, True)
        hit_seconds = time.monotonic() - start
        assert_equal(reference, second)

        cache_files = list(Path(cache_root).rglob("*.pt"))
        assert len(cache_files) == 1, cache_files
        stats = tokenizer._stb_shared_embedding_cache_stats
        assert stats == {"hits": 1, "misses": 1, "writes": 1}, stats

        # A later task's precompute worker should see the shared artifact and
        # skip graph loading/parsing entirely (graph=None), not merely skip the
        # final encoder forward.
        worker_identity = shared._tokenizer_identity(
            tokenizer, use_sequence=True
        )
        os.environ[shared._WORKER_TOKENIZER_IDENTITY_ENV] = __import__(
            "json"
        ).dumps(worker_identity)
        resolved, graph, matches, _, error = dataset_base._resolve_and_parse_worker(
            args.structure,
            args.chain,
            False,
            str(Path(args.structure).parent.parent),
            apply_checkpoint_training_filters=False,
        )
        assert resolved == args.chain
        assert graph is None
        assert matches is True
        assert error is None

        # STB resolves structures in a spawned process pool.  Confirm a fresh
        # spawned interpreter installs the same worker hook and takes the
        # metadata-only path, rather than relying on inherited fork state.
        worker_args = (
            args.structure,
            args.chain,
            False,
            str(Path(args.structure).parent.parent),
            None,
            None,
            frozenset(),
            False,
            frozenset(),
            None,
            None,
            False,
        )
        context = mp.get_context("spawn")
        with context.Pool(processes=1) as pool:
            spawned_result = pool.apply(
                dataset_base._resolve_and_parse_worker_star, (worker_args,)
            )
        spawned_resolved, spawned_graph, spawned_matches, _, spawned_error = (
            spawned_result
        )
        assert spawned_resolved == args.chain
        assert spawned_graph is None
        assert spawned_matches is True
        assert spawned_error is None

    print(
        "REAL_SHARED_CACHE_SMOKE_OK "
        f"shape={tuple(reference[0].shape)} "
        f"uncached_seconds={uncached_seconds:.3f} "
        f"first_cached_seconds={first_seconds:.3f} "
        f"hit_seconds={hit_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
