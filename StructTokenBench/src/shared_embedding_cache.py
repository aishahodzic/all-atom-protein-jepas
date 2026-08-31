"""Content-addressed, cross-task cache for frozen LaProteina embeddings.

StructTokenBench's existing cache is intentionally task/split/label specific.
That is the authoritative evaluation artifact and remains unchanged.  This
module adds a lower-level cache around ``WrappedLaProteinaTokenizer``'s frozen
encoder call so identical structures can be encoded once per checkpoint and
then joined to each task's labels by the existing dataset code.

The layer is operational only: it must produce exactly the same return tuple
as the original encoder method.  It is opt-in for direct Python invocations via
``STB_SHARED_EMBEDDING_CACHE=1``; ``run_eval.sh`` enables it for new runs.
"""

from __future__ import annotations

import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import time
from typing import Callable

import numpy as np
import torch

from evaluation_manifest import (
    canonical_json_sha256,
    file_identity,
    scientific_runtime_versions,
)


SHARED_EMBEDDING_CACHE_SCHEMA_VERSION = 1
_ENABLED_ENV = "STB_SHARED_EMBEDDING_CACHE"
_ROOT_ENV = "STB_SHARED_EMBEDDING_CACHE_ROOT"
_ORIGINAL_METHOD_ATTR = "_stb_uncached_encode_structure_v1"
_ORIGINAL_DATASET_CACHE_ATTR = "_stb_unshared_cache_all_tokenized_v1"
_ORIGINAL_RESOLVE_WORKER_ATTR = "_stb_unshared_resolve_and_parse_worker_v1"
_WORKER_TOKENIZER_IDENTITY_ENV = "STB_SHARED_EMBEDDING_TOKENIZER_IDENTITY_JSON"


def shared_embedding_cache_enabled() -> bool:
    value = os.environ.get(_ENABLED_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_ENABLED_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def _cache_root() -> Path:
    raw = os.environ.get(_ROOT_ENV)
    if not raw:
        raise RuntimeError(
            f"{_ROOT_ENV} is required when {_ENABLED_ENV}=1; run_eval.sh "
            "sets it below the STB release-data directory"
        )
    return Path(raw).expanduser().resolve()


@lru_cache(maxsize=131072)
def _sha256_for_unchanged_file(path: str, size: int, mtime_ns: int) -> str:
    # size/mtime are part of the memoization key, so an in-place structure-file
    # change cannot silently reuse a digest computed for the previous bytes.
    del size, mtime_ns
    return file_identity(path)["sha256"]


def _structure_file_sha256(path: str) -> str:
    stat = os.stat(path)
    return _sha256_for_unchanged_file(
        str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns
    )


@lru_cache(maxsize=1)
def _encoder_implementation_identity() -> dict:
    """Hash only code that can change the uncached encoder output.

    Dataset orchestration is excluded deliberately: changing how this cache is
    read must not invalidate the task-specific caches when the frozen encoder,
    graph construction, and feature code are unchanged.
    """

    src_dir = Path(__file__).resolve().parent
    repository_dir = src_dir.parents[1]
    paths = {
        src_dir / "tokenizer.py",
        src_dir / "protein_chain.py",
        src_dir / "laproteina_checkpoint.py",
        src_dir / "laproteina_eval_policy.py",
    }
    for source_root in (
        repository_dir / "proteinfoundation",
        repository_dir / "openfold",
    ):
        paths.update(source_root.rglob("*.py"))
    identities = {
        path.relative_to(repository_dir).as_posix(): file_identity(path)["sha256"]
        for path in sorted(paths)
        if path.exists()
    }
    return {
        "files": identities,
        "sha256": canonical_json_sha256(identities),
    }


def _tokenizer_identity(tokenizer, *, use_sequence: bool) -> dict:
    implementation = _encoder_implementation_identity()
    identity = {
        "schema_version": SHARED_EMBEDDING_CACHE_SCHEMA_VERSION,
        "checkpoint_sha256": tokenizer.checkpoint_sha256,
        "checkpoint_type": tokenizer.checkpoint_type,
        "encoder_source": tokenizer.encoder_source,
        "encoder_state_prefix": tokenizer.encoder_state_prefix,
        "encoder_config_sha256": tokenizer.encoder_cfg_sha256,
        "latent_z_dim": int(tokenizer.latent_z_dim),
        "evaluation_protocol": tokenizer.evaluation_protocol,
        "use_sequence": bool(use_sequence),
        "encoder_implementation_sha256": implementation["sha256"],
        "scientific_runtime_versions": scientific_runtime_versions(),
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _structure_identity(pdb_path: str, chain_id: str) -> dict:
    identity = {
        "file_sha256": _structure_file_sha256(pdb_path),
        # Chain IDs are case-sensitive.  Atlas intentionally uses one blank.
        "chain_id": str(chain_id),
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _cache_paths(tokenizer_identity: dict, structure_identity: dict) -> tuple[Path, Path]:
    namespace = tokenizer_identity["identity_sha256"]
    structure = structure_identity["identity_sha256"]
    directory = (
        _cache_root()
        / f"v{SHARED_EMBEDDING_CACHE_SCHEMA_VERSION}"
        / namespace
        / structure[:2]
    )
    cache_path = directory / f"{structure}.pt"
    return cache_path, cache_path.with_suffix(".lock")


def _cache_entry_exists(
    tokenizer_identity: dict, pdb_path: str, chain_id: str
) -> bool:
    structure_identity = _structure_identity(pdb_path, chain_id)
    cache_path, _ = _cache_paths(tokenizer_identity, structure_identity)
    return cache_path.exists()


def _cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


def _validate_result(result) -> tuple:
    if not isinstance(result, (list, tuple)) or len(result) != 3:
        raise ValueError("LaProteina encoder must return (embeddings, residue_index, sequence)")
    token_ids, residue_index, seqs = result
    if not torch.is_tensor(token_ids):
        raise TypeError(f"Expected tensor embeddings, got {type(token_ids).__name__}")
    if len(token_ids) != len(residue_index) or len(token_ids) != len(seqs):
        raise ValueError(
            "Shared embedding result has inconsistent lengths: "
            f"embeddings={len(token_ids)}, residue_index={len(residue_index)}, "
            f"sequence={len(seqs)}"
        )
    return tuple(_cpu_clone(item) for item in result)


def _load_cache(cache_path: Path, expected_identity: dict):
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("identity") != expected_identity:
            return None
        return _validate_result(payload["result"])
    except Exception:
        # Preserve bad bytes for diagnosis while removing them from the active
        # namespace.  Holding the per-entry lock makes this rename race-free.
        quarantine = cache_path.with_name(
            f"{cache_path.name}.corrupt-{int(time.time())}-{os.getpid()}"
        )
        try:
            os.replace(cache_path, quarantine)
        except FileNotFoundError:
            pass
        return None


def _atomic_save(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    )
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
        directory_fd = os.open(cache_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _stats(tokenizer) -> dict:
    if not hasattr(tokenizer, "_stb_shared_embedding_cache_stats"):
        tokenizer._stb_shared_embedding_cache_stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
        }
    return tokenizer._stb_shared_embedding_cache_stats


def _record(tokenizer, key: str) -> None:
    stats = _stats(tokenizer)
    stats[key] += 1
    total = stats["hits"] + stats["misses"]
    if total == 1 or total % 1000 == 0:
        print(
            "[shared-embedding-cache] "
            f"requests={total} hits={stats['hits']} misses={stats['misses']} "
            f"writes={stats['writes']}",
            flush=True,
        )


def get_or_compute_shared_embedding(
    tokenizer,
    pdb_path: str,
    chain_id: str,
    use_sequence: bool,
    compute: Callable[[], tuple],
) -> tuple:
    """Return the exact frozen-encoder tuple, sharing it across STB tasks."""

    if not shared_embedding_cache_enabled() or tokenizer.evaluation_protocol is None:
        return compute()

    tokenizer_identity = _tokenizer_identity(tokenizer, use_sequence=use_sequence)
    structure_identity = _structure_identity(pdb_path, chain_id)
    full_identity = {
        "schema_version": SHARED_EMBEDDING_CACHE_SCHEMA_VERSION,
        "tokenizer": tokenizer_identity,
        "structure": structure_identity,
    }
    cache_path, lock_path = _cache_paths(tokenizer_identity, structure_identity)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # flock is released by the kernel if a process crashes.  Unlike a handmade
    # O_EXCL sentinel, it cannot leave a stale lock that blocks future jobs.
    with open(lock_path, "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached = _load_cache(cache_path, full_identity)
        if cached is not None:
            _record(tokenizer, "hits")
            return cached

        _record(tokenizer, "misses")
        result = _validate_result(compute())
        payload = {
            "schema_version": SHARED_EMBEDDING_CACHE_SCHEMA_VERSION,
            "identity": full_identity,
            "created_at_unix": time.time(),
            "created_by": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
            },
            "result": result,
        }
        _atomic_save(cache_path, payload)
        _stats(tokenizer)["writes"] += 1
        return tuple(_cpu_clone(item) for item in result)


def install_shared_embedding_cache() -> None:
    """Idempotently wrap only LaProteina's frozen full-structure encoder."""

    import dataset.base as dataset_base
    from tokenizer import WrappedLaProteinaTokenizer

    if hasattr(WrappedLaProteinaTokenizer, _ORIGINAL_METHOD_ATTR):
        return
    original = WrappedLaProteinaTokenizer.encode_structure
    setattr(WrappedLaProteinaTokenizer, _ORIGINAL_METHOD_ATTR, original)

    def cached_encode_structure(self, pdb_path: str, chain_id: str, use_sequence=False):
        def compute():
            return original(self, pdb_path, chain_id, use_sequence)

        result = get_or_compute_shared_embedding(
            self, pdb_path, chain_id, use_sequence, compute
        )
        # cache_all_tokenized preloads parsed graphs into RAM.  The original
        # method pops each graph when encoding; a disk-cache hit must do the
        # same or an all-task run retains thousands of unused graphs.
        graph_cache = getattr(self, "_graph_cache", None)
        if graph_cache is not None:
            graph_cache.pop((pdb_path, chain_id), None)
        return result

    WrappedLaProteinaTokenizer.encode_structure = cached_encode_structure

    # On a shared-cache hit, avoid eagerly parsing/loading a graph that the
    # encoder wrapper will never consume.  This preserves the original worker
    # result contract: graph=None is already a supported metadata-only result.
    original_worker = dataset_base._resolve_and_parse_worker
    setattr(dataset_base, _ORIGINAL_RESOLVE_WORKER_ATTR, original_worker)

    def shared_aware_resolve_worker(
        pdb_path,
        chain_id,
        is_atlas,
        pdb_data_dir,
        min_length=None,
        max_length=None,
        exclude_ids=frozenset(),
        remove_non_standard_residues=False,
        experiment_types=frozenset(),
        best_resolution=None,
        worst_resolution=None,
        apply_checkpoint_training_filters=True,
    ):
        serialized_identity = os.environ.get(_WORKER_TOKENIZER_IDENTITY_ENV)
        if (
            serialized_identity
            and shared_embedding_cache_enabled()
            and not apply_checkpoint_training_filters
        ):
            try:
                resolved_chain_id = (
                    " "
                    if is_atlas
                    else dataset_base.resolve_chain_id_cached(
                        pdb_path, chain_id, pdb_data_dir
                    )
                )
                tokenizer_identity = json.loads(serialized_identity)
                if _cache_entry_exists(
                    tokenizer_identity, pdb_path, resolved_chain_id
                ):
                    # ``matches=True`` is correct for the locked protocol: its
                    # checkpoint-training selectors are deliberately disabled.
                    # ``was_pretrained_hit=True`` only suppresses a misleading
                    # "new graph processed" message; no graph was needed.
                    return resolved_chain_id, None, True, True, None
            except Exception:
                # The original worker produces the authoritative parse/failure
                # result and is the safe fallback for any cache-index problem.
                pass
        return original_worker(
            pdb_path,
            chain_id,
            is_atlas,
            pdb_data_dir,
            min_length,
            max_length,
            exclude_ids,
            remove_non_standard_residues,
            experiment_types,
            best_resolution,
            worst_resolution,
            apply_checkpoint_training_filters,
        )

    dataset_base._resolve_and_parse_worker = shared_aware_resolve_worker

    original_cache_all = dataset_base.BaseDataset.cache_all_tokenized
    setattr(dataset_base.BaseDataset, _ORIGINAL_DATASET_CACHE_ATTR, original_cache_all)

    def shared_aware_cache_all_tokenized(dataset):
        previous = os.environ.get(_WORKER_TOKENIZER_IDENTITY_ENV)
        tokenizer = getattr(dataset, "tokenizer", None)
        should_publish_identity = (
            shared_embedding_cache_enabled()
            and isinstance(tokenizer, WrappedLaProteinaTokenizer)
            and tokenizer.evaluation_protocol is not None
        )
        if should_publish_identity:
            identity = _tokenizer_identity(
                tokenizer, use_sequence=bool(dataset.use_sequence)
            )
            os.environ[_WORKER_TOKENIZER_IDENTITY_ENV] = json.dumps(
                identity, sort_keys=True, separators=(",", ":")
            )
        try:
            return original_cache_all(dataset)
        finally:
            if previous is None:
                os.environ.pop(_WORKER_TOKENIZER_IDENTITY_ENV, None)
            else:
                os.environ[_WORKER_TOKENIZER_IDENTITY_ENV] = previous

    dataset_base.BaseDataset.cache_all_tokenized = shared_aware_cache_all_tokenized


def shared_embedding_cache_manifest_record(tokenizer) -> dict:
    enabled = shared_embedding_cache_enabled()
    record = {
        "enabled": enabled,
        "schema_version": SHARED_EMBEDDING_CACHE_SCHEMA_VERSION,
        "semantics": "frozen-full-structure-embedding-only; labels-attached-afterward",
    }
    if enabled:
        # BaseDataset always requests sequence metadata from tokenizers, even
        # when the supervised probe itself does not consume sequence.  Record
        # the namespace actually used by every STB dataset call.
        identity = _tokenizer_identity(tokenizer, use_sequence=True)
        record.update(
            {
                "root": str(_cache_root()),
                "tokenizer_namespace_sha256": identity["identity_sha256"],
                "stats": dict(_stats(tokenizer)),
            }
        )
    return record
