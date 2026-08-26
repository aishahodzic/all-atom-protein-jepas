"""Content identities and immutable records for publication evaluation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
CACHE_METADATA_SCHEMA_VERSION = 2


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Return a file's SHA-256 from its bytes, independent of its pathname."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path, *, include_sha256=True):
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
    }
    if include_sha256:
        identity["sha256"] = sha256_file(resolved)
    return identity


def artifact_identity(path):
    """Content identity for either a regular file or checkpoint directory."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_file():
        return {"kind": "file", **file_identity(resolved)}
    if not resolved.is_dir():
        raise ValueError(f"Unsupported artifact type: {resolved}")

    records = []
    total_size = 0
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        size = child.stat().st_size
        total_size += size
        records.append(
            {
                "relative_path": child.relative_to(resolved).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(child),
            }
        )
    return {
        "kind": "directory",
        "path": str(resolved),
        "file_count": len(records),
        "size_bytes": total_size,
        "sha256": canonical_json_sha256(records),
    }


def atomic_write_json(path, payload):
    """Atomically replace a mutable JSON sidecar (used only for token caches)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_immutable_json(path, payload):
    """Atomically publish a JSON file exactly once; never overwrite it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses an existing destination;
        # unlike rename/replace, it can never overwrite an earlier manifest.
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return str(destination)


def git_identity(repository_dir):
    """Capture the checked-out commit plus a hash of tracked/untracked status."""

    repository_dir = str(Path(repository_dir).resolve())

    def run(*args):
        return subprocess.run(
            ["git", "-C", repository_dir, *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    try:
        commit = run("rev-parse", "HEAD").decode().strip()
        status = run("status", "--porcelain=v1", "-z", "--untracked-files=all")
        tracked_diff = run("diff", "--binary", "HEAD", "--")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}

    content_digest = hashlib.sha256()
    content_digest.update(tracked_diff)
    untracked_records = []
    for raw_entry in status.split(b"\0"):
        if not raw_entry.startswith(b"?? "):
            continue
        relative = raw_entry[3:].decode("utf-8", errors="surrogateescape")
        path = Path(repository_dir) / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        record = {"path": relative, "size_bytes": size}
        # Source/config files are small. Avoid re-reading multi-GB artifacts
        # that happen to be untracked; their explicit artifact identities are
        # recorded elsewhere in the manifest.
        if size <= 16 * 1024 * 1024:
            record["sha256"] = sha256_file(path)
        else:
            record["sha256"] = None
            record["content_hash_skipped_over_bytes"] = 16 * 1024 * 1024
        untracked_records.append(record)
    content_digest.update(
        json.dumps(untracked_records, sort_keys=True, separators=(",", ":")).encode()
    )

    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        # Avoid embedding a potentially enormous diff while still tying the
        # manifest to the actual tracked diff and small untracked contents.
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "status_entry_count": len([entry for entry in status.split(b"\0") if entry]),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "working_tree_content_sha256": content_digest.hexdigest(),
        "untracked_files": untracked_records,
    }


def runtime_identity():
    environment_keys = (
        "EVALUATION_RUN_ID",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "JARVIS_JOB_ID",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "environment": {
            key: os.environ[key] for key in environment_keys if key in os.environ
        },
    }


def scientific_runtime_versions():
    """Versions of runtimes that can change parsing or encoder numerics."""

    distributions = (
        "torch",
        "torch-geometric",
        "graphein",
        "biotite",
        "numpy",
        "scipy",
    )
    versions = {"python": platform.python_version()}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def safe_path_component(value):
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    )
