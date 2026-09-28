#!/bin/bash
# Move canonical LeJEPA layer-state and STB shared caches to the metadata
# dataset. Local data is removed only after remote size/SHA256 verification.

set -euo pipefail

PROJECT_ROOT=/mnt/data/slurm-storage/aishod/projects/la-proteina
RUN_ROOT="$PROJECT_ROOT/store/canonical_lejepa_v1_100k"
EVAL_ROOT="$RUN_ROOT/evaluation"
LAYER_STATES="$EVAL_ROOT/full_context_layer_states"
STB_CACHE="$EVAL_ROOT/stb_layerwise_shared_cache"
HF_BIN=/mnt/data/slurm-storage/aishod/envs/la-proteina/bin/hf
PYTHON_BIN=/mnt/data/slurm-storage/aishod/envs/la-proteina/bin/python
HF_REPO=msc-thesis-org/metadata
REMOTE_ROOT=canonical_lejepa_v1_100k/evaluation

if [[ -z "${HF_TOKEN:-}" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi
[[ -n "${HF_TOKEN:-}" ]]
[[ -x "$HF_BIN" && -x "$PYTHON_BIN" ]]
[[ -d "$LAYER_STATES" && -d "$STB_CACHE/v1" ]]

expected_namespaces=(
  4d25cb8c448cdc3ff041927199c07530aace3d1ed2ed3c8f459bb4690281b749
  9a1cdddad0cc087b9198dcf8de033ab2624b34099f1dba33f0afe9e0ceb96acc
  ddc7a89108778be23d66d432ed3e58e328a98d60c8f0a102569dafe880baaee9
)

mapfile -t actual_namespaces < <(
  find "$STB_CACHE/v1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)
[[ "${actual_namespaces[*]}" == "${expected_namespaces[*]}" ]]

in_use=$(find "$LAYER_STATES" "$STB_CACHE" -type f -print0 | xargs -0 -r fuser 2>/dev/null || true)
[[ -z "$in_use" ]] || {
  echo "Refusing to move cache files that are in use: $in_use" >&2
  exit 1
}

mkdir -p "$HOME/tmp"
TRANSFER_SCRATCH=$(mktemp -d "$HOME/tmp/canonical-layer-cache-hf-XXXXXX")
cleanup_scratch() {
  find "$TRANSFER_SCRATCH" -type f -delete 2>/dev/null || true
  find "$TRANSFER_SCRATCH" -type l -delete 2>/dev/null || true
  find "$TRANSFER_SCRATCH" -depth -type d -empty -delete 2>/dev/null || true
}
trap cleanup_scratch EXIT HUP INT TERM

export HF_HOME="$TRANSFER_SCRATCH/hf-home"
export HF_XET_CACHE="$TRANSFER_SCRATCH/hf-xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1

remote_layer_root="$REMOTE_ROOT/full_context_layer_states"
echo "layer_states_upload_started=$(date --iso-8601=seconds)"
"$HF_BIN" upload "$HF_REPO" "$LAYER_STATES" "$remote_layer_root" \
  --repo-type dataset \
  --commit-message "Archive canonical LeJEPA full-context layer states" \
  --quiet

"$PYTHON_BIN" - "$HF_REPO" "$LAYER_STATES" "$remote_layer_root" <<'PY'
import hashlib
import pathlib
import sys

from huggingface_hub import HfApi

repo, local_root_arg, remote_root = sys.argv[1:]
local_root = pathlib.Path(local_root_arg)
local = {
    f"{remote_root}/{path.relative_to(local_root).as_posix()}": path
    for path in local_root.rglob("*")
    if path.is_file() and not path.is_symlink()
}
entries = HfApi().list_repo_tree(
    repo,
    path_in_repo=remote_root,
    recursive=True,
    expand=True,
    repo_type="dataset",
    token=True,
)
remote = {
    entry.path: entry
    for entry in entries
    if getattr(entry, "size", None) is not None
}
assert set(local) == set(remote), (
    sorted(set(local) - set(remote)),
    sorted(set(remote) - set(local)),
)

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

verified_bytes = 0
for remote_path, local_path in sorted(local.items()):
    entry = remote[remote_path]
    size = local_path.stat().st_size
    assert entry.size == size, (remote_path, entry.size, size)
    if size >= 10 * 1024 * 1024:
        assert entry.lfs is not None, (remote_path, "missing LFS metadata")
        digest = sha256_file(local_path)
        assert entry.lfs.sha256 == digest, (remote_path, entry.lfs.sha256, digest)
    verified_bytes += size
print(
    f"layer_states_remote_verified=yes files={len(local)} bytes={verified_bytes}",
    flush=True,
)
PY

find "$LAYER_STATES" -type f -delete
find "$LAYER_STATES" -type l -delete
find "$LAYER_STATES" -depth -type d -empty -delete
[[ ! -e "$LAYER_STATES" ]]
echo "layer_states_local_removed=yes"

for namespace in "${expected_namespaces[@]}"; do
  namespace_path="$STB_CACHE/v1/$namespace"
  archive="$TRANSFER_SCRATCH/canonical-lejepa-v1-stb-cache-v1-$namespace.tar"
  remote_path="$REMOTE_ROOT/stb_layerwise_shared_cache/archives/$(basename "$archive")"
  [[ -d "$namespace_path" ]]

  echo "namespace_archive_started=$namespace $(date --iso-8601=seconds)"
  tar --exclude='*.lock' -cf "$archive" -C "$STB_CACHE" "v1/$namespace"
  archive_size=$(stat -c %s "$archive")
  archive_sha=$(sha256sum "$archive" | awk '{print $1}')

  "$HF_BIN" upload "$HF_REPO" "$archive" "$remote_path" \
    --repo-type dataset \
    --commit-message "Archive canonical LeJEPA STB cache namespace $namespace" \
    --quiet

  "$PYTHON_BIN" - "$HF_REPO" "$remote_path" "$archive_size" "$archive_sha" <<'PY'
import sys
from huggingface_hub import HfApi

repo, remote_path, expected_size, expected_sha = sys.argv[1:]
entries = HfApi().get_paths_info(
    repo,
    [remote_path],
    repo_type="dataset",
    expand=True,
    token=True,
)
assert len(entries) == 1, (remote_path, entries)
entry = entries[0]
assert entry.size == int(expected_size), (remote_path, entry.size, expected_size)
assert entry.lfs is not None, (remote_path, "missing LFS metadata")
assert entry.lfs.sha256 == expected_sha, (
    remote_path,
    entry.lfs.sha256,
    expected_sha,
)
print(
    f"namespace_remote_verified={remote_path} bytes={expected_size} sha256={expected_sha}",
    flush=True,
)
PY

  find "$namespace_path" -type f -delete
  find "$namespace_path" -type l -delete
  find "$namespace_path" -depth -type d -empty -delete
  [[ ! -e "$namespace_path" ]]
  unlink "$archive"
  echo "namespace_local_removed=$namespace"
done

rmdir "$STB_CACHE/v1"
rmdir "$STB_CACHE"
echo "canonical_layer_cache_move_complete=yes"
du -sh "$RUN_ROOT" "$PROJECT_ROOT"
df -h /mnt/data
