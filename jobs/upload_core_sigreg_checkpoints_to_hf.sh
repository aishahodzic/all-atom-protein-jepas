#!/bin/bash
# Upload the full canonical LeJEPA and protein-centred SIGReg checkpoints to
# msc-thesis-org/metadata, preserving the local run/checkpoints naming.
# Local files are removed only after every remote checkpoint has been verified.

set -euo pipefail

PROJECT_ROOT=/mnt/data/slurm-storage/aishod/projects/la-proteina
STORE_ROOT="$PROJECT_ROOT/store"
HF_BIN=/mnt/data/slurm-storage/aishod/envs/la-proteina/bin/hf
PYTHON_BIN=/mnt/data/slurm-storage/aishod/envs/la-proteina/bin/python
HF_REPO=msc-thesis-org/metadata
HF_SCRATCH="$HOME/tmp/core-sigreg-hf-upload"

if [[ -z "${HF_TOKEN:-}" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi
[[ -n "${HF_TOKEN:-}" ]]
[[ -x "$HF_BIN" && -x "$PYTHON_BIN" ]]

mkdir -p "$HF_SCRATCH"
export HF_HOME="$HF_SCRATCH/hf-home"
export HF_XET_CACHE="$HF_SCRATCH/hf-xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1

runs=(
  canonical_lejepa_v1_100k
  canonical_lejepa_rotations_100k
  canonical_lejepa_residue_iterative_protein_centered_sigreg_2k
  canonical_lejepa_residue_visible_only_protein_centered_sigreg_2k
)

checkpoint_relpaths=(
  canonical_lejepa_v1_100k/checkpoints/step-00010000.ckpt
  canonical_lejepa_v1_100k/checkpoints/step-00100000.ckpt
  canonical_lejepa_rotations_100k/checkpoints/step-00010000.ckpt
  canonical_lejepa_rotations_100k/checkpoints/step-00100000.ckpt
  canonical_lejepa_residue_iterative_protein_centered_sigreg_2k/checkpoints/step-00010000.ckpt
  canonical_lejepa_residue_iterative_protein_centered_sigreg_2k/checkpoints/step-00030000.ckpt
  canonical_lejepa_residue_visible_only_protein_centered_sigreg_2k/checkpoints/step-00010000.ckpt
  canonical_lejepa_residue_visible_only_protein_centered_sigreg_2k/checkpoints/step-00030000.ckpt
)

delete_relpaths=(
  canonical_lejepa_v1_100k/checkpoints/step-00010000.ckpt
  canonical_lejepa_rotations_100k/checkpoints/step-00010000.ckpt
  canonical_lejepa_residue_iterative_protein_centered_sigreg_2k/checkpoints/step-00010000.ckpt
  canonical_lejepa_residue_iterative_protein_centered_sigreg_2k/checkpoints/step-00030000.ckpt
  canonical_lejepa_residue_visible_only_protein_centered_sigreg_2k/checkpoints/step-00010000.ckpt
  canonical_lejepa_residue_visible_only_protein_centered_sigreg_2k/checkpoints/step-00030000.ckpt
)

verify_checkpoint() {
  local remote_path=$1
  local expected_size=$2
  local expected_sha=$3
  "$PYTHON_BIN" - "$HF_REPO" "$remote_path" "$expected_size" "$expected_sha" <<'PY'
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
    f"remote_verified={remote_path} bytes={expected_size} sha256={expected_sha}",
    flush=True,
)
PY
}

for run in "${runs[@]}"; do
  config=$(find "$STORE_ROOT/$run/checkpoints" -maxdepth 1 -type f -name 'exp_config_*.json' -print -quit)
  [[ -n "$config" ]]
  remote_path="$run/checkpoints/$(basename "$config")"
  "$HF_BIN" upload "$HF_REPO" "$config" "$remote_path" \
    --repo-type dataset \
    --commit-message "Add experiment config for $run" \
    --quiet
  echo "config_uploaded=$remote_path"
done

for relpath in "${checkpoint_relpaths[@]}"; do
  local_path="$STORE_ROOT/$relpath"
  [[ -f "$local_path" ]]
  checkpoint_size=$(stat -c %s "$local_path")
  checkpoint_sha=$(sha256sum "$local_path" | awk '{print $1}')
  echo "upload_started=$relpath bytes=$checkpoint_size $(date --iso-8601=seconds)"
  "$HF_BIN" upload "$HF_REPO" "$local_path" "$relpath" \
    --repo-type dataset \
    --commit-message "Add full checkpoint $relpath" \
    --quiet
  verify_checkpoint "$relpath" "$checkpoint_size" "$checkpoint_sha"
done

# Preserve both reported 100k canonical checkpoints. Remove only the uploaded
# 10k canonical checkpoints and the four protein-centred SIGReg checkpoints.
for relpath in "${delete_relpaths[@]}"; do
  local_path="$STORE_ROOT/$relpath"
  if fuser "$local_path" >/dev/null 2>&1; then
    echo "Refusing to remove checkpoint in use: $local_path" >&2
    exit 1
  fi
done

for relpath in "${delete_relpaths[@]}"; do
  local_path="$STORE_ROOT/$relpath"
  unlink "$local_path"
  [[ ! -e "$local_path" ]]
  echo "local_removed=$relpath"
done

echo "upload_verification_and_cleanup_complete=yes"
du -sh "$STORE_ROOT"
df -h /mnt/data
