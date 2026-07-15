#!/bin/bash
# Rebuilds the py3.10 conda env from scratch for training (jepa / partial_autoencoder / etc).
# Captures the fixes needed to get proteinfoundation/jepa/train.py running on a fresh
# GPU instance with CUDA 12.6 toolkit installed (e.g. a fresh Jarvis Labs box).
#
# Before running: export DATA_PATH=/path/to/data and export HF_TOKEN=<token provided
# to you>. Check `nvcc --version` first; if it isn't cu126, swap the cu126 references
# in install_env_deps.sh for the matching build.
#
# Usage: source setup_env.sh
# (must be sourced, not `bash`-ed, so the `conda activate py3.10` below persists in
# your shell after the script finishes)
#
# This only activates the conda env directly in the shell. The actual 
# install steps run in install_env_deps.sh as a real subprocess.
_setup_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

echo "== Activating py3.10 conda environment =="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py3.10

bash "$_setup_env_dir/install_env_deps.sh"
_setup_env_status=$?
unset _setup_env_dir

if [ "$_setup_env_status" -ne 0 ]; then
    echo "== install_env_deps.sh failed (exit $_setup_env_status); py3.10 is still active, fix the error and rerun install_env_deps.sh =="
fi
unset _setup_env_status
