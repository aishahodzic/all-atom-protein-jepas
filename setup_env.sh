#!/bin/bash
# Creates/updates the micromamba environment used for training on CoreWeave.
#
# Before running, export DATA_PATH and, for the initial private dataset download,
# HF_TOKEN. The environment itself is shared by all experiment worktrees.
#
# Usage: source setup_env.sh
# (must be sourced so the micromamba activation persists in the current shell).
_setup_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_setup_env_name="${LAPROTEINA_ENV_NAME:-laproteina_env}"

if ! command -v micromamba >/dev/null 2>&1; then
    echo "micromamba is required but was not found on PATH" >&2
    return 1 2>/dev/null || exit 1
fi

_setup_shell="${SHELL##*/}"
case "$_setup_shell" in
    bash|zsh) ;;
    *) _setup_shell="bash" ;;
esac
eval "$(micromamba shell hook --shell "$_setup_shell")"
if ! micromamba env list | awk 'NR > 1 {print $1}' | grep -qx "$_setup_env_name"; then
    echo "== Creating $_setup_env_name (Python 3.11) =="
    micromamba create -y -n "$_setup_env_name" -c conda-forge python=3.11 pip
fi

echo "== Activating $_setup_env_name =="
micromamba activate "$_setup_env_name"

bash "$_setup_env_dir/install_env_deps.sh"
_setup_env_status=$?

if [ "$_setup_env_status" -ne 0 ]; then
    echo "== install_env_deps.sh failed (exit $_setup_env_status); $_setup_env_name remains active =="
fi
unset _setup_env_dir _setup_env_name _setup_env_status _setup_shell
