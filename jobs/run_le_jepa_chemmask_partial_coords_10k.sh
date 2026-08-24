#!/usr/bin/env bash
set -euo pipefail

# Portable Jarvis launcher for the matched 10k old-vs-new predictor pilot.
# Usage:
#   bash jobs/run_le_jepa_chemmask_partial_coords_10k.sh old
#   bash jobs/run_le_jepa_chemmask_partial_coords_10k.sh new
#   SELF_EXCLUDE=true bash jobs/run_le_jepa_chemmask_partial_coords_10k.sh new

predictor_variant="${1:-old}"
case "$predictor_variant" in
  old)
    predictor_flag=false
    ;;
  new)
    predictor_flag=true
    ;;
  *)
    echo "usage: $0 {old|new}" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
num_workers="${NUM_WORKERS:-16}"
self_exclude="${SELF_EXCLUDE:-false}"
case "$self_exclude" in
  true) exclusion_name=selfexclude ;;
  false) exclusion_name=selfaccess ;;
  *)
    echo "SELF_EXCLUDE must be true or false" >&2
    exit 2
    ;;
esac
run_name="le_jepa_chemmask_aa_partial50_${predictor_variant}_${exclusion_name}_10k"

cd "$repo_root"
exec "$python_bin" proteinfoundation/jepa/train.py \
  run_name_="$run_name" \
  +single=true \
  +masking_mode=chemmask_partial_coordinates \
  +atom_mask_pct=0.5 \
  +partial_coordinate_self_exclude="$self_exclude" \
  +partial_coordinate_diagnostics=true \
  baseline_run_predictor="$predictor_flag" \
  baseline_run_ema=false \
  baseline_run_layernorm=false \
  opt.lr=0.0001 \
  +opt.max_steps=10000 \
  opt.val_check_interval=1000 \
  log.checkpoint_every_n_steps=5000 \
  dataset.datamodule.num_workers="$num_workers"
