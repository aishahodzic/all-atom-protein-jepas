#!/usr/bin/env bash
set -euo pipefail


experiment="${1:-}"
predictor_variant="${2:-}"
case "$experiment" in
  sc70)
    masking_mode=chemmask_sidechain_partial
    mask_pct=0.7
    experiment_name=aa_bb_sc70
    ;;
  noaa-sc50)
    masking_mode=chemmask_no_aa_sidechain_partial
    mask_pct=0.5
    experiment_name=noaa_bb_sc50
    ;;
  *)
    echo "usage: $0 {sc70|noaa-sc50} {old|new}" >&2
    exit 2
    ;;
esac

case "$predictor_variant" in
  old) predictor_flag=false ;;
  new) predictor_flag=true ;;
  *)
    echo "usage: $0 {sc70|noaa-sc50} {old|new}" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
num_workers="${NUM_WORKERS:-16}"
max_steps="${MAX_STEPS:-10000}"
val_interval="${VAL_INTERVAL:-1000}"
checkpoint_interval="${CHECKPOINT_INTERVAL:-5000}"
self_exclude="${SELF_EXCLUDE:-false}"
case "$self_exclude" in
  true) exclusion_name=selfexclude ;;
  false) exclusion_name=selfaccess ;;
  *)
    echo "SELF_EXCLUDE must be true or false" >&2
    exit 2
    ;;
esac
run_name="le_jepa_chemmask_${experiment_name}_${predictor_variant}_${exclusion_name}_${max_steps}"

cd "$repo_root"
exec "$python_bin" proteinfoundation/jepa/train.py \
  run_name_="$run_name" \
  +single=true \
  +masking_mode="$masking_mode" \
  +sidechain_mask_pct="$mask_pct" \
  +partial_coordinate_self_exclude="$self_exclude" \
  +partial_coordinate_diagnostics=true \
  baseline_run_predictor="$predictor_flag" \
  baseline_run_ema=false \
  baseline_run_layernorm=false \
  opt.lr=0.0001 \
  +opt.max_steps="$max_steps" \
  opt.val_check_interval="$val_interval" \
  log.checkpoint_every_n_steps="$checkpoint_interval" \
  dataset.datamodule.num_workers="$num_workers"
