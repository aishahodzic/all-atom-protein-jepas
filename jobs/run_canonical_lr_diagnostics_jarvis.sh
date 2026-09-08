#!/bin/bash
# Frozen raw-encoder representation diagnostics for one Jarvis LR-sweep run.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 RUN_NAME [STEP]" >&2
    exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_NAME=$1
STEP=${2:-10000}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ROOT="$PROJECT_ROOT/store/$RUN_NAME"
CHECKPOINT="$RUN_ROOT/checkpoints/step-$(printf '%08d' "$STEP").ckpt"
CONFIG="$RUN_ROOT/checkpoints/exp_config_${RUN_NAME}.json"

[[ -f "$CHECKPOINT" ]] || { echo "missing $CHECKPOINT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
: "${DATA_PATH:?DATA_PATH must point to the directory containing pdb_train/}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CANONICAL_RESIDUE_ANALYSIS_PROFILE=external-protein-perview-independent-locals
export CANONICAL_ENCODER_SOURCE=raw
export CANONICAL_EXPECTED_STEP="$STEP"
export CANONICAL_EXTERNAL_CHECKPOINT="$CHECKPOINT"
export CANONICAL_EXTERNAL_CONFIG="$CONFIG"
export CANONICAL_EXTERNAL_NAME="$RUN_NAME"
export CANONICAL_EXTERNAL_RESULT_TAG="${RUN_NAME}_raw_${STEP}"

SCRATCH_ROOT=${TMPDIR:-"$HOME/tmp"}
export MPLCONFIGDIR="$SCRATCH_ROOT/${RUN_NAME}-raw-${STEP}-matplotlib"
mkdir -p "$MPLCONFIGDIR"

exec "$PYTHON_BIN" Evaluation/diagnostics/analyze_canonical_residue_ab_10k.py
