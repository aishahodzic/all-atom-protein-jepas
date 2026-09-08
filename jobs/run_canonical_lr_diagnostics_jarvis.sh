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

ARTIFACT_ID="df_pdb_f1_minl50_maxl256_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse"
ARTIFACT_NAMES=(
    "${ARTIFACT_ID}.csv"
    "seq_${ARTIFACT_ID}.fasta"
    "cluster_seqid_0.5_${ARTIFACT_ID}_test.fasta"
    "cluster_seqid_0.5_${ARTIFACT_ID}_test.tsv"
)
ARTIFACT_HASHES=(
    c65ba9328adacc49a1c2488ee3908c224713c0d33a90f0bb55cdd94ad46e0c4b
    016dfe8e06287a3dae1286ceaca3964adb19d1bd49652d700da3fdcc1a9893cc
    9f41f4a621fda2769ec5fe09ef358c263ad30d182ae1188c2e642472c7bffee6
    06ae15f39f6f5e68c4bd53e0ac42380cb82e58f7a95474fbab5144ae0544ad31
)
for index in "${!ARTIFACT_NAMES[@]}"; do
    artifact="$DATA_PATH/pdb_train/${ARTIFACT_NAMES[$index]}"
    [[ -f "$artifact" ]] || { echo "missing frozen cohort artifact: $artifact" >&2; exit 1; }
    actual_hash=$(sha256sum "$artifact" | cut -d' ' -f1)
    if [[ "$actual_hash" != "${ARTIFACT_HASHES[$index]}" ]]; then
        echo "wrong frozen cohort artifact: $artifact" >&2
        echo "expected ${ARTIFACT_HASHES[$index]}, got $actual_hash" >&2
        exit 1
    fi
done

LABEL_CACHE="$PROJECT_ROOT/Evaluation/linear_probing/latents/labels.pt"
LABEL_CACHE_HASH=4cb784751329cf75e091a56328f21db43035525b555a4e4f149d74e4c430e6e5
[[ -f "$LABEL_CACHE" ]] || { echo "missing frozen label cache: $LABEL_CACHE" >&2; exit 1; }
actual_label_hash=$(sha256sum "$LABEL_CACHE" | cut -d' ' -f1)
if [[ "$actual_label_hash" != "$LABEL_CACHE_HASH" ]]; then
    echo "wrong frozen label cache: $LABEL_CACHE" >&2
    echo "expected $LABEL_CACHE_HASH, got $actual_label_hash" >&2
    exit 1
fi

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
