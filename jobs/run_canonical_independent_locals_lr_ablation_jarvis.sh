#!/bin/bash
# Run one fixed-96/48 independent-local protein LeJEPA LR ablation on Jarvis.
# Default: one A100 with local/global batch 96. Set TRAIN_GPUS=2 to use two
# visible A30s with local batch 48 and the same global batch 96.

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "high" && "$1" != "low" ) ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=0 TRAIN_GPUS=1 $0 {high|low}" >&2
    exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
MODE=$1
TRAIN_GPUS=${TRAIN_GPUS:-1}

if [[ "$TRAIN_GPUS" == "1" ]]; then
    LOCAL_BATCH=96
    TRAINER_STRATEGY=auto
elif [[ "$TRAIN_GPUS" == "2" ]]; then
    LOCAL_BATCH=48
    TRAINER_STRATEGY=ddp
else
    echo "TRAIN_GPUS must be 1 or 2" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
: "${DATA_PATH:?Set DATA_PATH to the directory containing pdb_train/}"
command -v "$PYTHON_BIN" >/dev/null

if [[ "$MODE" == "high" ]]; then
    RUN_NAME=canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e3_cosine_30k
    PEAK_LR=1.0e-3
    START_LR=2.5e-4
else
    RUN_NAME=canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e4_cosine_30k
    PEAK_LR=1.0e-4
    START_LR=2.5e-5
fi

CHECKPOINT_DIR="$PROJECT_ROOT/store/$RUN_NAME/checkpoints"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${RUN_NAME}-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RESUME_OVERRIDE=()
if [[ -f "$CHECKPOINT_DIR/last.ckpt" ]]; then
    RESUME_STEP=$(
        "$PYTHON_BIN" -c \
        'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)["global_step"]))' \
        "$CHECKPOINT_DIR/last.ckpt"
    )
    if (( RESUME_STEP >= 30000 )); then
        echo "$RUN_NAME is already complete at step $RESUME_STEP"
        exit 0
    fi
    echo "Resuming $RUN_NAME from step $RESUME_STEP"
    RESUME_OVERRIDE=("resume.ckpt_path=$CHECKPOINT_DIR/last.ckpt")
fi

echo "Launching $MODE-LR run with $TRAIN_GPUS GPU(s), global batch 96, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all-visible}"
exec "$PYTHON_BIN" -m proteinfoundation.jepa.train_canonical_lejepa \
    run_name_="$RUN_NAME" seed=42 alignment_level=protein \
    masked_kv_mode=visible_only residue_view_checkpointing=false \
    residue_compact_visible_kv=false independent_view_rotations=false \
    views.cropping=true views.packed_views=true \
    views.global_crop_length=96 views.global_crop_fraction_min=null \
    views.global_crop_fraction_max=null views.global_crop_min_length=96 \
    views.global_crop_max_length=null views.local_crop_length=48 \
    views.locals_within_global=true views.shared_crop_anchor=false \
    views.global_mask_pct=0.20 views.local_mask_pct=0.50 \
    sigreg.input=protein_mean sigreg.joint_views=false \
    dataset.datamodule.dataselector.min_length=96 \
    dataset.datamodule.batch_size="$LOCAL_BATCH" trainer.devices="$TRAIN_GPUS" \
    trainer.num_nodes=1 trainer.strategy="$TRAINER_STRATEGY" \
    trainer.use_distributed_sampler=false \
    trainer.precision=bf16-mixed trainer.accumulate_grad_batches=1 \
    trainer.max_steps=30000 optimizer.lr="$PEAK_LR" \
    schedule.total_steps=30000 schedule.peak_step=3600 \
    schedule.start_lr="$START_LR" schedule.end_lr=1.0e-5 \
    ema.enabled=true checkpoint.every_n_train_steps=10000 \
    checkpoint.additional_every_n_train_steps=16667 \
    checkpoint.last_every_n_train_steps=1500 checkpoint.save_last=true \
    checkpoint.save_weights_only=false "${RESUME_OVERRIDE[@]}"
