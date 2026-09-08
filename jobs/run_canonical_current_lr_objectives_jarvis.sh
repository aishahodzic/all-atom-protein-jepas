#!/bin/bash
# Matched protein, residue, and protein+rotation canonical LeJEPA runs for two
# A30 GPUs each.

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "protein" && "$1" != "residue" && "$1" != "rotation" ) ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=0,1 $0 {protein|residue|rotation}" >&2
    exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
OBJECTIVE=$1

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
: "${DATA_PATH:?Set DATA_PATH to the directory containing pdb_train/}"
command -v "$PYTHON_BIN" >/dev/null

if [[ "$OBJECTIVE" == "protein" ]]; then
    RUN_NAME=canonical_lejepa_crop_fixed96_protein_perview_b96_indlocal_start1e4_peak4e4_end1e5_warmup1200_30k
    ALIGNMENT_LEVEL=protein
    SIGREG_INPUT=protein_mean
    RESIDUE_CHECKPOINTING=false
    COMPACT_VISIBLE_KV=false
    RESIDUE_SAMPLES=1
    INDEPENDENT_RESIDUE_SAMPLES=false
    INDEPENDENT_VIEW_ROTATIONS=false
elif [[ "$OBJECTIVE" == "residue" ]]; then
    RUN_NAME=canonical_lejepa_crop_fixed96_residue_perview_b96_indlocal_k4_start1e4_peak4e4_end1e5_warmup1200_30k
    ALIGNMENT_LEVEL=residue
    SIGREG_INPUT=protein_centered_residue
    RESIDUE_CHECKPOINTING=true
    COMPACT_VISIBLE_KV=true
    RESIDUE_SAMPLES=4
    INDEPENDENT_RESIDUE_SAMPLES=true
    INDEPENDENT_VIEW_ROTATIONS=false
else
    RUN_NAME=canonical_lejepa_crop_fixed96_protein_perview_b96_indlocal_rotations_start1e4_peak4e4_end1e5_warmup1200_30k
    ALIGNMENT_LEVEL=protein
    SIGREG_INPUT=protein_mean
    RESIDUE_CHECKPOINTING=false
    COMPACT_VISIBLE_KV=false
    RESIDUE_SAMPLES=1
    INDEPENDENT_RESIDUE_SAMPLES=false
    INDEPENDENT_VIEW_ROTATIONS=true
fi

CHECKPOINT_DIR="$PROJECT_ROOT/store/$RUN_NAME/checkpoints"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1
SCRATCH_ROOT=${TMPDIR:-"$HOME/tmp"}
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH_ROOT/${RUN_NAME}-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

RESUME_OVERRIDE=()
if [[ -f "$CHECKPOINT_DIR/last.ckpt" ]]; then
    RESUME_STEP=$(
        "$PYTHON_BIN" -c \
        'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)["global_step"]))' \
        "$CHECKPOINT_DIR/last.ckpt"
    )
    if (( RESUME_STEP >= 30000 )); then
        echo "$RUN_NAME is complete at step $RESUME_STEP"
        exit 0
    fi
    echo "Resuming $RUN_NAME from step $RESUME_STEP"
    RESUME_OVERRIDE=("resume.ckpt_path=$CHECKPOINT_DIR/last.ckpt")
fi

echo "Launching $OBJECTIVE objective on two GPUs; global batch 96"
exec "$PYTHON_BIN" -m proteinfoundation.jepa.train_canonical_lejepa \
    run_name_="$RUN_NAME" seed=42 alignment_level="$ALIGNMENT_LEVEL" \
    masked_kv_mode=visible_only \
    residue_view_checkpointing="$RESIDUE_CHECKPOINTING" \
    residue_compact_visible_kv="$COMPACT_VISIBLE_KV" \
    independent_view_rotations="$INDEPENDENT_VIEW_ROTATIONS" \
    views.cropping=true views.packed_views=true \
    views.global_crop_length=96 views.global_crop_fraction_min=null \
    views.global_crop_fraction_max=null views.global_crop_min_length=96 \
    views.global_crop_max_length=null views.local_crop_length=48 \
    views.locals_within_global=true views.shared_crop_anchor=false \
    views.global_mask_pct=0.20 views.local_mask_pct=0.50 \
    sigreg.input="$SIGREG_INPUT" sigreg.joint_views=false \
    sigreg.residue_samples_per_protein="$RESIDUE_SAMPLES" \
    sigreg.independent_residue_samples="$INDEPENDENT_RESIDUE_SAMPLES" \
    dataset.datamodule.dataselector.min_length=96 \
    dataset.datamodule.batch_size=48 \
    trainer.devices=2 trainer.num_nodes=1 trainer.strategy=ddp \
    trainer.use_distributed_sampler=false trainer.precision=bf16-mixed \
    trainer.accumulate_grad_batches=1 trainer.max_steps=30000 \
    optimizer.lr=4.0e-4 schedule.total_steps=30000 \
    schedule.peak_step=1200 schedule.start_lr=1.0e-4 \
    schedule.end_lr=1.0e-5 ema.enabled=true \
    checkpoint.every_n_train_steps=10000 \
    checkpoint.additional_every_n_train_steps=16667 \
    checkpoint.last_every_n_train_steps=1500 checkpoint.save_last=true \
    checkpoint.save_weights_only=false "${RESUME_OVERRIDE[@]}"
