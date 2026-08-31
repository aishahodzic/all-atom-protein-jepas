#!/bin/bash
# Single entrypoint for LaProteina tokenizer evaluation on StructTokenBench.
#
# Usage:
#   bash src/script/run_eval.sh
#   DRY_RUN=1 D_MODEL_OVERRIDE=8 bash src/script/run_eval.sh  # print commands without loading checkpoints
#   NUM_WORKERS=4 bash src/script/run_eval.sh       # override DataLoader workers (default 8)
#   PRECISION=bf16-mixed bash src/script/run_eval.sh  # opt into mixed precision (default: 32-true)
#   ENCODER_SOURCE=target bash src/script/run_eval.sh  # JEPA EMA/target encoder (default: online)
#   TOKENIZER_EVAL_MAX_LENGTH=600 bash src/script/run_eval.sh  # evaluation-only length override
#   TOKENIZER_CACHE_TAG=featurefix-v1 bash src/script/run_eval.sh  # force a fresh preprocessing cache
#   bash src/script/run_eval.sh --protocol=upstream-l600-v1 ...  # locked publication protocol
#   STB_SKIP_BOOTSTRAP=1 STB_SKIP_DATA_SETUP=1 PYTHON_BIN=python bash ...  # portable/Jarvis mode
#   STB_SMOKE=1 bash ... --protocol=upstream-l600-v1  # full cohort cache, tiny probe
#   STB_SHARED_EMBEDDING_CACHE=0 bash ...  # disable cross-task frozen-embedding reuse
#
#   Skip the interactive prompts by passing checkpoint/experiment/tasks up front,
#   as flags or equivalently as env vars (CHECKPOINT_LABEL / EXPERIMENT / TASKS):
#   bash src/script/run_eval.sh --checkpoint=100k --experiment=ca-cond-baseline --tasks=BindInt,BindBio
#   bash src/script/run_eval.sh --checkpoint=100k --experiment=all --tasks=all
#   CHECKPOINT_LABEL=100k EXPERIMENT=ca-cond-baseline TASKS=1,2 bash src/script/run_eval.sh
#   (--tasks accepts task names or their menu numbers, comma or space separated, or "all")
#
# Creates the pstbench conda env if it doesn't exist yet, and installs all
# missing dependencies into it. Assumes: checkpoints are already staged under
# 100k-checkpoints/{ae,jepa,le-jepa}/... (checkpoint download/verification is
# intentionally NOT handled here) -- see 100k-checkpoints/shared/ for the encoder
# config shared by all 6 jepa/le-jepa experiments.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Multiprocessing-spawned workers need both StructTokenBench's src directory and
# the la-proteina root (for the local openfold/proteinfoundation packages).
export PYTHONPATH="$REPO_DIR/src:$(dirname "$REPO_DIR")${PYTHONPATH:+:$PYTHONPATH}"

export DIR="$REPO_DIR"
DRY_RUN="${DRY_RUN:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"          # DataLoader workers per task run (nproc-limited; see README perf notes)
PRECISION="${PRECISION:-}"               # opt-in, e.g. PRECISION=16-mixed or bf16-mixed; default keeps trainer.yaml's 32-true
ENCODER_SOURCE="${ENCODER_SOURCE:-online}" # JEPA branch: online or target (target_encoder/ema_encoder)
TOKENIZER_EVAL_MAX_LENGTH="${TOKENIZER_EVAL_MAX_LENGTH:-}" # blank: checkpoint policy; e.g. 600 for full STB cohort
TOKENIZER_CACHE_TAG="${TOKENIZER_CACHE_TAG:-}" # blank: reuse compatible cache; set after preprocessing revisions
TOKENIZER_EVAL_PROTOCOL="${TOKENIZER_EVAL_PROTOCOL:-}" # blank: historical integration; upstream-l600-v1: locked publication path
STB_SKIP_BOOTSTRAP="${STB_SKIP_BOOTSTRAP:-0}"
STB_SKIP_DATA_SETUP="${STB_SKIP_DATA_SETUP:-0}"
STB_SMOKE="${STB_SMOKE:-${SMOKE:-0}}"
STB_SHARED_EMBEDDING_CACHE="${STB_SHARED_EMBEDDING_CACHE:-1}"

# ---------------------------------------------------------------------------
# CLI args (optional): --checkpoint=, --experiment=, --tasks=,
# --encoder-source=, --eval-max-length=, --cache-tag=, --protocol=. Equivalent to
# setting CHECKPOINT_LABEL / EXPERIMENT / TASKS env vars -- either skips the
# matching interactive prompt below. Flags take precedence over env vars.
# ---------------------------------------------------------------------------
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-}"
EXPERIMENT="${EXPERIMENT:-}"
TASKS="${TASKS:-}"
for arg in "$@"; do
    case "$arg" in
        --checkpoint=*) CHECKPOINT_LABEL="${arg#*=}" ;;
        --experiment=*) EXPERIMENT="${arg#*=}" ;;
        --tasks=*) TASKS="${arg#*=}" ;;
        --encoder-source=*) ENCODER_SOURCE="${arg#*=}" ;;
        --eval-max-length=*) TOKENIZER_EVAL_MAX_LENGTH="${arg#*=}" ;;
        --cache-tag=*) TOKENIZER_CACHE_TAG="${arg#*=}" ;;
        --protocol=*) TOKENIZER_EVAL_PROTOCOL="${arg#*=}" ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "[Error] unknown argument: $arg (expected --checkpoint=, --experiment=, --tasks=, --encoder-source=, --eval-max-length=, --cache-tag=, --protocol=)" >&2
            exit 1
            ;;
    esac
done
if [ "$ENCODER_SOURCE" != "online" ] && [ "$ENCODER_SOURCE" != "target" ]; then
    echo "[Error] ENCODER_SOURCE must be 'online' or 'target', got '$ENCODER_SOURCE'" >&2
    exit 1
fi
if [ -n "$TOKENIZER_EVAL_MAX_LENGTH" ] && ! [[ "$TOKENIZER_EVAL_MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Error] TOKENIZER_EVAL_MAX_LENGTH must be a positive integer, got '$TOKENIZER_EVAL_MAX_LENGTH'" >&2
    exit 1
fi
if [ -n "$TOKENIZER_CACHE_TAG" ] && ! [[ "$TOKENIZER_CACHE_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[Error] TOKENIZER_CACHE_TAG may contain only letters, numbers, '.', '_' and '-'" >&2
    exit 1
fi
if [ -n "$TOKENIZER_EVAL_PROTOCOL" ] && [ "$TOKENIZER_EVAL_PROTOCOL" != "upstream-l600-v1" ]; then
    echo "[Error] unsupported protocol '$TOKENIZER_EVAL_PROTOCOL' (expected upstream-l600-v1)" >&2
    exit 1
fi
if [ "$STB_SMOKE" != "0" ] && [ "$STB_SMOKE" != "1" ]; then
    echo "[Error] STB_SMOKE must be 0 or 1" >&2
    exit 1
fi
if [ "$STB_SHARED_EMBEDDING_CACHE" != "0" ] && [ "$STB_SHARED_EMBEDDING_CACHE" != "1" ]; then
    echo "[Error] STB_SHARED_EMBEDDING_CACHE must be 0 or 1" >&2
    exit 1
fi

if [ "$TOKENIZER_EVAL_PROTOCOL" = "upstream-l600-v1" ]; then
    if [ "$ENCODER_SOURCE" != "online" ]; then
        echo "[Error] upstream-l600-v1 locks the online/student encoder" >&2
        exit 1
    fi
    if [ -n "$TOKENIZER_EVAL_MAX_LENGTH" ] && [ "$TOKENIZER_EVAL_MAX_LENGTH" != "600" ]; then
        echo "[Error] upstream-l600-v1 uses STB selected-residue L600; conflicting whole-chain max '$TOKENIZER_EVAL_MAX_LENGTH'" >&2
        exit 1
    fi
    if [ -n "$TOKENIZER_CACHE_TAG" ] && [ "$TOKENIZER_CACHE_TAG" != "upstream-l600-v1" ]; then
        echo "[Error] upstream-l600-v1 locks TOKENIZER_CACHE_TAG=upstream-l600-v1" >&2
        exit 1
    fi
    # An old explicit 600 is accepted as migration input but deliberately not
    # forwarded: STB applies 600 after domain/residue selection, whereas this
    # tokenizer option filters whole chains and would wrongly remove Homo rows.
    TOKENIZER_EVAL_MAX_LENGTH=""
    TOKENIZER_CACHE_TAG="upstream-l600-v1"
fi

if [ "$DRY_RUN" = "1" ]; then
    # A dry run must be side-effect free: never create environments, install
    # packages, or download the PDB corpus just to print commands.
    STB_SKIP_BOOTSTRAP=1
    STB_SKIP_DATA_SETUP=1
fi

export TOKENIZER_EVAL_PROTOCOL STB_SMOKE
export EVALUATION_RUN_ID="${EVALUATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-p$$}"
export EVALUATION_STARTED_AT_UTC="${EVALUATION_STARTED_AT_UTC:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# ---------------------------------------------------------------------------
# 1. Env activation
# ---------------------------------------------------------------------------
if [ "$STB_SKIP_BOOTSTRAP" = "1" ]; then
    PY="${PYTHON_BIN:-python}"
    PIP="${PIP_BIN:-pip}"
    if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
        echo "[Error] PYTHON_BIN '$PY' is not executable" >&2
        exit 1
    fi
    echo "== Using caller-provided Python environment: $PY =="
else
    CONDA_SH="${CONDA_SH:-/slurm-storage/miniconda3/etc/profile.d/conda.sh}"
    if [ ! -f "$CONDA_SH" ]; then
        echo "[Error] conda activation script not found at $CONDA_SH; set STB_SKIP_BOOTSTRAP=1 and PYTHON_BIN on Jarvis" >&2
        exit 1
    fi
    source "$CONDA_SH"
    if ! conda env list | grep -q '^pstbench '; then
        echo "  conda env 'pstbench' not found; creating it (python=3.11) -- one-time, ~1min"
        conda create -y -n pstbench python=3.11
    fi
    conda activate pstbench
    hash -r
    PY=python
    PIP=pip
fi

pkg_present () {
    # $1 = python import name to probe
    "$PY" -c "import $1" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# 2. Experiment / task menus, and prompts -- asked immediately, before any
#    dependency-install/data-download work, so you're not left waiting to be
#    asked what to actually run.
# ---------------------------------------------------------------------------
# All checkpoints staged under 100k-checkpoints/ (see 100k-checkpoints/shared/ for the
# jepa/le-jepa encoder config, reused across all 6 jepa/le-jepa rows since they share the
# same nn_ae:nn_130m architecture and dataselector -- only the online_encoder weights and
# the masking/action-conditioning behavior differ between them, and neither is read from
# this config; see tokenizer.py's WrappedLaProteinaTokenizer for what's actually consumed).
CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/100k-checkpoints}"
JEPA_ENCODER_CFG="${JEPA_ENCODER_CFG:-$CKPT_ROOT/shared/exp_config_jepa_nn130m.json}"
# External reference checkpoint (NVIDIA's published La-Proteina AE, not one of "my 8") --
# lives at the la-proteina repo root, not under 100k-checkpoints/. Same latent_z_dim=8,
# loaded via the plain "ae" path (no encoder-cfg-path), verified via a one-protein load
# test (encode_structure OK, feats_seq auto-includes x1_bb_angles/x1_sidechain_angles
# since WrappedLaProteinaTokenizer reads each checkpoint's own saved config).
AE1_CKPT="${AE1_CKPT:-$(dirname "$REPO_DIR")/checkpoints_laproteina/AE1_ucond_512.ckpt}"

EXPERIMENT_NAMES=(ca-cond-baseline fully-latent jepa-rand-mask-50 jepa-new-predictor le-jepa-new-predictor le-jepa-new-predictor-30k le-jepa-new-predictor-40k le-jepa-new-predictor-60k le-jepa-new-predictor-80k le-jepa-spatial3d-block-30k le-jepa-spatial3d-block-60k le-jepa-spatial3d-block-100k le-jepa-chemmask-exp1-30k le-jepa-chemmask-exp1-100k le-jepa-chemmask-exp3-30k le-jepa-chemmask-exp3-100k jepa-rand-mask-70 jepa-mutation le-jepa-rand-mask-50 le-jepa-rand-mask-70 le-jepa-mutation le-jepa-rand50-relative-pos-100k le-jepa-rand70-relative-pos-100k le-jepa-rand70-relative-pos-rotation-100k le-jepa-rand70-relative-pos-rotation-sigreg-0p01-100k le-jepa-spatial3d-relative-pos-d8-100k jepa-rand70-d64-independent-rotations-100k jepa-rand-mask-70-dim64 le-jepa-rand-mask-70-dim64 jepa-rand-mask-50-selfattn jepa-rand-mask-70-selfattn le-jepa-rand-mask-50-200k ae1-external random-init random-init-64 jepa-rand50-control-30k jepa-rand50-ema-30k jepa-rand50-layernorm-30k jepa-rand50-predictor-30k)

experiment_model_args () {
    # sets TOKENIZERNAME and EXTRA_MODEL_ARGS for experiment name $1
    case "$1" in
        ca-cond-baseline)
            TOKENIZERNAME="laproteina_ca_cond"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/ae/ca-cond/100k-EMA.ckpt tokenizer_checkpoint_type=ae"
            ;;
        fully-latent)
            TOKENIZERNAME="laproteina_fully_latent"
            # Exact step-100k EMA snapshot from the resumed fully-latent training run.
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/ae/fully_latent/100k-EMA.ckpt tokenizer_checkpoint_type=ae"
            ;;
        jepa-rand-mask-50)
            TOKENIZERNAME="laproteina_jepa_rand_mask_50"
            # placeholder: jepa_random_masking_50_v1 hasn't reached step 100000 yet
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/jepa/random_masking_50pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-new-predictor)
            TOKENIZERNAME="laproteina_jepa_new_predictor"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/jepa/new_predictor/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-new-predictor)
            TOKENIZERNAME="laproteina_le_jepa_new_predictor"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/new_predictor/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-new-predictor-30k)
            TOKENIZERNAME="laproteina_le_jepa_new_predictor_30k"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/new_predictor/30k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-new-predictor-40k|le-jepa-new-predictor-60k|le-jepa-new-predictor-80k)
            checkpoint_step="${1##*-}"
            TOKENIZERNAME="laproteina_le_jepa_new_predictor_${checkpoint_step}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/new_predictor/${checkpoint_step}.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-spatial3d-block-30k|le-jepa-spatial3d-block-60k|le-jepa-spatial3d-block-100k)
            checkpoint_step="${1##*-}"
            TOKENIZERNAME="laproteina_le_jepa_spatial3d_block_${checkpoint_step}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/spatial3d_block_dim64_new_predictor/${checkpoint_step}.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-chemmask-exp1-30k|le-jepa-chemmask-exp1-100k)
            checkpoint_step="${1##*-}"
            TOKENIZERNAME="laproteina_le_jepa_chemmask_exp1_${checkpoint_step}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/chem_masking/exp1_aa_partial50/${checkpoint_step}.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-chemmask-exp3-30k|le-jepa-chemmask-exp3-100k)
            checkpoint_step="${1##*-}"
            TOKENIZERNAME="laproteina_le_jepa_chemmask_exp3_${checkpoint_step}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/chem_masking/exp3_noaa_sc50/${checkpoint_step}.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand-mask-70)
            TOKENIZERNAME="laproteina_jepa_rand_mask_70"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/jepa/random_masking_70pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-mutation)
            TOKENIZERNAME="laproteina_jepa_mutation"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/jepa/mutation/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand-mask-50)
            TOKENIZERNAME="laproteina_le_jepa_rand_mask_50"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/random_masking_50pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand-mask-70)
            TOKENIZERNAME="laproteina_le_jepa_rand_mask_70"
            # placeholder: le_jepa_rand50_70_v1 hasn't reached step 100000 yet
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/random_masking_70pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-mutation)
            TOKENIZERNAME="laproteina_le_jepa_mutation"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/mutation/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand50-relative-pos-100k)
            TOKENIZERNAME="laproteina_le_jepa_rand50_relative_pos_100k"
            : "${LEJEPA_RELATIVE_POS_R50_CKPT:?Set LEJEPA_RELATIVE_POS_R50_CKPT to the final 100k checkpoint}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$LEJEPA_RELATIVE_POS_R50_CKPT tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand70-relative-pos-100k)
            TOKENIZERNAME="laproteina_le_jepa_rand70_relative_pos_100k"
            : "${LEJEPA_RELATIVE_POS_R70_CKPT:?Set LEJEPA_RELATIVE_POS_R70_CKPT to the final 100k checkpoint}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$LEJEPA_RELATIVE_POS_R70_CKPT tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand70-relative-pos-rotation-100k)
            TOKENIZERNAME="laproteina_le_jepa_rand70_relative_pos_rotation_100k"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/le-jepa/relative-position-and-rotation/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand70-relative-pos-rotation-sigreg-0p01-100k)
            TOKENIZERNAME="laproteina_le_jepa_rand70_relative_pos_rotation_sigreg_0p01_100k"
            : "${LEJEPA_RELROT_SIGREG_0P01_CKPT:?Set LEJEPA_RELROT_SIGREG_0P01_CKPT to the final 100k checkpoint}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$LEJEPA_RELROT_SIGREG_0P01_CKPT tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-spatial3d-relative-pos-d8-100k)
            TOKENIZERNAME="laproteina_le_jepa_spatial3d_relative_pos_d8_100k"
            : "${LEJEPA_SPATIAL3D_RELATIVE_POS_D8_CKPT:?Set LEJEPA_SPATIAL3D_RELATIVE_POS_D8_CKPT to the final 100k checkpoint}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$LEJEPA_SPATIAL3D_RELATIVE_POS_D8_CKPT tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand70-d64-independent-rotations-100k)
            TOKENIZERNAME="laproteina_jepa_rand70_d64_independent_rotations_100k"
            : "${JEPA_RAND70_D64_INDEPENDENT_ROTATIONS_CKPT:?Set JEPA_RAND70_D64_INDEPENDENT_ROTATIONS_CKPT to the final 100k checkpoint}"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$JEPA_RAND70_D64_INDEPENDENT_ROTATIONS_CKPT tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand-mask-70-dim64)
            TOKENIZERNAME="laproteina_jepa_rand_mask_70_dim64"
            # 64-dim latent (vs. the other jepa/le-jepa entries' 8-dim); both
            # WrappedLaProteinaTokenizer's encoder and d_model (below, in the main
            # loop) infer latent_z_dim from the checkpoint's own weights, so the
            # same shared JEPA_ENCODER_CFG works here unchanged.
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/bottle-inspect/64dim/jepa/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand-mask-70-dim64)
            TOKENIZERNAME="laproteina_le_jepa_rand_mask_70_dim64"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/bottle-inspect/64dim/le-jepa/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand-mask-50-selfattn)
            TOKENIZERNAME="laproteina_jepa_rand_mask_50_selfattn"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/bottle-inspect/self-attn/jepa/random_masking_50pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand-mask-70-selfattn)
            TOKENIZERNAME="laproteina_jepa_rand_mask_70_selfattn"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/bottle-inspect/self-attn/jepa/random_masking_70pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        le-jepa-rand-mask-50-200k)
            # 200k steps, not 100k -- lives in its own top-level 200k-checkpoint/ dir, not
            # under CKPT_ROOT. --checkpoint= is just this run's output/summary-row label,
            # not a path component here, so running this under --checkpoint=100k is fine.
            TOKENIZERNAME="laproteina_le_jepa_rand_mask_50_200k"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$REPO_DIR/200k-checkpoint/le-jepa/random_masking_50pct/200k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        ae1-external)
            TOKENIZERNAME="laproteina_ae1_external"
            # +force_remove_non_standard_residues=true: see tokenizer.py for why. "+" is required --
            # none of the task configs (biolip2.yaml, interpro.yaml, etc.) declare this key, so a
            # plain override is rejected by Hydra ("Key ... is not in struct").
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$AE1_CKPT tokenizer_checkpoint_type=ae"
            if [ -z "$TOKENIZER_EVAL_PROTOCOL" ]; then
                EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS +tokenizer_force_remove_non_standard_residues=true"
            fi
            ;;
        jepa-rand50-control-30k)
            TOKENIZERNAME="laproteina_jepa_rand50_control_30k"
            # 30k steps, matched-step baseline for the 3 rand-50 ablations below --
            # not a 100k checkpoint, --checkpoint= is just this run's output label.
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/baseline_ablations/control/last.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand50-ema-30k)
            TOKENIZERNAME="laproteina_jepa_rand50_ema_30k"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/baseline_ablations/ema/last.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand50-layernorm-30k)
            TOKENIZERNAME="laproteina_jepa_rand50_layernorm_30k"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/baseline_ablations/layernorm/last.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-rand50-predictor-30k)
            TOKENIZERNAME="laproteina_jepa_rand50_predictor_30k"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/baseline_ablations/predictor/last.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        random-init)
            TOKENIZERNAME="laproteina_random_init"
            # Untrained floor -- same EncoderTransformer config as all 8 in-house checkpoints
            # (verified byte-identical in Evaluation/linear_probing/checkpoints.py), same fixed
            # weights (seed=42) as the linear-probing/kNN-purity random-init floor, just repacked
            # into the "jepa" checkpoint_type's expected {"state_dict": {"online_encoder.*": ...}}
            # format (see 100k-checkpoints/random-init/100k.ckpt) so it loads through the exact
            # same WrappedLaProteinaTokenizer path as every other checkpoint here -- not a special
            # case downstream.
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/random-init/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        random-init-64)
            TOKENIZERNAME="laproteina_random_init_64"
            # Untrained seed-42 floor matched to the 64-dim spatial-block run.
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/random-init-64/step0.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        *)
            echo "[Error] unknown experiment: $1" >&2
            exit 1
            ;;
    esac
}

TASK_NAMES=(BindInt BindBio BindShake CatInt CatBio Con Rep Ept FlexRMSF FlexBFactor FlexNEQ Homo)
TASK_CONFIGS=(interpro.yaml biolip2.yaml proteinshake_binding_site.yaml interpro.yaml biolip2.yaml interpro.yaml interpro.yaml proteinglue_epitope_region.yaml atlas.yaml atlas.yaml atlas.yaml remote_homology.yaml)
TASK_TARGET_FIELDS=(binding_label binding_label binding_site activesite_label catalytic_label conservedsite_label repeat_label epitope_label rmsf_score bfactor_score neq_score fold_label)
TASK_PREFIXES=(bindint bindbio bindshake catint catbio con rep ept flexrmsf flexbfactor flexneq homo)
# Matches upstream README's EXTRA_TASK_ARGS for FlexRMSF/FlexBFactor/FlexNEQ.
# No-op today (AtlasDataset.retrieve_pdb_path uses data_path, not
# pdb_data_dir) -- kept for README fidelity. Real fix for the flexrmsf/
# flexbfactor/flexneq failures is in tokenizer.py's _get_native_datamodule.
PHYSCHEM_EXTRA="data.pdb_data_dir=${STB_RELEASE_DATA_ROOT:-$DIR/struct_token_bench_release_data}/data/physicochemical/ lightning.callbacks.checkpoint.monitor='validation_spearmanr' lightning.callbacks.early_stopping.monitor='validation_spearmanr' lightning.callbacks.checkpoint.mode='max' lightning.callbacks.early_stopping.mode='max'"
TASK_EXTRA_ARGS=("" "" "" "" "" "" "" "" "$PHYSCHEM_EXTRA" "$PHYSCHEM_EXTRA" "$PHYSCHEM_EXTRA" "optimization.micro_batch_size=64")

if [ -z "$CHECKPOINT_LABEL" ]; then
    echo
    echo "== What checkpoint step should the results table/summary be labeled with? (e.g. 100k, 200k) =="
    read -r CHECKPOINT_LABEL
fi
export CHECKPOINT_LABEL

STB_LOG_ROOT="${STB_LOG_ROOT:-$DIR/struct_token_bench_logs}"
SUMMARY_FILE="${SUMMARY_FILE:-$STB_LOG_ROOT/summary_results.csv}"
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$(dirname "$SUMMARY_FILE")"
    mkdir -p "$STB_LOG_ROOT"
fi
export SUMMARY_FILE STB_LOG_ROOT

if [ -n "$EXPERIMENT" ]; then
    exp_choice="$EXPERIMENT"
    valid=0
    for e in "${EXPERIMENT_NAMES[@]}" "all"; do
        [ "$exp_choice" = "$e" ] && valid=1
    done
    if [ "$valid" -ne 1 ]; then
        echo "[Error] --experiment/EXPERIMENT='$exp_choice' is not one of: ${EXPERIMENT_NAMES[*]} all" >&2
        exit 1
    fi
else
    echo
    echo "== Which experiment do you want to evaluate? =="
    select exp_choice in "${EXPERIMENT_NAMES[@]}" "all"; do
        [ -n "${exp_choice:-}" ] && break
        echo "Invalid choice, try again."
    done
fi
if [ "$exp_choice" = "all" ]; then
    SELECTED_EXPERIMENTS=("${EXPERIMENT_NAMES[@]}")
else
    SELECTED_EXPERIMENTS=("$exp_choice")
fi

if [ -n "$TASKS" ]; then
    TASK_INPUT="$TASKS"
else
    echo
    echo "== Which task(s) do you want to evaluate? =="
    for i in "${!TASK_NAMES[@]}"; do
        printf "  %2d) %s\n" "$((i+1))" "${TASK_NAMES[$i]}"
    done
    echo "Enter numbers separated by spaces (e.g. '1 3 9'), or 'all':"
    read -r TASK_INPUT
fi

if [ "$TASK_INPUT" = "all" ]; then
    SELECTED_TASK_IDX=($(seq 0 $((${#TASK_NAMES[@]} - 1))))
else
    SELECTED_TASK_IDX=()
    # accept task names and/or 1-based menu numbers, comma or space separated
    for n in $(echo "$TASK_INPUT" | tr ',' ' '); do
        if [[ "$n" =~ ^[0-9]+$ ]]; then
            if [ "$n" -lt 1 ] || [ "$n" -gt "${#TASK_NAMES[@]}" ]; then
                echo "[Error] task number '$n' is outside 1-${#TASK_NAMES[@]}" >&2
                exit 1
            fi
            SELECTED_TASK_IDX+=($((n - 1)))
        else
            found=0
            for i in "${!TASK_NAMES[@]}"; do
                if [ "${TASK_NAMES[$i]}" = "$n" ]; then
                    SELECTED_TASK_IDX+=("$i")
                    found=1
                    break
                fi
            done
            if [ "$found" -ne 1 ]; then
                echo "[Error] unknown task '$n' (expected one of: ${TASK_NAMES[*]}, or its menu number)" >&2
                exit 1
            fi
        fi
    done
fi

if [ "${#SELECTED_TASK_IDX[@]}" -eq 0 ]; then
    echo "[Error] no evaluation tasks selected" >&2
    exit 1
fi

# Fail before package installs or data downloads if a selected checkpoint was
# not staged on this machine. This catches registry/path mistakes immediately.
for preflight_experiment in "${SELECTED_EXPERIMENTS[@]}"; do
    experiment_model_args "$preflight_experiment"
    preflight_ckpt="${EXTRA_MODEL_ARGS#*tokenizer_pretrained_ckpt_path=}"
    preflight_ckpt="${preflight_ckpt%% *}"
    if [ "$DRY_RUN" != "1" ] && [ ! -f "$preflight_ckpt" ]; then
        echo "[Error] checkpoint for '$preflight_experiment' not found: $preflight_ckpt" >&2
        exit 1
    fi
    if [ "$DRY_RUN" != "1" ] && [[ "$EXTRA_MODEL_ARGS" == *"tokenizer_checkpoint_type=jepa"* ]] && [ ! -f "$JEPA_ENCODER_CFG" ]; then
        echo "[Error] JEPA encoder config not found: $JEPA_ENCODER_CFG" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 3. Base StructTokenBench deps (README), skipping packages that get a
#    version-pinned reinstall in step 3 anyway (torch itself, torch-scatter,
#    torch-cluster) — installing them here first would either be wasted work
#    or risk the same CUDA/torch ABI-mismatch class of failure step 3 exists
#    to avoid.
# ---------------------------------------------------------------------------
if [ "$STB_SKIP_BOOTSTRAP" != "1" ]; then
# Pin the CUDA/PyTorch ABI before installing Lightning, DeepSpeed, or any PyG
# consumer. On a fresh environment those packages would otherwise pull a
# different default Torch first and force a multi-GB reinstall.
echo "== Checking pinned Torch/PyG runtime =="
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ "$TORCH_VER" != "2.7.0+cu118" ]; then
    echo "  torch is '$TORCH_VER', pinning to 2.7.0+cu118 (+matching torchvision)"
    $PIP install "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu118
elif ! pkg_present torchvision; then
    $PIP install "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu118
fi
PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-2.7.0+cu118.html"
for pyg_import in torch_scatter torch_sparse torch_cluster; do
    if ! pkg_present "$pyg_import"; then
        echo "  installing $pyg_import against torch 2.7.0+cu118"
        $PIP install "$pyg_import" -f "$PYG_WHEEL_INDEX"
    fi
done
if ! pkg_present torch_geometric; then
    $PIP install torch_geometric
fi

echo "== Checking base StructTokenBench dependencies =="
declare -A BASE_DEPS=(
    [lmdb]="lmdb"
    [hydra]="hydra-core"
    [lightning]="lightning"
    [transformers]="transformers"
    [deepspeed]="deepspeed"
    [tensorboard]="tensorboard"
    [ipdb]="ipdb"
    [esm]="esm"
    [httpx]="httpx"  
    [cloudpathlib]="cloudpathlib"
    [pipreqs]="pipreqs"
    [lxml]="lxml"
    [proteinshake]="proteinshake"
    [tmtools]="tmtools"
    [tape]="tape_proteins"
    [accelerate]="accelerate"
    [line_profiler]="line_profiler"
    [mini3di]="mini3di"
    [tree]="dm-tree"
    [colorcet]="colorcet"
    [ogb]="ogb==1.2.1"
    [sympy]="sympy"
    [ase]="ase"
    [jax]="jax==0.4.25"
    [tensorflow]="tensorflow"
    [Bio]="biopython"
    [seaborn]="seaborn"
    [gdown]="gdown"
)
for import_name in "${!BASE_DEPS[@]}"; do
    if ! pkg_present "$import_name"; then
        echo "  installing ${BASE_DEPS[$import_name]} (missing: $import_name)"
        $PIP install "${BASE_DEPS[$import_name]}"
    fi
done
$PIP install --upgrade packaging >/dev/null

# ---------------------------------------------------------------------------
# 4. LaProteina-specific runtime dependencies.
# ---------------------------------------------------------------------------
echo "== Checking LaProteina-specific dependencies =="
if ! pkg_present graphein; then
    $PIP install graphein==1.7.7 --no-deps
fi
declare -A LAPROTEINA_DEPS=(
    [loguru]="loguru"
    [wget]="wget==3.2"
    [jaxtyping]="jaxtyping"
    [cpdb_protein]="cpdb-protein==0.2.0"
    [deepdiff]="deepdiff>=8.6.1"
    [multipledispatch]="multipledispatch"
    [plotly]="plotly"
    [rich_click]="rich-click"
    [xarray]="xarray"
    [ml_collections]="ml-collections==0.1.1"
    [sequence_models]="sequence_models"
)
for import_name in "${!LAPROTEINA_DEPS[@]}"; do
    if ! pkg_present "$import_name"; then
        echo "  installing ${LAPROTEINA_DEPS[$import_name]} (missing: $import_name)"
        $PIP install "${LAPROTEINA_DEPS[$import_name]}"
    fi
done
else
    echo "== Skipping dependency installation (STB_SKIP_BOOTSTRAP=1) =="
fi

# ---------------------------------------------------------------------------
# 5. Data: PDB mmCIF files + downstream labels (skip if already present)
# ---------------------------------------------------------------------------
# $DIR/pdb_data is a symlink to /mnt/hopper-raid/aishod/data/struct_token_bench_pdb_data --
# the extracted mmCIF corpus is ~150-200GB and didn't fit on /mnt/data (repo disk).
# /mnt/hopper-raid is NOT mounted on hopper itself (confirmed: `mount` has no entry for
# it there) -- it's hopper's storage exported *for other nodes*, not something hopper
# loops back and mounts on itself. On hopper the same data lives locally at
# /mnt/raid/aishod/data/struct_token_bench_pdb_data (confirmed present). This is the
# same distinction the JEPA training job scripts already make for DATA_PATH.
if [ -z "${PDB_DATA_ROOT:-}" ]; then
    if [ "$(hostname -s)" = "hopper" ]; then
        PDB_DATA_ROOT="/mnt/raid/aishod/data/struct_token_bench_pdb_data"
    else
        PDB_DATA_ROOT="$DIR/pdb_data"
    fi
fi
STB_RELEASE_DATA_ROOT="${STB_RELEASE_DATA_ROOT:-$DIR/struct_token_bench_release_data}"
STB_SHARED_EMBEDDING_CACHE_ROOT="${STB_SHARED_EMBEDDING_CACHE_ROOT:-$STB_RELEASE_DATA_ROOT/shared_embedding_cache}"
export PDB_DATA_ROOT STB_RELEASE_DATA_ROOT
export STB_SHARED_EMBEDDING_CACHE STB_SHARED_EMBEDDING_CACHE_ROOT
if [ "$STB_SHARED_EMBEDDING_CACHE" = "1" ]; then
    echo "== Shared frozen-embedding cache: $STB_SHARED_EMBEDDING_CACHE_ROOT =="
fi

# Guarded by a flock so concurrent eval jobs sharing $DIR over NFS don't all
# probe/touch this section at once.
if [ "$STB_SKIP_DATA_SETUP" != "1" ]; then
(
    flock -x 200

    MMCIF_DIR="$PDB_DATA_ROOT/mmcif_files"
    if [ -d "$MMCIF_DIR" ] && [ -n "$(ls -A "$MMCIF_DIR" 2>/dev/null)" ]; then
        echo "== PDB mmCIF data already present, skipping download =="
    else
        echo "== Downloading PDB mmCIF data =="
        mkdir -p "$MMCIF_DIR"
        (
            cd "$MMCIF_DIR"
            aws s3 cp s3://openfold/pdb_mmcif.zip "$MMCIF_DIR" --no-sign-request
            unzip -q pdb_mmcif.zip
            [ -f obsolete.dat ] || wget https://files.pdbj.org/pub/pdb/data/status/obsolete.dat
        )
    fi

    LABELS_DIR="$STB_RELEASE_DATA_ROOT/data"
    if [ -d "$LABELS_DIR" ] && [ -n "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
        echo "== Downstream label data already present, skipping download =="
    else
        echo "== Downloading downstream label data =="
        (
            cd "$(dirname "$STB_RELEASE_DATA_ROOT")"
            gdown https://drive.google.com/uc?id=1wJ4dSNdMyuF0985ET4UuwViHgV-clF4K
            unzip -q struct_token_bench_release_data_download.zip
            mv struct_token_bench_release_data_download "$STB_RELEASE_DATA_ROOT"
            rm -f struct_token_bench_release_data_download.zip
        )
    fi
) 200>"$DIR/.data_setup.lock"
else
    echo "== Skipping automatic data setup (STB_SKIP_DATA_SETUP=1) =="
    if [ "$DRY_RUN" != "1" ]; then
        if [ ! -d "$PDB_DATA_ROOT/mmcif_files" ]; then
            echo "[Error] missing PDB directory: $PDB_DATA_ROOT/mmcif_files" >&2
            exit 1
        fi
        if [ ! -d "$STB_RELEASE_DATA_ROOT/data" ]; then
            echo "[Error] missing STB release data: $STB_RELEASE_DATA_ROOT/data" >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
tokenizer="WrappedLaProteinaTokenizer"
lr=0.001

for experiment in "${SELECTED_EXPERIMENTS[@]}"; do
    experiment_model_args "$experiment"
    eval_variant=""
    if [[ "$EXTRA_MODEL_ARGS" == *"tokenizer_checkpoint_type=jepa"* ]]; then
        EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS tokenizer_encoder_source=$ENCODER_SOURCE"
        if [ "$ENCODER_SOURCE" = "target" ]; then
            eval_variant="${eval_variant}-target"
        fi
    elif [ "$ENCODER_SOURCE" != "online" ]; then
        echo "[Error] ENCODER_SOURCE=target is only valid for JEPA checkpoints ($experiment is not JEPA)" >&2
        exit 1
    fi
    if [ -n "$TOKENIZER_EVAL_PROTOCOL" ]; then
        EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS ++tokenizer_eval_protocol=$TOKENIZER_EVAL_PROTOCOL tokenizer_cache_tag=$TOKENIZER_CACHE_TAG"
        eval_variant="${eval_variant}-${TOKENIZER_EVAL_PROTOCOL}"
    elif [ -n "$TOKENIZER_EVAL_MAX_LENGTH" ]; then
        EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS tokenizer_eval_max_length=$TOKENIZER_EVAL_MAX_LENGTH"
        eval_variant="${eval_variant}-L${TOKENIZER_EVAL_MAX_LENGTH}"
    fi
    if [ -z "$TOKENIZER_EVAL_PROTOCOL" ] && [ -n "$TOKENIZER_CACHE_TAG" ]; then
        EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS tokenizer_cache_tag=$TOKENIZER_CACHE_TAG"
        eval_variant="${eval_variant}-cache-${TOKENIZER_CACHE_TAG}"
    fi
    TOKENIZERNAME="${TOKENIZERNAME}${eval_variant//-/_}"
    export EXPERIMENT_LABEL="${experiment}${eval_variant}"

    # d_model must match this experiment's own latent_z_dim (the downstream probe's
    # input width). Inferred from the checkpoint's own weights (same reasoning as
    # _infer_latent_z_dim in Evaluation/linear_probing/checkpoints.py) rather than
    # hardcoded -- ae checkpoints are fixed at 8 with no variation seen so far, so
    # only jepa/le-jepa ones (where dimension actually varies across checkpoints,
    # e.g. the dim64 experiments) pay for this peek.
    if [[ "$EXTRA_MODEL_ARGS" == *"tokenizer_checkpoint_type=jepa"* ]]; then
        ckpt_path="${EXTRA_MODEL_ARGS#*tokenizer_pretrained_ckpt_path=}"
        ckpt_path="${ckpt_path%% *}"
        if [ -n "${D_MODEL_OVERRIDE:-}" ]; then
            d_model="$D_MODEL_OVERRIDE"
        elif [ "$DRY_RUN" = "1" ]; then
            echo "[Error] DRY_RUN for a JEPA checkpoint requires D_MODEL_OVERRIDE (usually 8 or 64); checkpoint bytes are deliberately not loaded" >&2
            exit 1
        else
            d_model=$("$PY" -c "
import torch
ckpt = torch.load('$ckpt_path', map_location='cpu', weights_only=False)
w = None
for k, v in ckpt['state_dict'].items():
    if k.endswith('online_encoder.latent_decoder_mean_n_log_scale.1.weight'):
        w = v
        break
print(w.shape[0] // 2)
")
        fi
        echo "== $experiment: resolved d_model=$d_model for $ckpt_path =="
    else
        d_model=8
    fi
    for idx in "${SELECTED_TASK_IDX[@]}"; do
        config_name="${TASK_CONFIGS[$idx]}"
        target_field="${TASK_TARGET_FIELDS[$idx]}"
        experiment_prefix="${TASK_PREFIXES[$idx]}"
        extra_task_args="${TASK_EXTRA_ARGS[$idx]}"
        export TASK_LABEL="${TASK_NAMES[$idx]}"

        protocol_data_args=""
        resolved_run_name="tryout_test"
        if [ -n "$TOKENIZER_EVAL_PROTOCOL" ]; then
            protocol_data_args="data.filter_length=600"
            safe_experiment="${experiment//[^A-Za-z0-9_.-]/_}"
            safe_task="${TASK_NAMES[$idx]//[^A-Za-z0-9_.-]/_}"
            resolved_run_name="${EVALUATION_RUN_ID}_${safe_experiment}_${safe_task}"
        fi
        SHARED_ARGS="tokenizer=$tokenizer model.d_model=$d_model trainer.devices=[0] optimization.optimizer.lr=$lr data.target_field=$target_field data.num_workers=$NUM_WORKERS data.prefetch_factor=2 $protocol_data_args experiment_name=${experiment_prefix}_${TOKENIZERNAME}_lr${lr} run_name=$resolved_run_name default_data_dir=$STB_RELEASE_DATA_ROOT/ data.pdb_data_dir=$PDB_DATA_ROOT/mmcif_files/ trainer.default_root_dir=$STB_LOG_ROOT/ ${extra_task_args} ${EXTRA_MODEL_ARGS}"
        if [ -n "$PRECISION" ]; then
            SHARED_ARGS="$SHARED_ARGS trainer.precision=$PRECISION"
        fi
        if [ "$STB_SMOKE" = "1" ]; then
            # Token preprocessing still traverses the complete cohort and
            # emits exact counts; only the downstream probe is shortened.
            SHARED_ARGS="$SHARED_ARGS max_steps=2 trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=2 trainer.check_val_every_n_epoch=1 lightning.callbacks.checkpoint.every_n_epochs=1"
        fi

        echo
        echo "== Running: experiment=$experiment task=${TASK_NAMES[$idx]} (config=$config_name) =="
        CUDA_DEVICE_SPEC="${CUDA_VISIBLE_DEVICES:-0}"
        CMD="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_SPEC $PY ./src/script/run_supervised_task.py --config-name=$config_name $SHARED_ARGS"
        if [ "$DRY_RUN" = "1" ]; then
            echo "$CMD"
        else
            eval "$CMD"
        fi
    done
done

# ---------------------------------------------------------------------------
# 7. Summary across every (experiment, task) run this invocation performed
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" != "1" ] && [ -f "$SUMMARY_FILE" ]; then
    echo
    echo "== Summary (checkpoint: $CHECKPOINT_LABEL) =="
    if command -v column >/dev/null 2>&1; then
        column -t -s, "$SUMMARY_FILE"
    else
        cat "$SUMMARY_FILE"
    fi
fi
