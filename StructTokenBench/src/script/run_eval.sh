#!/bin/bash
# Single entrypoint for LaProteina tokenizer evaluation on StructTokenBench.
#
# Usage:
#   bash src/script/run_eval.sh
#   DRY_RUN=1 bash src/script/run_eval.sh          # print the resolved commands instead of running them
#   NUM_WORKERS=4 bash src/script/run_eval.sh       # override DataLoader workers (default 8)
#   PRECISION=bf16-mixed bash src/script/run_eval.sh  # opt into mixed precision (default: 32-true)
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

# src/tokenizer.py imports the `openfold` package, which lives in the la-proteina
# repo root (one level up) rather than being pip-installed.
export PYTHONPATH="$(dirname "$REPO_DIR")${PYTHONPATH:+:$PYTHONPATH}"

export DIR="$REPO_DIR"
DRY_RUN="${DRY_RUN:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"          # DataLoader workers per task run (nproc-limited; see README perf notes)
PRECISION="${PRECISION:-}"               # opt-in, e.g. PRECISION=16-mixed or bf16-mixed; default keeps trainer.yaml's 32-true

# ---------------------------------------------------------------------------
# CLI args (optional): --checkpoint=, --experiment=, --tasks=. Equivalent to
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
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "[Error] unknown argument: $arg (expected --checkpoint=, --experiment=, --tasks=)" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Env activation
# ---------------------------------------------------------------------------
source /slurm-storage/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^pstbench '; then
    echo "  conda env 'pstbench' not found; creating it (python=3.11) -- one-time, ~1min"
    conda create -y -n pstbench python=3.11
fi
conda activate pstbench
hash -r

PY=python
PIP=pip

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
CKPT_ROOT="$REPO_DIR/100k-checkpoints"
JEPA_ENCODER_CFG="$CKPT_ROOT/shared/exp_config_jepa_nn130m.json"
# External reference checkpoint (NVIDIA's published La-Proteina AE, not one of "my 8") --
# lives at the la-proteina repo root, not under 100k-checkpoints/. Same latent_z_dim=8,
# loaded via the plain "ae" path (no encoder-cfg-path), verified via a one-protein load
# test (encode_structure OK, feats_seq auto-includes x1_bb_angles/x1_sidechain_angles
# since WrappedLaProteinaTokenizer reads each checkpoint's own saved config).
AE1_CKPT="$(dirname "$REPO_DIR")/checkpoints_laproteina/AE1_ucond_512.ckpt"

EXPERIMENT_NAMES=(ca-cond-baseline fully-latent jepa-rand-mask-50 jepa-new-predictor jepa-rand-mask-70 jepa-mutation le-jepa-rand-mask-50 le-jepa-rand-mask-70 le-jepa-mutation jepa-rand-mask-70-dim64 le-jepa-rand-mask-70-dim64 jepa-rand-mask-50-selfattn jepa-rand-mask-70-selfattn le-jepa-rand-mask-50-200k ae1-external random-init jepa-rand50-control-30k jepa-rand50-ema-30k jepa-rand50-layernorm-30k jepa-rand50-predictor-30k)

experiment_model_args () {
    # sets TOKENIZERNAME and EXTRA_MODEL_ARGS for experiment name $1
    case "$1" in
        ca-cond-baseline)
            TOKENIZERNAME="laproteina_ca_cond"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/ae/ca-cond/100k-EMA.ckpt tokenizer_checkpoint_type=ae"
            ;;
        fully-latent)
            TOKENIZERNAME="laproteina_fully_latent"
            # HF only has a "last-EMA" checkpoint (~103k steps); treated as the 100k snapshot.
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/ae/fully_latent/100k-EMA.ckpt tokenizer_checkpoint_type=ae"
            ;;
        jepa-rand-mask-50)
            TOKENIZERNAME="laproteina_jepa_rand_mask_50"
            # placeholder: jepa_random_masking_50_v1 hasn't reached step 100000 yet
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/jepa/random_masking_50pct/100k.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
            ;;
        jepa-new-predictor)
            TOKENIZERNAME="laproteina_jepa_new_predictor"
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$CKPT_ROOT/jepa/new_predictor/100k/chk_epoch=00000019_step=000000100000.ckpt tokenizer_checkpoint_type=jepa tokenizer_encoder_cfg_path=$JEPA_ENCODER_CFG"
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
            EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$AE1_CKPT tokenizer_checkpoint_type=ae +tokenizer_force_remove_non_standard_residues=true"
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
PHYSCHEM_EXTRA="data.pdb_data_dir=$DIR/struct_token_bench_release_data/data/physicochemical/ lightning.callbacks.checkpoint.monitor='validation_spearmanr'"
TASK_EXTRA_ARGS=("" "" "" "" "" "" "" "" "$PHYSCHEM_EXTRA" "$PHYSCHEM_EXTRA" "$PHYSCHEM_EXTRA" "optimization.micro_batch_size=64")

if [ -z "$CHECKPOINT_LABEL" ]; then
    echo
    echo "== What checkpoint step should the results table/summary be labeled with? (e.g. 100k, 200k) =="
    read -r CHECKPOINT_LABEL
fi
export CHECKPOINT_LABEL

SUMMARY_FILE="$DIR/struct_token_bench_logs/summary_results.csv"
mkdir -p "$(dirname "$SUMMARY_FILE")"
export SUMMARY_FILE

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

# ---------------------------------------------------------------------------
# 3. Base StructTokenBench deps (README), skipping packages that get a
#    version-pinned reinstall in step 3 anyway (torch itself, torch-scatter,
#    torch-cluster) — installing them here first would either be wasted work
#    or risk the same CUDA/torch ABI-mismatch class of failure step 3 exists
#    to avoid.
# ---------------------------------------------------------------------------
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
    [torch_geometric]="torch_geometric"
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
# 4. LaProteina-specific stack: pinned torch build + matching PyG extensions
#    + graphein/openfold/proteinfoundation runtime deps.
# ---------------------------------------------------------------------------
echo "== Checking LaProteina-specific dependencies =="
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ "$TORCH_VER" != "2.7.0+cu118" ]; then
    echo "  torch is '$TORCH_VER', pinning to 2.7.0+cu118 (+matching torchvision)"
    $PIP install "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu118
    $PIP install torch_geometric torch_scatter torch_sparse torch_cluster \
        -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
fi

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
if [ "$(hostname -s)" = "hopper" ]; then
    PDB_DATA_ROOT="/mnt/raid/aishod/data/struct_token_bench_pdb_data"
else
    PDB_DATA_ROOT="$DIR/pdb_data"
fi
export PDB_DATA_ROOT

# Guarded by a flock so concurrent eval jobs sharing $DIR over NFS don't all
# probe/touch this section at once.
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

    LABELS_DIR="$DIR/struct_token_bench_release_data/data"
    if [ -d "$LABELS_DIR" ] && [ -n "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
        echo "== Downstream label data already present, skipping download =="
    else
        echo "== Downloading downstream label data =="
        (
            cd "$DIR"
            gdown https://drive.google.com/uc?id=1wJ4dSNdMyuF0985ET4UuwViHgV-clF4K
            unzip -q struct_token_bench_release_data_download.zip
            mv struct_token_bench_release_data_download struct_token_bench_release_data
            rm -f struct_token_bench_release_data_download.zip
        )
    fi
) 200>"$DIR/.data_setup.lock"

# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
tokenizer="WrappedLaProteinaTokenizer"
lr=0.001

for experiment in "${SELECTED_EXPERIMENTS[@]}"; do
    experiment_model_args "$experiment"
    export EXPERIMENT_LABEL="$experiment"

    # d_model must match this experiment's own latent_z_dim (the downstream probe's
    # input width). Inferred from the checkpoint's own weights (same reasoning as
    # _infer_latent_z_dim in Evaluation/linear_probing/checkpoints.py) rather than
    # hardcoded -- ae checkpoints are fixed at 8 with no variation seen so far, so
    # only jepa/le-jepa ones (where dimension actually varies across checkpoints,
    # e.g. the dim64 experiments) pay for this peek.
    if [[ "$EXTRA_MODEL_ARGS" == *"tokenizer_checkpoint_type=jepa"* ]]; then
        ckpt_path=$(echo "$EXTRA_MODEL_ARGS" | grep -oP 'tokenizer_pretrained_ckpt_path=\K[^ ]+')
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
        echo "== $experiment: inferred d_model=$d_model from $ckpt_path =="
    else
        d_model=8
    fi
    for idx in "${SELECTED_TASK_IDX[@]}"; do
        config_name="${TASK_CONFIGS[$idx]}"
        target_field="${TASK_TARGET_FIELDS[$idx]}"
        experiment_prefix="${TASK_PREFIXES[$idx]}"
        extra_task_args="${TASK_EXTRA_ARGS[$idx]}"
        export TASK_LABEL="${TASK_NAMES[$idx]}"

        SHARED_ARGS="tokenizer=$tokenizer model.d_model=$d_model trainer.devices=[0] optimization.optimizer.lr=$lr data.target_field=$target_field data.num_workers=$NUM_WORKERS data.prefetch_factor=2 experiment_name=${experiment_prefix}_${TOKENIZERNAME}_lr${lr} run_name=tryout_test default_data_dir=$DIR/struct_token_bench_release_data/ data.pdb_data_dir=$PDB_DATA_ROOT/mmcif_files/ trainer.default_root_dir=$DIR/struct_token_bench_logs/ ${extra_task_args} ${EXTRA_MODEL_ARGS}"
        if [ -n "$PRECISION" ]; then
            SHARED_ARGS="$SHARED_ARGS trainer.precision=$PRECISION"
        fi

        echo
        echo "== Running: experiment=$experiment task=${TASK_NAMES[$idx]} (config=$config_name) =="
        CMD="CUDA_VISIBLE_DEVICES=0 $PY ./src/script/run_supervised_task.py --config-name=$config_name $SHARED_ARGS"
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
    column -t -s, "$SUMMARY_FILE"
fi
