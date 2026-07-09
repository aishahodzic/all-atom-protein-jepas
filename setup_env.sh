#!/bin/bash
# Rebuilds the py3.10 conda env from scratch for training (jepa / partial_autoencoder / etc).
# Captures the fixes needed to get proteinfoundation/jepa/train.py running on a fresh
# GPU instance with CUDA 12.6 toolkit installed (e.g. a fresh Jarvis Labs box).
#
# Usage: conda activate py3.10 && bash setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "== Installing pip requirements =="
pip install -r condaenv.83f118h7.requirements.txt

echo "== Installing lightning, wandb, python-dotenv, loguru =="
pip install "lightning==2.5.0" wandb python-dotenv loguru

echo "== Installing torch matching the system's CUDA toolkit (cu126) =="
# NOTE: check `nvcc --version` first. If the box has a different CUDA toolkit,
# swap cu126 below (and in the torch_geometric line) for the matching build.
pip install --force-reinstall --no-deps torch==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# torchvision version must match the torch minor version (0.26.x <-> torch 2.11.x).
# Check https://github.com/pytorch/vision#installation for the current pairing.
pip install --force-reinstall --no-deps torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126

echo "== Installing PyG extensions (prebuilt wheels matching torch+cu126) =="
pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cu126.html

echo "== Installing graphein (no-deps, per environment.yaml) =="
pip install graphein==1.7.7 --no-deps

echo "== Installing mmseqs2 (needed for sequence-similarity dataset splitting) =="
conda install -y --override-channels -c conda-forge -c bioconda mmseqs2

echo "== Working around dead ligand-expo.rcsb.org download (graphein PDBManager) =="
# graphein's PDBManager tries to download a ligand map from ligand-expo.rcsb.org on
# first run. That host/port is unreachable from this box and the data isn't needed
# (has_ligands/remove_ligands are unused in our configs), so drop in an empty file
# so PDBManager sees it as already downloaded and skips it.
: "${DATA_PATH:?Set DATA_PATH before running this script, e.g. export DATA_PATH=/home/data}"
mkdir -p "$DATA_PATH/pdb_train"
touch "$DATA_PATH/pdb_train/cc-to-pdb.tdd"

echo "== Downloading pre-processed PDB dataset from Hugging Face (skips re-downloading/re-processing from RCSB) =="
if [ -d "$DATA_PATH/pdb_train/processed" ] && [ -n "$(ls -A "$DATA_PATH/pdb_train/processed" 2>/dev/null)" ]; then
    echo "Processed dataset already present at $DATA_PATH/pdb_train/processed, skipping download."
else
    : "${HF_TOKEN:?Set HF_TOKEN before running this script, e.g. export HF_TOKEN=hf_...}"
    hf download aishahodzic/pdb-processed processed.tar.gz --repo-type dataset --local-dir "$DATA_PATH" --token "$HF_TOKEN"
    tar -xzf "$DATA_PATH/processed.tar.gz" -C "$DATA_PATH/"
    mv "$DATA_PATH/processed" "$DATA_PATH/pdb_train/processed"
    rm "$DATA_PATH/processed.tar.gz"
fi

echo "== Checkpoint resume =="
read -rp "Resume training from an existing checkpoint on Hugging Face, or train from scratch? [resume/scratch]: " CKPT_CHOICE
if [[ "$CKPT_CHOICE" == "resume" ]]; then
    : "${HF_TOKEN:?Set HF_TOKEN before running this script, e.g. export HF_TOKEN=hf_...}"
    read -rp "HF repo id containing the checkpoint (e.g. aishahodzic/pdb-checkpoints): " CKPT_REPO
    read -rp "Checkpoint filename in that repo (e.g. chk_epoch=...-EMA.ckpt): " CKPT_FILE
    mkdir -p "$DATA_PATH/checkpoints"
    hf download "$CKPT_REPO" "$CKPT_FILE" --repo-type dataset --local-dir "$DATA_PATH/checkpoints" --token "$HF_TOKEN"
    echo "Downloaded checkpoint to $DATA_PATH/checkpoints/$CKPT_FILE"
    echo "Set pretrain_ckpt_path: $DATA_PATH/checkpoints/$CKPT_FILE in your training config (e.g. training_local_latents.yaml) to resume from it."
else
    echo "Training from scratch (leave pretrain_ckpt_path unset in your training config)."
fi

echo "== Done. Verify with: =="
echo "  python -c \"import torch, torchvision, lightning, wandb, graphein, torch_scatter; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
