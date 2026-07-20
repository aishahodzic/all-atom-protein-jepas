#!/bin/bash
# Installs training dependencies into the already-activated micromamba env and
# downloads the pre-processed PDB dataset. Not meant to be run/sourced directly —
# invoked by setup_env.sh as a subprocess so a failure here can't kill the shell.

set -euo pipefail

cd "$(dirname "$0")"

echo "== Installing torch and bundled CUDA 12.6 runtime =="
# Install this first so Lightning's broad torch dependency does not select a newer
# CUDA build. Keep dependencies enabled: CoreWeave login nodes do not expose a
# system CUDA runtime, so the wheel's nvidia-* runtime packages are required.
python -m pip install --force-reinstall torch==2.11.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
# torchvision version must match the torch minor version (0.26.x <-> torch 2.11.x).
python -m pip install --force-reinstall torchvision==0.26.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

echo "== Installing pip requirements =="
python -m pip install -r condaenv.83f118h7.requirements.txt

echo "== Installing lightning, wandb, python-dotenv, loguru =="
python -m pip install "lightning==2.5.0" "transformers==4.48.3" \
    "huggingface_hub[cli]>=0.34,<1.0" "fsspec[http]<2026.0" \
    bioservices wandb python-dotenv loguru

echo "== Installing PyG extensions (prebuilt wheels matching torch+cu126) =="
python -m pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cu126.html

echo "== Installing graphein (no-deps, per environment.yaml) =="
python -m pip install graphein==1.7.7 --no-deps

echo "== Installing mmseqs2 (needed for sequence-similarity dataset splitting) =="
micromamba install -y -n "${LAPROTEINA_ENV_NAME:-laproteina_env}" \
    --override-channels -c conda-forge -c bioconda mmseqs2

if [ "${SKIP_DATA_SETUP:-0}" = "1" ]; then
    echo "== SKIP_DATA_SETUP=1; dependency installation complete =="
    exit 0
fi

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
    if [ -n "${HF_TOKEN:-}" ]; then
        hf download msc-thesis-org/pdb-processed processed.tar.gz \
            --repo-type dataset --local-dir "$DATA_PATH" --token "$HF_TOKEN"
    else
        echo "HF_TOKEN is unset; trying cached Hugging Face authentication."
        hf download msc-thesis-org/pdb-processed processed.tar.gz \
            --repo-type dataset --local-dir "$DATA_PATH"
    fi
    tar -xzf "$DATA_PATH/processed.tar.gz" -C "$DATA_PATH/"
    mv "$DATA_PATH/processed" "$DATA_PATH/pdb_train/processed"
    rm "$DATA_PATH/processed.tar.gz"
fi

echo "== Writing DATA_PATH, HF_TOKEN to .env so train.py picks these up automatically =="
touch .env
grep -v '^DATA_PATH=\|^HF_TOKEN=' .env > .env.tmp || true
mv .env.tmp .env
{
    echo "DATA_PATH=$DATA_PATH"
    echo "HF_TOKEN=${HF_TOKEN:-}"
} >> .env

echo "== Env ready. Checkpoint resume is now handled by train.py's interactive prompt, not this script. =="
echo "== Verify install with: =="
echo "  python -c \"import torch, torchvision, lightning, wandb, graphein, torch_scatter; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
