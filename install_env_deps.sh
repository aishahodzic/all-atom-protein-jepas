#!/bin/bash
# Installs training dependencies into the already-activated py3.10 conda env and
# downloads the pre-processed PDB dataset. Not meant to be run/sourced directly —
# invoked by setup_env.sh as a subprocess so a failure here can't kill the shell.

set -euo pipefail

cd "$(dirname "$0")"

echo "== Installing pip requirements =="
pip install -r condaenv.83f118h7.requirements.txt

echo "== Installing lightning, wandb, python-dotenv, loguru =="
pip install "lightning==2.5.0" wandb python-dotenv loguru

echo "== Installing torch matching the system's CUDA toolkit (cu126) =="
pip install --force-reinstall --no-deps torch==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install --force-reinstall --no-deps torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126

echo "== Installing PyG extensions (prebuilt wheels matching torch+cu126) =="
pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cu126.html

echo "== Installing graphein (no-deps, per environment.yaml) =="
pip install graphein==1.7.7 --no-deps

echo "== Installing mmseqs2 (needed for sequence-similarity dataset splitting) =="
conda install -y --override-channels -c conda-forge -c bioconda mmseqs2

echo "== Working around dead ligand-expo.rcsb.org download (graphein PDBManager) =="
: "${DATA_PATH:?Set DATA_PATH before running this script, e.g. export DATA_PATH=/home/data}"
mkdir -p "$DATA_PATH/pdb_train"
touch "$DATA_PATH/pdb_train/cc-to-pdb.tdd"

echo "== Downloading pre-processed PDB dataset from Hugging Face (skips re-downloading/re-processing from RCSB) =="
if [ -d "$DATA_PATH/pdb_train/processed" ] && [ -n "$(ls -A "$DATA_PATH/pdb_train/processed" 2>/dev/null)" ]; then
    echo "Processed dataset already present at $DATA_PATH/pdb_train/processed, skipping download."
else
    if [ -z "${HF_TOKEN:-}" ]; then
        read -rsp "Please enter your HF read token: " HF_TOKEN
        echo
    fi
    hf download msc-thesis-org/pdb-processed processed.tar.gz --repo-type dataset --local-dir "$DATA_PATH" --token "$HF_TOKEN"
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
