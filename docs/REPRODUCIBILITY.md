# End-to-end reproducibility guide

This guide is the operational handoff for the thesis and paper. It follows the
project lifecycle:

```text
environment + data -> training -> checkpoint archive -> downstream evaluation
                   -> representation analysis -> reconstruction -> results
```

Commands are run from the repository root unless stated otherwise.

## 1. Clone and authenticate

```bash
git clone https://github.com/aishahodzic/msc-thesis.git
cd msc-thesis
git switch main

hf auth login
wandb login  # optional unless training or logging new evaluations
```

Hugging Face access is required for the private checkpoint and processed-data
datasets. Authentication is read from the normal Hugging Face credential store
or `HF_TOKEN`. Never commit a token to `.env`, a job script, or Git.

- checkpoints: <https://huggingface.co/datasets/msc-thesis-org/ckpts>
- processed training data: <https://huggingface.co/datasets/msc-thesis-org/pdb-processed>

## 2. Environment and training data

Training uses Python 3.10 and the original La-Proteina dependency set. The
provided installer targets CUDA 12.6, so check the machine first:

```bash
nvcc --version
export DATA_PATH=/path/with/sufficient/disk
source setup_env.sh
```

`setup_env.sh` must be sourced because it activates the environment in the
current shell. It invokes `install_env_deps.sh`, downloads the processed PDB
dataset when absent, and writes local settings to the ignored `.env` file. If
the machine does not use CUDA 12.6, adjust the PyTorch wheel URLs in
`install_env_deps.sh` first.

The resulting layout is:

```text
$DATA_PATH/
  pdb_train/
    processed/
    cc-to-pdb.tdd
```

Manual data download:

```bash
hf download msc-thesis-org/pdb-processed processed.tar.gz \
  --repo-type dataset --local-dir "$DATA_PATH"
```

StructTokenBench uses Python 3.11 and separate downstream datasets. Its
launcher can bootstrap them, or an existing environment can be selected with
`STB_SKIP_BOOTSTRAP=1 PYTHON_BIN=/path/to/python`. See
[`StructTokenBench/README.md`](../StructTokenBench/README.md) for task and data
definitions.

## 3. Training

### Preserved training branches

The exact historical model families were developed on separate branches.
`main` is the evaluation/results handoff; each training branch preserves its
model code and Hydra defaults. Use worktrees so `main` remains available:

```bash
git worktree add ../msc-thesis-ca-cond ca-cond-baseline
git worktree add ../msc-thesis-fully-latent fully-latent
git worktree add ../msc-thesis-jepa jepa-latent
git worktree add ../msc-thesis-sigreg le-jepa
git worktree add ../msc-thesis-canonical canonical-lejepa
```

Training writes to `store/<run_name>/checkpoints/`. Checkpoints record their
real `global_step`; do not assume `last.ckpt` is a particular step.

### Autoencoders

| Model | Worktree/branch | Command |
|---|---|---|
| C-alpha-conditioned AE | `ca-cond-baseline` | `python proteinfoundation/partial_autoencoder/train.py` |
| Fully latent AE | `fully-latent` | `python proteinfoundation/partial_autoencoder/train.py` |

Example:

```bash
cd ../msc-thesis-fully-latent
python proteinfoundation/partial_autoencoder/train.py \
  hardware.ngpus_per_node_=1 \
  hardware.nnodes_=1 \
  hardware.ncpus_per_task_train_=8
```

The launcher prompts for `scratch`, a local checkpoint, or a registered remote
checkpoint. Autoencoder training writes raw and EMA variants.

### JEPA and JEPA+SIGReg

```bash
# Standard JEPA experiments
cd ../msc-thesis-jepa
python proteinfoundation/jepa/train.py

# Historical JEPA+SIGReg experiments
cd ../msc-thesis-sigreg
python proteinfoundation/jepa/train.py
```

The exact random masking, mutation, chemistry, relative-position, rotation,
and latent-dimension overrides are preserved in `jobs/train_*.job` on the
corresponding branch. Prefer the relevant job file over reconstructing a long
Hydra command manually.

### Canonical LeJEPA

Canonical LeJEPA has a dedicated entry point and locked configuration on the
`canonical-lejepa` branch:

```bash
cd ../msc-thesis-canonical
python -m proteinfoundation.jepa.train_canonical_lejepa \
  trainer.devices=1 \
  trainer.strategy=auto
```

Production launcher:

```bash
sbatch jobs/train_canonical_lejepa_v1_100k_a10.job
```

Canonical checkpoints contain the normal encoder in `state_dict` and its EMA
copy in `state_dict_ema`. Resume with `CANONICAL_RESUME_CKPT` in the Slurm job
or `resume.ckpt_path=/path/to/checkpoint.ckpt` as a Hydra override.

### Slurm portability

`jobs/` is both a launcher library and an experiment ledger. Many files retain
the original cluster's partitions, environment paths, and storage paths. On a
different cluster, change the resource/filesystem header but preserve the
training command, Hydra overrides, seed, and checkpoint selection.

## 4. Retrieve and stage evaluated checkpoints

The canonical private archive is
[`msc-thesis-org/ckpts`](https://huggingface.co/datasets/msc-thesis-org/ckpts).
Its `manifest.csv` records manuscript labels, historical run names, global
steps, latent dimensions, evaluated parameter sources, sizes, and SHA-256
checksums. SHA-256 verifies that a downloaded checkpoint is byte-for-byte
identical to the archived file; it is unrelated to the L256/L600 evaluation
protocols.

Stage the core AE/JEPA/JEPA+SIGReg models expected by `main`:

```bash
git switch main
python script_utils/stage_thesis_checkpoints.py --group core
```

The helper downloads through the Hugging Face cache, verifies the manifest,
creates compatibility links below `StructTokenBench/100k-checkpoints/`, and
installs the frozen JEPA encoder configuration. Checkpoints are not duplicated.

Preview mappings without downloading:

```bash
python script_utils/stage_thesis_checkpoints.py --group core --dry-run
```

Stage selected models:

```bash
python script_utils/stage_thesis_checkpoints.py \
  --model ca-cond \
  --model fully-latent \
  --model jepa-random-70
```

Direct downloads remain available:

```bash
hf download msc-thesis-org/ckpts manifest.csv --repo-type dataset

hf download msc-thesis-org/ckpts \
  jepa_sigreg/relative/random-70/100k.ckpt \
  --repo-type dataset --local-dir artifacts/hf/ckpts
```

Manifest terminology:

- `raw`: ordinary encoder parameters from the checkpoint state dictionary;
- `ema`: EMA parameters, in a separate AE checkpoint or `state_dict_ema` for
  canonical LeJEPA;
- `online`: the JEPA student encoder below `online_encoder.`;
- `[raw, ema]`: both sources were evaluated from one stored full checkpoint.

Use the archive path and manifest as stable identifiers; historical run names
are retained only for provenance.

## 5. StructTokenBench evaluation

The final benchmark uses the locked `upstream-l600-v1` protocol. Preview the
resolved commands first:

```bash
cd StructTokenBench
DRY_RUN=1 bash src/script/run_eval.sh \
  --protocol=upstream-l600-v1 \
  --checkpoint=100k \
  --experiment=jepa-rand-mask-70 \
  --tasks=all
```

Then run it:

```bash
bash src/script/run_eval.sh \
  --protocol=upstream-l600-v1 \
  --checkpoint=100k \
  --experiment=jepa-rand-mask-70 \
  --tasks=all
```

Task names are `BindInt`, `BindBio`, `BindShake`, `CatInt`, `CatBio`, `Con`,
`Rep`, `Ept`, `FlexRMSF`, `FlexBFactor`, `FlexNEQ`, and `Homo`. A comma-separated
subset can be supplied with `--tasks=`.

For cluster runs, use the matching `jobs/eval_*_upstream_l600*.job`. These jobs
lock hashes and output paths but retain original cluster-specific paths.

Useful controls:

```bash
STB_SMOKE=1                 # full cohort preprocessing, tiny downstream probe
NUM_WORKERS=4               # reduce loader concurrency
PRECISION=bf16-mixed        # opt into mixed precision
STB_SKIP_BOOTSTRAP=1        # use an existing environment
PYTHON_BIN=/path/to/python  # interpreter with STB_SKIP_BOOTSTRAP=1
```

Outputs are below `StructTokenBench/struct_token_bench_logs/`. The saved final
L600 summaries and manifests are in
`StructTokenBench/struct_token_bench_logs/upstream-l600-v1-full/`; embedding
caches and probe checkpoints can be regenerated.

## 6. Representation analyses

Run these from the repository root after checkpoint staging.

```bash
# Linear probes
python Evaluation/linear_probing/run.py --no-wandb --seed 42

# k-NN purity
python Evaluation/knn_purity/run.py --no-wandb --seed 42

# Training-free fold retrieval
python Evaluation/fold_retrieval/run.py --no-wandb --device cuda

# t-SNE from cached full-context residue embeddings
python Evaluation/latent_viz/run_tsne.py \
  --checkpoints ca-cond fully-latent jepa-rand-mask-70 \
  --n-points 15000 --seed 42
```

Outputs:

- `Evaluation/linear_probing/results/linear_probe_results.csv`
- `Evaluation/knn_purity/results/knn_purity_results.csv`
- `Evaluation/fold_retrieval/results/fold_retrieval_results.csv`
- `Evaluation/latent_viz/results/` and PNGs in `Evaluation/latent_viz/figures/`

t-SNE panels have arbitrary orientation and scale. Use original-dimensional
k-NN/probe results for quantitative comparisons.

The archived analysis inventory is documented in
[`Evaluation/ANALYSIS_ARCHIVE.md`](../Evaluation/ANALYSIS_ARCHIVE.md) and
[`Evaluation/analysis_source_manifest.json`](../Evaluation/analysis_source_manifest.json).

## 7. Reconstruction

Reconstruction trains a new decoder against a frozen encoder; the archived
encoder is not modified:

```bash
python Evaluation/structure_rec/train_decoder.py \
  --encoder jepa-rand-mask-70 \
  --max-steps 10000 \
  --eval-every 2000 \
  --no-wandb
```

Decoder weights are intermediate evaluation artifacts and are not in the
encoder archive. Retained learning curves are in
`Evaluation/structure_rec/curves/`.

Download the ten qualitative reconstructions separately:

```bash
hf download msc-thesis-org/ckpts \
  --repo-type dataset \
  --include 'pymol_reconstruction/**' \
  --local-dir artifacts/hf/ckpts
```

These are PDB structures and audit metadata for PyMOL, not model weights.

## 8. Results and manuscript assets

| Artifact | Location |
|---|---|
| Linear probes | `Evaluation/linear_probing/results/` |
| k-NN purity | `Evaluation/knn_purity/results/` |
| Fold retrieval | `Evaluation/fold_retrieval/results/` |
| Latent-space metrics/manifests | `Evaluation/latent_viz/results/` |
| Canonical LeJEPA diagnostics | `Evaluation/diagnostics/results/` and `Evaluation/canonical_lejepa/results/` |
| Reconstruction curves | `Evaluation/structure_rec/curves/` |
| StructTokenBench L600 summaries | `StructTokenBench/struct_token_bench_logs/upstream-l600-v1-full/` |
| Other StructTokenBench summaries | `StructTokenBench/struct_token_bench_logs/results/` and retained run summaries |
| Curated thesis and paper figures | `images/` |

Git retains compact CSV/JSON manifests, final PNG figures, and manuscript
sources. It excludes checkpoints, latent caches, downstream probe weights,
bulk PDB exports, and LaTeX build products.

## 9. Integrity checklist

Before treating a new run as comparable to the archive:

1. record the Git commit and branch;
2. record the checkpoint archive path and SHA-256;
3. confirm the checkpoint's internal `global_step`;
4. record `raw`, `ema`, `online`, or target selection;
5. use `upstream-l600-v1` for the final StructTokenBench protocol;
6. retain the generated CSV/JSON manifest and final PNG only;
7. never commit credentials, weights, caches, or temporary data.
