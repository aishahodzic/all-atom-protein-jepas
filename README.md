## Setup (collaborators)

Prerequisites: conda, an NVIDIA GPU box. Check `nvcc --version` before step 2 — if it
isn't cu126, edit the cu126 references in `setup_env.sh`.

1. `git clone <repo-url> && cd msc-thesis`
2. `export DATA_PATH=/path/to/data`, `export HF_TOKEN=<token provided to you>`, then
   `conda activate py3.10 && bash setup_env.sh` (installs the env, downloads the
   processed PDB dataset, writes `DATA_PATH`/`HF_TOKEN` to `.env`)
3. `wandb login` (needs an invite to the `aisha-hodzic-org` wandb team first — ask Aisha)
4. `git checkout jepa-latent` (or another branch, see below)
5. `python proteinfoundation/jepa/train.py` — prompts for which checkpoint to resume
   from; pick `download` to pull the latest shared checkpoint from HF and continue
   training from it (needs `resume_hf_filename` set in `configs/training_jepa.yaml`)

For branches other than `jepa-latent` (which use `partial_autoencoder/train.py`, no
`download` option), fetch a checkpoint manually and point `pretrain_ckpt_path` at it:
```bash
hf download aishahodzic/pdb-checkpoints <filename> --repo-type dataset --local-dir "$DATA_PATH/checkpoints" --token "$HF_TOKEN"
```

Original NVIDIA La-Proteina repository and setup instructions:
https://github.com/NVIDIA-BioNeMo/la-proteina

Weights & Biases project:
https://wandb.ai/aisha-hodzic-org/protein_ae?nw=nwuseraishahodzic

Hugging face: 
https://huggingface.co/aishahodzic/datasets


## Experiment Branches

```text
main
├── ca-cond-baseline
├── fully-latent
└── jepa-latent
```

### Branch descriptions

- **main**: La Proteina codebase
- **ca-cond-baseline**: CA-conditioned autoencoder baseline used
- **fully-latent**: removes explicit CA-coordinate conditioning

## Autoencoder Experiments

### 1. Train AE with latent residues and PDB-given C-alpha (current setup)

Training:

```bash
sbatch jobs/train_ae.job
```
or 

```bash
python proteinfoundation/partial_autoencoder/train.py
```

## Training config

| | Value |
|---|---|
| Per-GPU batch size | 2 |
| `accumulate_grad_batches` | 2 |
| GPUs (DDP) | 2 |
| Effective global batch | 2 × 2 × 2 = 8 |
| Dataloader workers per rank | 12 (`ncpus_per_task_train_`) |
| Total dataloader worker processes | 24 (12 × 2 ranks) |
| Model size | Full PDB run keeps DiT-B (`token_dim: 768`, `nheads: 12`). Demo: DiT-B → DiT-S (`token_dim: 768→384`, `nheads: 12→6`).  |
| Sequence features | Removed `x1_bb_angles`, `x1_sidechain_angles` |
| `latent_z_dim` | 8 (kept) |
| `dist_strategy` | `ddp` → `auto` |



Evaluation:

```bash
sbatch jobs/inference_ae.job
```

```bash
python proteinfoundation/partial_autoencoder/inference.py --config_name inference_ae
```

Checkpoint:

```text
store/{release-name}/checkpoints/last-EMA.ckpt
```

### 2. Train fully latent AE

Evaluate structure reconstruction and sequence recovery.

Compare both setups. Initial expectation: the fully latent AE will perform worse, since the decoder no longer receives the true C-alpha coordinates.

## Demo Results (data fraction: 0.001)

Metrics:

* Structure reconstruction (`all_atom_rmsd`)
* Sequence recovery (`seq_rec_rate`)

Results: 

Evaluated on 4 protein structures. Total data size: 664. 

| Model | 60k RMSD | 60k Seq Rec. | 100k RMSD | 100k Seq Rec. |
|---------|---------:|---------:|---------:|---------:|
| CA-conditioned AE | 0.274 Å | 1.000 | 0.241 Å | 1.000 |
| Fully Latent AE | 0.420 Å | 1.000 | 0.371 Å | 1.000 |

## Results on full PDB 

Total structures: 344,065. Training: 338,787, val: 4,775, test: 503.

<img width="747" height="602" alt="Screenshot 2026-07-09 at 14 10 46" src="https://github.com/user-attachments/assets/de4d1d5d-b272-409c-840a-2e417f3881ff" />



### 3. Evaluate both latent representations on a downstream task
