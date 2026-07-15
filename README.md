## Setup

Prerequisites: conda, an NVIDIA GPU box. 

1. `git clone https://github.com/aishahodzic/msc-thesis.git && cd msc-thesis`
2. `export DATA_PATH={/path/to/data} (e.g. export DATA_PATH=/home/data)`,
3. `export HF_TOKEN=<your-token>`
4. `source setup_env.sh` 
5. `wandb login` (needs an invite to the `aisha-hodzic-org` wandb team. If command fails, run `export WANDB_API_KEY=<your-API-key>`)
6. `git checkout <branch>` - each branch is a different experiment:

   | Branch | Train command |
   |---|---|
   | `ca-cond-baseline` | `python proteinfoundation/partial_autoencoder/train.py` |
   | `fully-latent` | `python proteinfoundation/partial_autoencoder/train.py` |
   | `jepa-latent` | `python proteinfoundation/jepa/train.py` |

7. Run the matching train command above. It prompts from which checkpoint to resume: pick `last` to pull the latest shared checkpoint or `scratch` to start fresh.

### Hardware
Defined in training_ae.yaml (for AE) and training_jepa.yaml (for Jepa).  
```
hardware:
  ncpus_per_task_train_: 24  # Number of CPUs per tast during training
  ncpus_per_task_prepro_: 32  # Number of CPUs used for data preprocessing run
  accelerator: gpu
  ngpus_per_node_: 2  # Number of GPUs per node
  nnodes_: 1  # Number of nodes
```

### Uploading checkpoints

After training, upload `last.ckpt` (and `last-EMA.ckpt` for AE training) to HF:

```bash
export HF_TOKEN=<your-write-token>

huggingface-cli upload msc-thesis-org/checkpoints \
  store/<run_name>/checkpoints/last.ckpt <hf-path>/last.ckpt \
  --repo-type dataset --token $HF_TOKEN
```

`<hf-path>` depends on the branch:

| Branch | `<hf-path>` |
|---|---|
| `ca-cond-baseline` | `msc-thesis-org/checkpoints/ae/ca-cond` |
| `fully-latent` | `msc-thesis-org/checkpoints/ae/fully_latent` |
| `jepa-latent` | `msc-thesis-org/checkpoints/jepa/random_masking` |

### Links

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

#### Branch descriptions:

- **main**: La Proteina codebase
- **ca-cond-baseline**: CA-conditioned autoencoder baseline used
- **fully-latent**: removes explicit CA-coordinate conditioning
- **jepa-latent**: JEPA with random masking

## Experiments

| Stage | Where | Command |
|---|---|---|
| Training | Cluster | `sbatch jobs/train_ae.job` |
| | Local | `python proteinfoundation/partial_autoencoder/train.py` |
| Evaluation | Cluster | `sbatch jobs/inference_ae.job` |
| | Local | `python proteinfoundation/partial_autoencoder/inference.py --config_name inference_ae` |


Checkpoints at:

```text
store/{release-name}/checkpoints/last-EMA.ckpt
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

