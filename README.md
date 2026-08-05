## Setup

Prerequisites: conda, an NVIDIA GPU box. 

1. `git clone https://github.com/aishahodzic/msc-thesis.git && cd msc-thesis`
2. `export DATA_PATH={/path/to/data} (e.g. export DATA_PATH=/home/data)`,
3. `export HF_TOKEN=<your-token>`
4. `bash setup_env.sh` 
5. `wandb login` (needs an invite to the `aisha-hodzic-org` wandb team. If command fails, run `export WANDB_API_KEY=<your-API-key>`)
6. `git checkout <branch>` - each branch is a different experiment:

   | Branch | Train command |
   |---|---|
   | `ca-cond-baseline` | `python proteinfoundation/partial_autoencoder/train.py` |
   | `fully-latent` | `python proteinfoundation/partial_autoencoder/train.py` |
   | `jepa-latent` (50% random mask) | `python proteinfoundation/jepa/train.py` |
   | `jepa-latent` (70% random mask) | `python proteinfoundation/jepa/train.py +masked_pct=0.7` |
   | `jepa-latent` (mutation action) | `python proteinfoundation/jepa/train.py +use_action_conditioning=true` |
   | `le-jepa` (50% random mask) | `python proteinfoundation/jepa/train.py` |
   | `le-jepa` (70% random mask) | `python proteinfoundation/jepa/train.py +masked_pct=0.7` |
   | `le-jepa` (mutation action) | `python proteinfoundation/jepa/train.py +use_action_conditioning=true` |

8. Change the hardware settings (optional) and run the matching train command above. It prompts from which checkpoint to resume: pick `last` to pull the latest shared checkpoint or `scratch` to start fresh. `le-jepa` (50% random mask) has no shared checkpoint yet, so pick `scratch`.

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

hf upload msc-thesis-org/checkpoints \
  store/<run_name>/checkpoints/last.ckpt <hf-path>/last.ckpt \
  --repo-type dataset 
```

`<run_name>` and `<hf-path>` depend on the branch:

| Branch | `<run_name>` | `<hf-path>` | `<wadnb-display-name>` | 
|---|---|---| ---|
| `ca-cond-baseline` | `full_pdb_test_release_ae` | `ae/ca-cond` | `ae-ca-cond-baseline` |
| `fully-latent` | `full_pdb_fully_latent_ae` | `ae/fully_latent` | `ae-fully-latent` |
| `jepa-latent` (50% random mask) | `jepa_random_masking_50_v1` |  | `jepa-rand-mask-50`|
| `jepa-latent` (70% random mask) | `jepa_random_masking_70_v3` | `jepa/random_masking_70pct` | `jepa-rand-mask-70` |
| `jepa-latent` (mutation action) | `mut-jepa_v2` | `jepa/mutation` | `jepa-mutation`|
| `le-jepa` (50% random mask) | `le_jepa_rand50_v1.3` | `le-jepa/random_masking_50pct` | `le-jepa-rand-mask-50` |
| `le-jepa` (70% random mask) | `le_jepa_rand70_v1` |  | `le-jepa-rand-mask-70` |
| `le-jepa` (mutation action) | `le_jepa_mut_v2` | `le-jepa/mutation` | `le-jepa-mutation` |

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
├── jepa-latent
└── le-jepa
```

#### Branch descriptions:

- **main**: La Proteina codebase
- **ca-cond-baseline**: CA-conditioned autoencoder baseline used
- **fully-latent**: removes explicit CA-coordinate conditioning
- **jepa-latent**: I-JEPA-style JEPA (EMA target encoder + stop-gradient), random masking (50% and 70%), with an mutation-conditioning (action) variant
- **le-jepa**: LeJEPA — single encoder, no EMA/stop-gradient, SIGReg regularizer instead. Random masking (50% and 70%), with an mutation-conditioning (action) variant

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

