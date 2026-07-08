Original NVIDIA La-Proteina repository and setup instructions:
https://github.com/NVIDIA-BioNeMo/la-proteina

Weights & Biases project:
https://wandb.ai/aisha-hodzic-org/protein_ae?nw=nwuseraishahodzic


## Experiment Branches

```text
main
├── ca-cond-baseline
├── fully-latent
└── {insert-future-ae}
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

Config changes:

* DiT-B → DiT-S (`token_dim: 768→384`, `nheads: 12→6`)
* Removed `x1_bb_angles`, `x1_sidechain_angles`
* Kept `latent_z_dim = 8`
* `dist_strategy: ddp → auto`
* Enabled W&B logging

Evaluation:

```bash
sbatch jobs/inference_ae.job
```

```bash
python proteinfoundation/partial_autoencoder/inference.py --config_name inference_ae
```

Checkpoint:

```text
store/test_release_ae/checkpoints/last-EMA.ckpt
```

### 2. Train fully latent AE

Evaluate structure reconstruction and sequence recovery.

Compare both setups. Initial expectation: the fully latent AE will perform worse, since the decoder no longer receives the true C-alpha coordinates.

## 1 & 2 Results


Metrics:

* Structure reconstruction (`all_atom_rmsd`)
* Sequence recovery (`seq_rec_rate`)


Results: 

Evaluated on 4 protein structures. Total data size: 664. 

| Model | 60k RMSD | 60k Seq Rec. | 100k RMSD | 100k Seq Rec. |
|---------|---------:|---------:|---------:|---------:|
| CA-conditioned AE | 0.274 Å | 1.000 | 0.241 Å | 1.000 |
| Fully Latent AE | 0.420 Å | 1.000 | 0.371 Å | 1.000 |


### 3. Evaluate both latent representations on a downstream task
