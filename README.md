Original NVIDIA La-Proteina repository and setup instructions:
https://github.com/NVIDIA-BioNeMo/la-proteina

Weights & Biases project:
https://wandb.ai/aisha-hodzic-org/protein_ae?nw=nwuseraishahodzic

## Autoencoder Experiments

### 1. Train AE with latent residues and PDB-given C-alpha (current setup)

Training:

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
python proteinfoundation/partial_autoencoder/inference.py --config_name inference_ae
```

Checkpoint:

```text
store/test_release_ae/checkpoints/last-EMA.ckpt
```

Metrics:

* Structure reconstruction (`all_atom_rmsd`)
* Sequence recovery (`seq_rec_rate`)

### 2. Train fully latent AE

Evaluate structure reconstruction and sequence recovery.

Compare both setups. Initial expectation: the fully latent AE will perform worse, since the decoder no longer receives the true C-alpha coordinates.

### 3. Evaluate both latent representations on a downstream task
