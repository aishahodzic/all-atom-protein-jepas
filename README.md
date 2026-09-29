# MSc thesis: protein representation learning with La-Proteina

This repository contains the training code, evaluation pipelines, archived
results, and manuscript assets for the MSc thesis and associated paper. It
extends [NVIDIA BioNeMo's La-Proteina](https://github.com/NVIDIA-BioNeMo/la-proteina)
with autoencoder controls, JEPA and JEPA+SIGReg experiments, canonical LeJEPA,
StructTokenBench integration, representation diagnostics, and reconstruction
evaluation.

The default branch is the handoff and evaluation branch. Historical training
families remain on their original branches so that their exact configurations
and code revisions are preserved.

## Start here

The complete train-to-results workflow is in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). It covers:

- environment and data setup;
- which branch trains each model family;
- training and resume commands;
- downloading the evaluated checkpoints from Hugging Face;
- running the locked StructTokenBench protocol;
- linear probes, k-NN purity, fold retrieval, t-SNE, and reconstruction;
- where checkpoints, logs, numerical results, and manuscript figures live.

For an existing environment, the shortest evaluation setup is:

```bash
git clone https://github.com/aishahodzic/msc-thesis.git
cd msc-thesis
hf auth login
python script_utils/stage_thesis_checkpoints.py --group core

cd StructTokenBench
bash src/script/run_eval.sh \
  --protocol=upstream-l600-v1 \
  --checkpoint=100k \
  --experiment=jepa-rand-mask-70 \
  --tasks=all
```

The checkpoint staging command downloads each file once into the Hugging Face
cache, verifies it against the archive manifest, and creates the compatibility
paths used by the evaluation code.

## Reproducibility archives

- Private model archive:
  [`msc-thesis-org/ckpts`](https://huggingface.co/datasets/msc-thesis-org/ckpts)
- Checkpoint inventory and SHA-256 provenance:
  [`manifest.csv`](https://huggingface.co/datasets/msc-thesis-org/ckpts/blob/main/manifest.csv)
- Selected reconstruction examples:
  [`pymol_reconstruction/`](https://huggingface.co/datasets/msc-thesis-org/ckpts/tree/main/pymol_reconstruction)
- Processed training data:
  [`msc-thesis-org/pdb-processed`](https://huggingface.co/datasets/msc-thesis-org/pdb-processed)

Access to private Hugging Face datasets is inherited from the logged-in user;
no access token is stored in this repository.

## Repository map

| Path | Purpose |
|---|---|
| `proteinfoundation/` | Models, training entry points, datasets, and checkpoint utilities |
| `configs/` | Hydra training, model, dataset, inference, and evaluation configuration |
| `jobs/` | Exact Slurm launchers used for training, evaluation, and analysis |
| `StructTokenBench/` | Integrated downstream benchmark and frozen-encoder wrapper |
| `Evaluation/` | Linear probes, k-NN purity, retrieval, latent visualisation, diagnostics, and reconstruction |
| `Evaluation/**/results/` | Compact numerical results and run manifests retained in Git |
| `thesis-latex/` | Thesis source and final thesis figures |
| `iclr2027/` | Paper source and final paper figures |
| `datasetcard/`, `modelcard/` | Dataset and model documentation |

Large checkpoints, embedding caches, decoder intermediates, and bulk generated
structures are intentionally excluded from Git. The retained checkpoints and
ten qualitative reconstruction examples are in the private Hugging Face
archive above.

## Model families and branches

| Family | Source branch | Main entry point |
|---|---|---|
| C-alpha-conditioned AE | `ca-cond-baseline` | `proteinfoundation/partial_autoencoder/train.py` |
| Fully latent AE | `fully-latent` | `proteinfoundation/partial_autoencoder/train.py` |
| JEPA and JEPA interventions | `jepa-latent` | `proteinfoundation/jepa/train.py` |
| Historical JEPA+SIGReg runs | `le-jepa` | `proteinfoundation/jepa/train.py` |
| Canonical LeJEPA | `canonical-lejepa` | `proteinfoundation/jepa/train_canonical_lejepa.py` |
| Evaluation and archived results | `main` | `StructTokenBench/` and `Evaluation/` |

Use Git worktrees when reproducing training so that `main` remains available
for evaluation and result inspection. Exact commands are provided in the
reproducibility guide.

## Citation and upstream project

This work builds on the original
[La-Proteina repository](https://github.com/NVIDIA-BioNeMo/la-proteina) and the
[StructTokenBench project](https://github.com/KatarinaYuan/StructTokenBench).
See the thesis and paper sources for the complete bibliography and experimental
methodology.
