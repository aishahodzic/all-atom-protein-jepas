# Canonical LeJEPA LR diagnostics on Jarvis

This audit compares raw encoder representations at an exact checkpoint. It is
not StructTokenBench: AA/SS purity and RSA correlation are k-nearest-neighbour
proxies. The script fixes the held-out cohort, 15,000-residue sample, cosine
neighbour definition, cross-protein exclusion, seed, and metric formulas.

From the repository root, run each experiment after its 10k checkpoint exists:

```bash
PYTHON_BIN=python jobs/run_canonical_lr_diagnostics_jarvis.sh \
  canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e3_cosine_30k \
  10000

PYTHON_BIN=python jobs/run_canonical_lr_diagnostics_jarvis.sh \
  canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_lr1e4_cosine_30k \
  10000
```

`DATA_PATH` must name the directory containing `pdb_train/`; `.env` is loaded
when present. One CUDA GPU must be visible. The launcher always selects the raw
encoder and validates that the checkpoint is the protein-level, per-view
SIGReg, fixed-96/48 crop, packed-view, independent-local, global-batch-96
recipe. It intentionally permits LR schedule differences.

The exact min-length-50 evaluation cohort requires four files in
`$DATA_PATH/pdb_train`: the selector CSV, its `seq_*.fasta`, and the matching
`cluster_seqid_0.5_*_test.{fasta,tsv}`. The launcher verifies their frozen
SHA256 hashes before loading a checkpoint, so a regenerated or mismatched split
fails instead of silently changing the 5,224-protein cohort.

It also requires the frozen label cache at
`Evaluation/linear_probing/latents/labels.pt`. Its hash is checked as well;
this avoids recomputing SS/RSA labels with potentially different dependency
versions and makes the first Jarvis audit substantially faster.

Results are written under `Evaluation/diagnostics/results/` as one CSV and one
JSON per run. Compare them with the committed historical raw-10k CSV/JSON for
`canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_independent_locals_30k`.

Prefer higher AA/SS purity and RSA correlation, provided residue effective
rank remains healthy and chain-length/raw-index/normalized-position correlations
do not increase. These diagnostics screen representation health and shortcuts;
use official StructTokenBench tasks for the downstream-performance decision.
