# Analysis source archive

This archive accompanies the experiment results saved through 11 September
2026. It makes the analysis implementations readable on `main` without merging
the canonical training branch or changing the archived numerical results.

`analysis_source_manifest.json` records the path and SHA-256 of each of the 65
verbatim source files: 34 Python files, 30 historical Slurm analysis launchers,
and the canonical learning-rate diagnostics notes. Many of these scripts were
previously untracked; the source checkout's commit alone therefore does not
identify their contents. The manifest captures the files as archived, not proof
that every historical result was produced by that exact file version.

## Reading order

| Analysis | Entry points |
| --- | --- |
| Layerwise probes | `linear_probing/run_canonical_layerwise_probes.py`; `canonical_lejepa/extract_full_context_layer_states.py` |
| Head decomposition | `linear_probing/run_canonical_head_decomposition_probes.py` |
| Neighbour interventions | `canonical_lejepa/run_neighbor_influence.py`; `canonical_lejepa/run_neighbor_group_influence.py`; the `check_neighbor_*_eligibility.py` helpers |
| Checkpoint loading | `canonical_lejepa/checkpoint.py`; `canonical_lejepa/stage_for_structtokenbench.py` |
| Representation and learning-rate diagnostics | `diagnostics/CANONICAL_LR_DIAGNOSTICS.md`; `diagnostics/analyze_canonical_residue_ab_10k.py`; `canonical_lejepa/plot_stb_lr_trajectories.py` |
| Position/rotation visualizations | The archived `latent_viz/analyze_canonical_*.py`, colouring and composition scripts |

Paths in this table are relative to `Evaluation/`. Matching analysis job files
are in `jobs/`; the checksum manifest is the exact archive inventory.

## Model names and interpretation

- `jepa-latent` corresponds to JEPA.
- The historical `le-jepa` branch and result names correspond to JEPA+SIGReg.
- `canonical-lejepa` corresponds to LeJEPA.

Keep original run identifiers and distinguish raw/online encoders from
evaluation-EMA encoders. Check actual checkpoint steps and cohort/config hashes;
a run name ending in `30k` does not establish that it reached that step.

The combined learning-rate plotting script currently selects Homo AUROC. This
is preserved historical behaviour, not a correction to the thesis metric.
StructTokenBench reports macro-F1 for Homo. Existing archived Homo manifests
also contain F1 values, but record `legacy_batch_mean` aggregation: a mean of
batchwise metrics rather than a single whole-split calculation. Do not relabel
AUROC values as F1 or mix the two aggregation protocols.

Other evaluation descriptions must follow their implementations: reconstruction
uses unaligned RMSD with supplied reference C-alpha coordinates (including fixed
output C-alpha atoms), and fold retrieval uses Euclidean distance. Their code is
in `structure_rec/train_decoder.py` and `fold_retrieval/retrieval.py`.

## Runtime requirements and limitations

This is a source archive, not a self-contained rerun bundle. Full analyses need
the project environment, the matching frozen checkpoints and encoder configs,
protein data, and saved labels/cohort or hidden-state artifacts. In particular,
the layer/head analyses reference
`store/canonical_lejepa_v1_100k/evaluation/full_context_layer_states/full/` and
`Evaluation/linear_probing/latents/labels.pt`. These artifacts are not added by
this archive.

`diagnostics/diagnose_canonical_loss_balance.py` additionally imports
`proteinfoundation.jepa.canonical_lejepa_model`, which remains on the
`canonical-lejepa` training branch. Use the matching branch implementation and
recorded experiment configuration for that diagnostic; importing it from
`main` alone is insufficient. The source checkout base is recorded in the
manifest. Its unrelated unstaged training edits were not copied.

The Slurm launchers are preserved as historical records. They contain original
Noether checkout/environment paths; some also set `MPLCONFIGDIR` under `/tmp`.
Before running them, adapt the checkout paths and redirect all temporary/cache
locations to a task-specific directory under `$HOME/tmp`. Export `DATA_PATH`
for data-dependent imports. Never publish `.env`, credentials, or agent-session
logs with the archive.

## Validation of this archive

- All 65 copied files matched the source SHA-256 checksums.
- All 34 Python files passed syntax parsing, and all 30 job files passed `bash -n`.
- Nine CPU unit tests passed across `test_checkpoint`,
  `test_full_context_layer_states`, and `test_neighbor_influence` in the
  project's Python environment. Their temporary files were directed under
  `$HOME/tmp`; an unused `DATA_PATH` satisfied an import-time requirement.
- No training, dataset extraction, full probing run, or GPU experiment was
  rerun. These checks do not validate every archived scientific conclusion.
