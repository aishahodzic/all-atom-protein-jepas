# Canonical LeJEPA v1: layerwise STB panel

Scheduled 14 September 2026 at the author's request. These jobs are evaluations
of frozen encoder readouts, not new encoder training or completed findings.

## Selected comparison

- Run: `canonical_lejepa_v1_100k`, step 100000, **EMA weights**.
- Readouts: post-block **6 (index 5)** first, then **1 (index 0)**, then
  **12 (index 11)**. All are 768D; no extra projection is applied.
- Tasks: **BindShake, Ept, FlexRMSF, FlexBFactor, FlexNEQ, Rep**.
  This is four topic groups but six separately trained task heads per readout:
  18 head evaluations across the three blocks, not four heads per block.
- Use `upstream-l600-v1`, full unmasked encoder context, the existing task splits,
  probe seed 1234, batch 8, LR 0.001, hidden width 512 and existing stopping rules.
- Existing final-8D results are the reference; no 8D rerun is scheduled.
  Before combining tables, match cohort hashes, task settings and weight provenance.
  Input dimension/head parameter count differs between 768D and 8D, so the
  contrast does not isolate latent dimensionality alone.

This selection supersedes the earlier proposed CatBio/FlexRMSF-only follow-up.
It does not request CatBio, BindInt, extra encoder seeds or spatial interventions.

## Jobs and scheduling

Current policy: independent jobs, with **no completion dependencies**. Block 6
was submitted first; this does not require the other blocks to wait for it.

| Readout | Slurm job | Dependency | State at dependency-removal check |
|---|---:|---|---|
| Block 6 / index 5 | 258849 | none | running on noether |
| Block 1 / index 0 | 258850 | none | running on noether |
| Block 12 / index 11 | 258851 | none | pending, resources |

Historically, at the first status check, block 6 was **running on noether**, had passed its
GPU integration fixture, and was extracting BindShake embeddings. Blocks 1 and
12 remained dependency-pending. No task score was complete at that check. On
14 September, the author clarified that these jobs must not depend on one
another. Dependencies were removed in place from 258850/258851 and verified as
`Dependency=(null)`; running job 258849 was not changed by this correction.

All three jobs now use **short QoS**, with one GPU on noether, 16 CPUs,
64 GB RAM and a **three-hour limit per job** (the short-QoS maximum).
The jobs may overlap when resources permit, with a maximum aggregate allocation
of nine GPU-hours; that is a limit, not a runtime forecast. Do not reintroduce
completion dependencies. No existing training job was changed.

QoS correction on 14 September: pending jobs 258850/258851 were updated in place.
Slurm refused an in-place QoS change for running job 258849, so it was requeued,
held, changed to short and released with the **same job ID and dependencies**.
No task manifest had completed before that requeue. Existing embedding caches
were retained, and the original log was preserved as
`logs/stb/publication_upstream/stb_canon_b6_100k-258849.before-short-requeue.out`.
The launcher now defaults to short. Only its reviewed QoS-header change was
reflected in the source guard; encoder/evaluator hashes were not regenerated.

Launcher: `jobs/eval_canonical_lejepa_v1_100k_layerwise_panel.job`.
Logs: `logs/stb/publication_upstream/stb_canon_b{6,1,12}_100k-<job>.out`.
Results/manifests:
`StructTokenBench/struct_token_bench_logs/upstream-l600-v1-full/runs/slurm-<job>-canonical-lejepa-v1-ema-step100k-block<block>-panel/`.
Experiment labels end in `-block<block>-upstream-l600-v1`.

## Provenance and validation

- Original checkpoint SHA-256:
  `4f09dff610adaa75c8ac51b4cf01acb001c236c72dd91ac31568ffa191484527`.
- Staged EMA checkpoint:
  `store/canonical_lejepa_v1_100k/evaluation/canonical-lejepa-v1-step100000-ema-staged.ckpt`;
  SHA-256 `49aaae3dccc3835005bb6470b0f483a57cee4e28bead2408994693974389e273`.
- Shared encoder config SHA-256:
  `98e0350d418e7d840fa245ddfd7150064e87a9ac74c72c9150a946f0f66ccd9d`.
- The legacy loader calls the staged prefix `online_encoder`; these are still
  the verified **EMA** weights, not the original online training weights.
- Added an opt-in `TOKENIZER_READOUT_BLOCK` / `tokenizer_readout_block` adapter.
  Both cross-task and per-task cache identities distinguish blocks explicitly;
  equal-width hidden layers cannot share embeddings accidentally. Manifests
  record the block number, zero-based module index, width and context.
- All 48 CPU evaluator unit tests passed. A real cached chain (`1kn0_G`, 184
  residues) verified block 6 against a direct forward hook, preserved the final
  mean output, and passed alignment/finite-output checks on CPU. This is an
  integration fixture, not an STB result. Each scheduled job repeats its
  block-specific fixture on its allocated GPU before running the task panel.
- `jobs/canonical_layerwise_source_guard.sha256` pins 211 source/config files.
  A queued job fails before evaluation if those files change; do not silently
  regenerate the guard to accept unreviewed code changes.
- Temporary files live under `$HOME/tmp/stb-canonical-layerwise/<job>/`.
  Reusable block-specific embeddings live under
  `store/canonical_lejepa_v1_100k/evaluation/stb_layerwise_shared_cache/`.

No scores have been added here or to thesis tables by this scheduling action.
