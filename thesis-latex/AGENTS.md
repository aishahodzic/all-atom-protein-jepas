# Thesis-writing instructions

These instructions apply within `thesis-latex/`. Follow the repository-root
`AGENTS.md` as well. Start by reading `THESIS_HANDOFF.md`; use the inventories in
`handoff/` only as needed for the current writing task.

## Preserve the author's writing

- Treat existing thesis prose as author-owned, polished text. Do not rewrite,
  shorten, delete, or silently correct it unless the user explicitly requests
  that wording change. This is especially important for Background, Related
  Work, and every already-written Methodology sentence.
- When asked to reorganize, move complete sentences/paragraphs verbatim and add
  headings around them. If a factual correction is needed, explain it and offer
  replacement wording separately before editing protected prose.
- Preserve existing annotations. Blue in `skeleton.tex` marks missing chapter
  coverage, not completed work. Yellow in Methodology marks proposed removal;
  orange marks a proposed move. A mark is not permission to act on it. Preserve
  bold `{unpolished}` markers and any remaining green/orange placement marks.
- `Project.tex` is the master document; it includes `text/sec3-relevant-works.tex`,
  not the historical `text/drafts.tex`. Check the current include graph.
- Keep `skeleton.tex` concise throughout: chapter -> named paragraph -> keywords.
  Detailed Results plans live separately in `results-skeleton.tex`: question,
  finding, evidence, coloured 1--10 strength rating, experimental control and
  support/replication. Weak = red (1--3), Medium = amber (4--7), Strong = green
  (8--10); scores are editorial judgements, not statistical confidence. Sources
  are in `handoff/experiments/RESULTS_EVIDENCE.md`, and asset readiness is in
  `RESULTS_READINESS.md`. Do not turn proposed experiments into Results.
- The author clarified on 14 September that **only the Results skeleton's
  introduction should be shortened**, not its detailed result entries. The
  previous detailed body has been restored; do not reapply the two-page rewrite.
- Add/maintain Numbers/Figures readiness flags beside result titles without
  changing the detailed prose. Green = usable; red = missing, incomplete or
  needs edits. Omit inapplicable assets; distinguish ready probes from the
  incomplete layerwise STB panel. Existing figures can still need caption caveats.
- In the detailed Results outline, put Control on its own line. "To improve"
  states evidence that would strengthen the claim, not a funded task list.
  Preserve "How to report" instructions and checkpoint distinctions. Decoder
  recovery stays a 1--2-sentence mention, with existing details in the appendix.

## Scope, terminology, and length

- Cover all experiment variants that were run unless explicitly excluded by the
  author. The preliminary cropped LeJEPA runs using minimum chain length 64 are
  excluded from thesis results; retain their historical records. This does not
  exclude 64-dimensional latent representations or minimum-length-96 cropped
  runs. See `THESIS_HANDOFF.md` for the excluded run identifiers.
  Distinguish substantive experiments, partial runs, failed attempts,
  controls, and smoke tests. Their placement in main tables, supporting sections,
  or the appendix is a separate decision; do not silently drop variants.
- Use **JEPA**, **JEPA+SIGReg**, and **LeJEPA**. Historical `le-jepa` branch/result
  names mean JEPA+SIGReg, not canonical LeJEPA. Preserve original identifiers in
  provenance and verify individual variant configurations.
- Introduction: 1--2 pages. Background: 3--4 pages. Related Work: 2--3 pages
  maximum. The author's planning estimate is approximately 14 sentences/page;
  check the rendered PDF rather than treating sentence counts as exact layout.
- Keep experimental setup separate from evaluation protocols. Background
  explains necessary concepts; Related Work distinguishes prior approaches;
  Methodology states what this project actually implemented.
- Results budget policy (14 September): final encoder comparisons use 100k,
  except cropped LeJEPA at 10k (batch 96); these are not compute-matched.
  Label earlier diagnostics separately. Do not prescribe blanket seed reruns,
  decoder extensions, or spatial-neighbour work; targeted repeats are optional.
- The author confirms the minimum-length-50 probe splits. Restrict outstanding
  held-out probe verification/reruns to minimum-length-96 runs if included.
  This does not resolve historical training-control differences or STB protocols.
- Preserve the current Related Work order unless asked otherwise: foundation
  models; structural SSL; SLAE; ProtJEPA; ProteinJEPA; research gap. Do not restore
  the older outline's JEPA-first order or its CHEAP paragraph automatically.

## Evidence and scientific accuracy

- Do not infer an outcome from an outline heading, a run name, or a plot alone.
  Cite the actual result, implementation, saved configuration, and checkpoint.
  Keep recorded training steps distinct from W&B history-event indices and
  downstream probe-training steps. Distinguish raw/online from evaluation-EMA.
- Known issues are recorded in `THESIS_HANDOFF.md`: reconstruction is unaligned
  and C-alpha anchored; retrieval is Euclidean; Homo uses macro-F1 for
  benchmark-aligned reporting, but the historical panel contains AUROC and the
  archived protocol uses batch-mean aggregation. Never just relabel metrics.
- Frozen latent-head, hidden-layer, and intervention diagnostics are not
  interchangeable with downstream benchmark scores. Computational interventions
  do not by themselves establish biological causality.
- Match task cohorts, preprocessing, checkpoint policy, and metric aggregation
  before making comparisons. Flag differences instead of calling them controlled
  ablations. JEPA versus the default JEPA+SIGReg changes more than regularisation.
- Verify claims against primary papers; preserve citation keys. Historical
  research notes are a reading aid, not proof that every claim was verified.
  There is no supervisor feedback yet; never invent it.

## Working boundaries

- Work in small, reviewable steps. Do not run training/evaluations, change model
  code, commit, push, or upload the thesis/notes unless specifically requested.
- Do not copy `.env`, API keys, private session logs, or whole conversations into
  the handoff. Existing paper access does not require copying credentials.
- Put temporary files and caches in a clearly named directory under `$HOME/tmp`,
  never system `/tmp`. Keep durable writing outputs in this thesis directory.
- When compiling, keep build artifacts separate from author-owned sources and
  do not overwrite the current PDF without approval for a rebuild.
