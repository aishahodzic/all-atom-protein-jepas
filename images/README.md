# Thesis and paper images

This directory contains the curated figures used in the thesis and paper. It
excludes exploratory plots, embedding caches, and generated per-protein bulk
outputs.

## Methods

- `approach.pdf`: project overview from structure input through JEPA-family
  representation learning and evaluation.
- `jepa.pdf`: standard JEPA context/target encoder and predictor architecture.
- `jepa-sigreg.pdf`: JEPA objective augmented with SIGReg.
- `lejepa.pdf`: LeJEPA global/local view objective.
- `cropping.pdf`: global and local crop construction.

These are the original vector diagrams used by the LaTeX manuscripts.

## Latent spaces

- `chain-length-crop-control-transparent.png`: README version of the matched
  uncropped/cropped chain-length comparison.
- `chain-length-crop-control-paper.png`: original manuscript export of that
  comparison.
- `full-12-model.png`: full model-family latent-space comparison.
- `model-decisions.png`: latent-space comparison for the principal design
  decisions.
- `model-family-comparison.png`: C-alpha AE, JEPA, uncropped LeJEPA, and
  rotation-aware LeJEPA comparison.
- `jepa-sigreg-r50-absolute-v-relative-index.png` and
  `jepa-sigreg-r70-absolute-v-relative-index.png`: absolute versus relative
  positional controls.
- `jepa-r70-rotation-60k.png` and `jepa-r70-rotation-comparison.png`: rotation
  intervention comparisons.

## Reconstruction

- `4etr_A-transparent.png` and `5a13_G-transparent.png`: transparent versions
  used in the repository README.
- `4etr_A-paper.png`, `5a13_G.png`, `5l0o_A.png`, `5xr7_A.png`, and
  `8cbv_A.png`: manuscript whole-protein native/reconstruction overlays.
- `failure-mode-paper.png`: paired reconstruction example used in the paper.
- `failure-distorted-bond.png`, `failure-overlapping-sidechains.png`,
  `failure-misplaced-sidechains.png`, and `failure-distorted-connection.png`:
  appendix examples of local reconstruction failures.

Native structures are grey and postprocessed JEPA reconstructions are orange.

## Evaluation

- `lejepa-uncropped-selected-layerwise-stb.png`: selected layerwise
  StructTokenBench results.
- `lejepa-raw100k-layerwise-stb-all-metrics.png`: complete layerwise appendix
  panel.
- `lejepa-raw100k-bottleneck8-comparison.png`: bottleneck-8 comparison.
- `lejepa-full-context-layerwise-probes.png`: full-context layerwise probing.
