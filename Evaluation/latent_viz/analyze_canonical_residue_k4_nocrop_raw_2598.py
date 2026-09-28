"""Frozen-cohort raw-step2598 audit for no-crop residue K=4 LeJEPA."""

from __future__ import annotations

import pathlib
import sys

import torch


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import analyze_canonical_lejepa_v1_10k_30k as analysis  # noqa: E402
import checkpoints  # noqa: E402
import run_tsne as run_tsne_module  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)


CHECKPOINT_ID = "canonical-lejepa-v1-residue-k4-nocrop-raw-step2598"
RUN_NAME = "canonical_lejepa_v1_residue_k4_nocrop_visible_only_minvalid4_10k"
STEP = 2_598
RUN_ROOT = REPO / "store" / RUN_NAME
SOURCE = RUN_ROOT / "evaluation/snapshots/raw-step00002598-full-state.ckpt"
CONFIG = RUN_ROOT / "checkpoints" / f"exp_config_{RUN_NAME}.json"
STAGED = RUN_ROOT / "evaluation/canonical-residue-k4-nocrop-raw-step2598-staged.ckpt"


def configure() -> None:
    analysis.CHECKPOINTS.clear()
    analysis.CHECKPOINT_STEPS.clear()
    analysis.STAGED.clear()
    analysis.ROW_LABELS.clear()
    analysis.CHECKPOINTS[CHECKPOINT_ID] = SOURCE
    analysis.CHECKPOINT_STEPS[CHECKPOINT_ID] = STEP
    analysis.STAGED[CHECKPOINT_ID] = STAGED
    analysis.ROW_LABELS[CHECKPOINT_ID] = "Residue K=4, no crop, raw step 2,598"
    analysis.RESULT = (
        HERE / "results/canonical_residue_k4_nocrop_raw_2598_position.csv"
    )
    analysis.MANIFEST = (
        HERE / "results/canonical_residue_k4_nocrop_raw_2598_position.json"
    )
    analysis.POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_residue_k4_nocrop_raw_2598_position-and-length.png"
    )
    analysis.RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_residue_k4_nocrop_raw_2598_aa-and-ss.png"
    )
    analysis.ANALYSIS_LABEL = "Canonical residue K=4 LeJEPA, no crop"
    analysis.OUTPUT_TAG = "canonical-residue-k4-nocrop-raw-2598"
    analysis.MANIFEST_DESCRIPTION = (
        "Frozen-cohort interim raw-step2598 no-crop residue K=4 audit"
    )
    analysis.ENCODER_SOURCE = "raw"
    analysis.ENCODER_LABEL = "Raw"
    run_tsne_module.SHORT_LABEL.update(analysis.ROW_LABELS)


def register_raw() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    if not STAGED.is_file():
        stage_structtokenbench_checkpoint(SOURCE, STAGED, source="raw")

    staged = torch.load(STAGED, map_location="cpu", weights_only=False, mmap=True)
    provenance = staged.get("canonical_lejepa_evaluation", {})
    expected = {
        "encoder_source": "raw",
        "source_global_step": STEP,
        "latent_dim": 8,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid staged raw checkpoint: {mismatches}")

    checkpoints.JEPA_ENCODER_CFG = CONFIG
    entry = (str(STAGED), "jepa")
    checkpoints.CHECKPOINTS[CHECKPOINT_ID] = entry
    checkpoints.ALL_CHECKPOINTS[CHECKPOINT_ID] = entry


def main() -> None:
    configure()
    analysis.register = register_raw
    sys.argv = [sys.argv[0]]
    analysis.main()


if __name__ == "__main__":
    main()
