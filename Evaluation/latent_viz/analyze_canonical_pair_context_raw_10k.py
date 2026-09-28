"""Frozen-cohort raw-10k t-SNE audit for pair-only + pair-context LeJEPA."""

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


CHECKPOINT_ID = "canonical-lejepa-pair-context-raw-step10000"
RUN_NAME = "canonical_lejepa_v1_rotations_pair_context_100k"
STEP = 10_000
RUN_ROOT = REPO / "store" / RUN_NAME
SOURCE = RUN_ROOT / "checkpoints" / f"step-{STEP:08d}.ckpt"
CONFIG = RUN_ROOT / "checkpoints" / f"exp_config_{RUN_NAME}.json"
STAGED = RUN_ROOT / "evaluation" / "pair-context-step10000-raw-staged.ckpt"


def configure() -> None:
    analysis.CHECKPOINTS.clear()
    analysis.CHECKPOINT_STEPS.clear()
    analysis.STAGED.clear()
    analysis.ROW_LABELS.clear()
    analysis.CHECKPOINTS[CHECKPOINT_ID] = SOURCE
    analysis.CHECKPOINT_STEPS[CHECKPOINT_ID] = STEP
    analysis.STAGED[CHECKPOINT_ID] = STAGED
    analysis.ROW_LABELS[CHECKPOINT_ID] = "Pair-only + pair context, raw 10k"
    analysis.RESULT = (
        HERE / "results/canonical_lejepa_pair_context_raw_10k_position.csv"
    )
    analysis.MANIFEST = (
        HERE / "results/canonical_lejepa_pair_context_raw_10k_position.json"
    )
    analysis.POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_pair_context_raw_10k_position-and-length.png"
    )
    analysis.RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_pair_context_raw_10k_aa-and-ss.png"
    )
    analysis.ANALYSIS_LABEL = "Canonical LeJEPA pair-only + pair context"
    analysis.OUTPUT_TAG = "canonical-lejepa-pair-context-raw-10k"
    analysis.MANIFEST_DESCRIPTION = (
        "Frozen-cohort raw-10k pair-only-position plus pair-context audit"
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

    staged = torch.load(
        STAGED, map_location="cpu", weights_only=False, mmap=True
    )
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

    # The pair-context bridge adds parameters, so this evaluation must use the
    # run's own frozen architecture config rather than the historical shared
    # JEPA config. Strict loading in load_frozen_encoder validates every tensor.
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
