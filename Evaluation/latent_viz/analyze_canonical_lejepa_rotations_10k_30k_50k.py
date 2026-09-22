"""Matched 10k/30k/50k(/70k) audit for canonical LeJEPA with view rotations.

This deliberately reuses the canonical-V1 audit implementation and its frozen
protein cohort, physical-residue sample, t-SNE settings, and quantitative
diagnostics.  Only checkpoint identities and output namespaces differ.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import analyze_canonical_lejepa_v1_10k_30k as analysis  # noqa: E402
import run_tsne as run_tsne_module  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)


DEFAULT_STEPS = (10_000, 30_000, 50_000)
STEPS = DEFAULT_STEPS
CHECKPOINTS: dict[str, pathlib.Path] = {}
STAGED: dict[str, pathlib.Path] = {}


def configure_checkpoints(steps: tuple[int, ...]) -> None:
    global STEPS
    STEPS = steps
    CHECKPOINTS.clear()
    CHECKPOINTS.update(
        {
            f"canonical-lejepa-v1-rotations-ema-step{step}": (
                REPO
                / "store/canonical_lejepa_rotations_100k/checkpoints"
                / f"step-{step:08d}.ckpt"
            )
            for step in STEPS
        }
    )
    STAGED.clear()
    STAGED.update(
        {
            checkpoint_id: (
                REPO
                / "store/canonical_lejepa_rotations_100k/evaluation"
                / f"canonical-lejepa-v1-rotations-step{step}-ema-staged.ckpt"
            )
            for checkpoint_id, step in zip(CHECKPOINTS, STEPS)
        }
    )


def configure_shared_analysis() -> None:
    analysis.CHECKPOINTS.clear()
    analysis.CHECKPOINTS.update(CHECKPOINTS)
    analysis.CHECKPOINT_STEPS.clear()
    analysis.CHECKPOINT_STEPS.update(
        dict(zip(CHECKPOINTS, STEPS))
    )
    analysis.STAGED.clear()
    analysis.STAGED.update(STAGED)

    labels = [f"{step // 1000}k" for step in STEPS]
    stem = "canonical_lejepa_v1_rotations_ema_" + "_".join(labels)
    comparison = "-v-".join(labels)
    analysis.RESULT = HERE / f"results/{stem}_position.csv"
    analysis.MANIFEST = HERE / f"results/{stem}_position.json"
    analysis.POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        f"tsne_canonical_lejepa_v1_rotations_ema_{comparison}_position-and-length.png"
    )
    analysis.RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        f"tsne_canonical_lejepa_v1_rotations_ema_{comparison}_aa-and-ss.png"
    )
    analysis.ANALYSIS_LABEL = (
        "Canonical LeJEPA with independent SO(3) view rotations"
    )
    analysis.OUTPUT_TAG = f"canonical-lejepa-v1-rotations-ema-{comparison}"
    analysis.MANIFEST_DESCRIPTION = (
        "Matched canonical LeJEPA independent-view-rotation EMA "
        f"{'/'.join(labels)} position and length audit"
    )
    run_tsne_module.SHORT_LABEL.update(
        {
            checkpoint_id: f"Canonical LeJEPA + rotations EMA ({step // 1000}k)"
            for checkpoint_id, step in zip(CHECKPOINTS, STEPS)
        }
    )


def stage_and_validate_ema() -> None:
    for checkpoint_id, step in zip(CHECKPOINTS, STEPS):
        checkpoint_path = CHECKPOINTS[checkpoint_id]
        staged_path = STAGED[checkpoint_id]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if not staged_path.is_file():
            stage_structtokenbench_checkpoint(
                checkpoint_path, staged_path, source="ema"
            )
        analysis.validate_staged_ema(checkpoint_path, staged_path, step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-70k",
        action="store_true",
        help="extend the existing matched audit through the immutable 70k EMA checkpoint",
    )
    args = parser.parse_args()
    configure_checkpoints(
        DEFAULT_STEPS + ((70_000,) if args.include_70k else ())
    )
    configure_shared_analysis()
    stage_and_validate_ema()
    sys.argv = [sys.argv[0]]
    analysis.main()


if __name__ == "__main__":
    main()
