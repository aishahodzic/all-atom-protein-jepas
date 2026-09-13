"""Fixed-sample latent audit for the seed-43 rotation replicate at step 10k.

This reuses the canonical-LeJEPA audit implementation verbatim so the protein
cohort, 15k physical-residue sample, t-SNE settings, and all quantitative
diagnostics are identical to the seed-42 rotation step-10k analysis.
"""

from __future__ import annotations

import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import analyze_canonical_lejepa_v1_10k_30k as analysis  # noqa: E402
import run_tsne as run_tsne_module  # noqa: E402


CHECKPOINT_ID = "canonical-lejepa-v1-rotations-seed43-ema-step10000"
STEP = 10_000
CHECKPOINT = (
    REPO
    / "store/canonical_lejepa_rotations_seed43_10k/checkpoints"
    / "step-00010000.ckpt"
)
STAGED = (
    REPO
    / "store/canonical_lejepa_rotations_seed43_10k/evaluation"
    / "canonical-lejepa-v1-rotations-seed43-step10000-ema-staged.ckpt"
)


def configure_shared_analysis() -> None:
    analysis.CHECKPOINTS.clear()
    analysis.CHECKPOINTS[CHECKPOINT_ID] = CHECKPOINT
    analysis.CHECKPOINT_STEPS.clear()
    analysis.CHECKPOINT_STEPS[CHECKPOINT_ID] = STEP
    analysis.STAGED.clear()
    analysis.STAGED[CHECKPOINT_ID] = STAGED
    analysis.RESULT = (
        HERE
        / "results/canonical_lejepa_v1_rotations_seed43_ema_10k_position.csv"
    )
    analysis.MANIFEST = (
        HERE
        / "results/canonical_lejepa_v1_rotations_seed43_ema_10k_position.json"
    )
    analysis.POSITION_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_v1_rotations_seed43_ema_10k_position-and-length.png"
    )
    analysis.RESIDUE_FIGURE = HERE / (
        "figures/canonical_lejepa/"
        "tsne_canonical_lejepa_v1_rotations_seed43_ema_10k_aa-and-ss.png"
    )
    analysis.ANALYSIS_LABEL = (
        "Canonical LeJEPA with independent SO(3) view rotations, seed 43"
    )
    analysis.OUTPUT_TAG = "canonical-lejepa-v1-rotations-seed43-ema-10k"
    analysis.MANIFEST_DESCRIPTION = (
        "Fixed-sample canonical LeJEPA independent-view-rotation seed-43 "
        "EMA step-10k position and length audit"
    )
    run_tsne_module.SHORT_LABEL[CHECKPOINT_ID] = (
        "Canonical LeJEPA + rotations seed 43 EMA (10k)"
    )


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    analysis.validate_staged_ema(CHECKPOINT, STAGED, STEP)
    configure_shared_analysis()
    # The reused entry point has optional multi-checkpoint extension flags.
    sys.argv = [sys.argv[0]]
    analysis.main()


if __name__ == "__main__":
    main()
