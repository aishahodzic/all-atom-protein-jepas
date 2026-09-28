"""Matched no-crop canonical LeJEPA raw 10k/100k x rotation audit."""

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


SPECS = (
    ("canonical-lejepa-v1-raw-step10000", "canonical_lejepa_v1_100k", 10_000, False),
    ("canonical-lejepa-v1-rotations-raw-step10000", "canonical_lejepa_rotations_100k", 10_000, True),
    ("canonical-lejepa-v1-raw-step100000", "canonical_lejepa_v1_100k", 100_000, False),
    ("canonical-lejepa-v1-rotations-raw-step100000", "canonical_lejepa_rotations_100k", 100_000, True),
)


def configure() -> None:
    analysis.CHECKPOINTS.clear()
    analysis.CHECKPOINT_STEPS.clear()
    analysis.STAGED.clear()
    analysis.ROW_LABELS.clear()
    for checkpoint_id, run_name, step, rotated in SPECS:
        source = REPO / "store" / run_name / "checkpoints" / f"step-{step:08d}.ckpt"
        infix = "canonical-lejepa-v1-rotations" if rotated else "canonical-lejepa-v1"
        staged = REPO / "store" / run_name / "evaluation" / f"{infix}-step{step}-raw-staged.ckpt"
        analysis.CHECKPOINTS[checkpoint_id] = source
        analysis.CHECKPOINT_STEPS[checkpoint_id] = step
        analysis.STAGED[checkpoint_id] = staged
        rotation_label = "+ rotation" if rotated else "no rotation"
        analysis.ROW_LABELS[checkpoint_id] = f"Raw {step // 1000}k, {rotation_label}"

    analysis.RESULT = HERE / "results/canonical_lejepa_nocrop_raw_rotation_10k_100k_position.csv"
    analysis.MANIFEST = HERE / "results/canonical_lejepa_nocrop_raw_rotation_10k_100k_position.json"
    analysis.POSITION_FIGURE = HERE / "figures/canonical_lejepa/tsne_canonical_lejepa_nocrop_raw_rotation_10k-100k_position-and-length.png"
    analysis.RESIDUE_FIGURE = HERE / "figures/canonical_lejepa/tsne_canonical_lejepa_nocrop_raw_rotation_10k-100k_aa-and-ss.png"
    analysis.ANALYSIS_LABEL = "Canonical LeJEPA, no crop, raw encoder: rotation ablation"
    analysis.OUTPUT_TAG = "canonical-lejepa-nocrop-raw-rotation-10k-v-100k"
    analysis.MANIFEST_DESCRIPTION = "Matched no-crop canonical LeJEPA raw 10k/100k with/without independent view rotations"
    analysis.ENCODER_SOURCE = "raw"
    analysis.ENCODER_LABEL = "Raw"
    run_tsne_module.SHORT_LABEL.update(analysis.ROW_LABELS)


def register_raw() -> None:
    for checkpoint_id, _, step, _ in SPECS:
        staged_path = analysis.STAGED[checkpoint_id]
        if not staged_path.is_file():
            raise FileNotFoundError(staged_path)
        staged = torch.load(staged_path, map_location="cpu", weights_only=False, mmap=True)
        provenance = staged.get("canonical_lejepa_evaluation", {})
        expected = {"encoder_source": "raw", "source_global_step": step, "latent_dim": 8}
        mismatches = {
            key: (provenance.get(key), value)
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"invalid staged raw checkpoint {checkpoint_id}: {mismatches}")
        if sum(str(key).startswith("online_encoder.") for key in staged.get("state_dict", {})) != 381:
            raise RuntimeError(f"unexpected encoder tensor count for {checkpoint_id}")
        entry = (str(staged_path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def main() -> None:
    configure()
    analysis.register = register_raw
    sys.argv = [sys.argv[0]]
    analysis.main()


if __name__ == "__main__":
    main()
