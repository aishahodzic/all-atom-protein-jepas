"""Matched raw time-course for cropped per-protein canonical LeJEPA + rotations."""

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


ROOT = REPO / "StructTokenBench/100k-checkpoints/canonical-lejepa/jarvis/rotation"
SPECS = (
    {
        "id": "canonical-lejepa-jarvis-rotation-raw-step10000",
        "step": 10_000,
        "source": ROOT / "CLJP_step-00010000.ckpt",
        "staged": ROOT / "CLJP_step-00010000-raw-stb.ckpt",
        "sha256": "06bec76f5bf695ef930f494cdc5449c34375854b91fe53f7876bf74edd897607",
        "label": "Raw 10k",
    },
    {
        "id": "canonical-lejepa-protein-perview-rot-middle-lr-raw-step15980",
        "step": 15_980,
        "source": ROOT / "CLJP_last.ckpt",
        "staged": ROOT / "CLJP_step-00015980-raw-stb.ckpt",
        "sha256": "4126b308c6887c396c31a8a24faaa0c69922fecb24d70f8c0c27433162d3fc3f",
        "label": "Raw 15,980",
    },
)


def configure() -> None:
    analysis.CHECKPOINTS.clear()
    analysis.CHECKPOINT_STEPS.clear()
    analysis.STAGED.clear()
    analysis.ROW_LABELS.clear()
    for spec in SPECS:
        checkpoint_id = str(spec["id"])
        analysis.CHECKPOINTS[checkpoint_id] = pathlib.Path(spec["source"])
        analysis.CHECKPOINT_STEPS[checkpoint_id] = int(spec["step"])
        analysis.STAGED[checkpoint_id] = pathlib.Path(spec["staged"])
        analysis.ROW_LABELS[checkpoint_id] = str(spec["label"])

    stem = "canonical_lejepa_crop_protein_rotation_middle_lr_raw_10k_15980"
    analysis.RESULT = HERE / f"results/{stem}_position.csv"
    analysis.MANIFEST = HERE / f"results/{stem}_position.json"
    analysis.POSITION_FIGURE = HERE / f"figures/canonical_lejepa/tsne_{stem}_position-and-length.png"
    analysis.RESIDUE_FIGURE = HERE / f"figures/canonical_lejepa/tsne_{stem}_aa-and-ss.png"
    analysis.ANALYSIS_LABEL = "Cropped per-protein LeJEPA + rotations, Middle LR"
    analysis.OUTPUT_TAG = "canonical-lejepa-crop-protein-rotation-middle-lr-raw-10k-v-15980"
    analysis.MANIFEST_DESCRIPTION = "Matched raw 10k/15,980 time-course for cropped per-protein canonical LeJEPA with independent local crops and rotations"
    analysis.ENCODER_SOURCE = "raw"
    analysis.ENCODER_LABEL = "Raw"
    run_tsne_module.SHORT_LABEL.update(analysis.ROW_LABELS)


def register_raw() -> None:
    for spec in SPECS:
        checkpoint_id = str(spec["id"])
        staged_path = pathlib.Path(spec["staged"])
        if not staged_path.is_file():
            raise FileNotFoundError(staged_path)
        staged = torch.load(staged_path, map_location="cpu", weights_only=False, mmap=True)
        provenance = staged.get("canonical_lejepa_evaluation", {})
        expected = {
            "encoder_source": "raw",
            "source_global_step": int(spec["step"]),
            "source_checkpoint_sha256": str(spec["sha256"]),
            "latent_dim": 8,
        }
        mismatches = {
            key: (provenance.get(key), value)
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"invalid staged checkpoint {checkpoint_id}: {mismatches}")
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
