"""Matched t-SNE panels for four fixed-96/48 canonical LeJEPA EMA runs."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import torch


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)
from extract_latents import extract  # noqa: E402
from run_tsne import run as run_tsne  # noqa: E402
import run_tsne as run_tsne_module  # noqa: E402
from viz import SEED  # noqa: E402


SPECS = {
    "canonical-lejepa-residue-cropped-fixed96-perview-sigreg-b96-ema-step10000": {
        "run": "canonical_lejepa_crop_fixed96_residue_perview_sigreg_b96_100k_relaunch",
        "stage": "canonical-lejepa-residue-cropped-fixed96-perview-sigreg-b96-step10000-ema-staged.ckpt",
        "sha256": "639a11a9dc84692ebe477cc09ee8c63eb781a546f1187158fd6792a226be1ce6",
        "alignment": "residue",
        "sigreg_input": "protein_centered_residue",
        "joint_views": False,
        "per_device_batch": 48,
        "label": "Residue loss / per-view SIGReg / B96",
    },
    "canonical-lejepa-protein-cropped-fixed96-perview-sigreg-b96-ema-step10000": {
        "run": "canonical_lejepa_crop_fixed96_protein_perview_sigreg_b96_100k",
        "stage": "canonical-lejepa-protein-cropped-fixed96-perview-sigreg-b96-step10000-ema-staged.ckpt",
        "sha256": "1ee8aaa31705fb353ed19d06c37c0e2347dbf7499878e6c62d078abbe4f4e054",
        "alignment": "protein",
        "sigreg_input": "protein_mean",
        "joint_views": False,
        "per_device_batch": 48,
        "label": "Protein loss / per-view SIGReg / B96",
    },
    "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-ema-step10000": {
        "run": "canonical_lejepa_crop_fixed96_protein_joint_sigreg_b96_10k",
        "stage": "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-step10000-ema-staged.ckpt",
        "sha256": "173ce9054361ab197586c155302140e3037b24244ab9b1217a6bfb29bb2a7dd1",
        "alignment": "protein",
        "sigreg_input": "protein_mean",
        "joint_views": True,
        "per_device_batch": 48,
        "label": "Protein loss / joint-view SIGReg / B96",
    },
    "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b32-ema-step10000": {
        "run": "canonical_lejepa_crop_fixed96_protein_joint_sigreg_b32_100k",
        "stage": "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b32-step10000-ema-staged.ckpt",
        "sha256": "0037f4cb00122f42729a4be3791be2c68bd9a99be3e5e599602a8454c502c905",
        "alignment": "protein",
        "sigreg_input": "protein_mean",
        "joint_views": True,
        "per_device_batch": 16,
        "label": "Protein loss / joint-view SIGReg / B32",
    },
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register() -> None:
    for checkpoint_id, spec in SPECS.items():
        run_root = REPO / "store" / str(spec["run"])
        raw = run_root / "checkpoints/step-00010000.ckpt"
        config_path = run_root / "checkpoints" / f"exp_config_{spec['run']}.json"
        staged = run_root / "evaluation" / str(spec["stage"])
        if sha256_file(raw) != spec["sha256"]:
            raise RuntimeError(f"raw checkpoint hash mismatch: {raw}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["alignment_level"] == spec["alignment"]
        assert config["sigreg"]["input"] == spec["sigreg_input"]
        assert config["sigreg"]["joint_views"] is spec["joint_views"]
        assert config["dataset"]["datamodule"]["batch_size"] == spec["per_device_batch"]
        assert config["trainer"]["devices"] == 2
        views = config["views"]
        assert views["cropping"] is True and views["packed_views"] is True
        assert views["global_crop_length"] == 96 and views["local_crop_length"] == 48
        checkpoint = torch.load(raw, map_location="cpu", weights_only=False, mmap=True)
        assert checkpoint.get("global_step") == 10_000
        meta = checkpoint.get("state_dict_ema_meta", {})
        assert meta.get("num_updates") == 312 and meta.get("last_ema_step") == 9_984
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.is_file():
            stage_structtokenbench_checkpoint(raw, staged, source="ema")
        staged_checkpoint = torch.load(
            staged, map_location="cpu", weights_only=False, mmap=True
        )
        provenance = staged_checkpoint.get("canonical_lejepa_evaluation", {})
        assert provenance.get("encoder_source") == "ema"
        assert provenance.get("source_global_step") == 10_000
        assert provenance.get("source_checkpoint_sha256") == spec["sha256"]
        entry = (str(staged), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry
        run_tsne_module.SHORT_LABEL[checkpoint_id] = str(spec["label"])


def main() -> None:
    register()
    checkpoint_ids = list(SPECS)
    extract(checkpoint_ids=checkpoint_ids, device="cuda")
    run_tsne(
        checkpoint_ids=checkpoint_ids,
        n_points=15_000,
        perplexity=30.0,
        seed=SEED,
        max_iter=1_000,
        force=False,
        output_tag="canonical-crop-fixed96-four-ema-10k",
        grid_ncols=2,
        include_purity=False,
        show_suptitle=True,
    )


if __name__ == "__main__":
    main()
