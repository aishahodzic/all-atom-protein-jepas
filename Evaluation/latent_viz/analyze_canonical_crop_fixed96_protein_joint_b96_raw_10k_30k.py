"""Matched raw-online 10k/30k t-SNE audit for cropped joint-SIGReg B96."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import tempfile

import torch


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
for path in (REPO, LINEAR, KNN, HERE):
    sys.path.insert(0, str(path))

import checkpoints  # noqa: E402
import color_canonical_crop_fixed96_four_ema_10k_position as position  # noqa: E402
import run_tsne as run_tsne_module  # noqa: E402
from Evaluation.canonical_lejepa.checkpoint import (  # noqa: E402
    stage_structtokenbench_checkpoint,
)
from extract_latents import extract  # noqa: E402
from run_tsne import run as run_tsne  # noqa: E402
from viz import SEED  # noqa: E402


RUN_NAME = "canonical_lejepa_crop_fixed96_protein_joint_sigreg_b96_10k"
RUN_ROOT = REPO / "store" / RUN_NAME
SPECS = {
    10_000: {
        "sha256": "173ce9054361ab197586c155302140e3037b24244ab9b1217a6bfb29bb2a7dd1",
        "label": "Protein loss / joint-view SIGReg / B96 / raw-10k",
    },
    30_000: {
        "sha256": "4a895be7e605ef5eff434a2e8c8c8f30cc1e2f682fab4bc26232155c580c11dc",
        "label": "Protein loss / joint-view SIGReg / B96 / raw-30k",
    },
}
OUTPUT_TAG = "canonical-crop-fixed96-protein-joint-b96-raw-10k-30k"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register() -> list[str]:
    config_path = RUN_ROOT / "checkpoints" / f"exp_config_{RUN_NAME}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["alignment_level"] == "protein"
    assert config["sigreg"]["input"] == "protein_mean"
    assert config["sigreg"]["joint_views"] is True
    assert config["dataset"]["datamodule"]["batch_size"] == 48
    assert config["trainer"]["devices"] == 2

    checkpoint_ids = []
    for step, spec in SPECS.items():
        checkpoint_id = (
            "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
            f"raw-step{step}"
        )
        raw = RUN_ROOT / f"checkpoints/step-{step:08d}.ckpt"
        if sha256_file(raw) != spec["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {raw}")
        checkpoint = torch.load(raw, map_location="cpu", weights_only=False, mmap=True)
        assert checkpoint.get("global_step") == step
        del checkpoint

        staged = RUN_ROOT / "evaluation" / (
            f"canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-step{step}-raw-staged.ckpt"
        )
        if not staged.is_file():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{staged.name}.", suffix=".tmp", dir=staged.parent
            )
            os.close(descriptor)
            temporary = pathlib.Path(temporary_name)
            try:
                stage_structtokenbench_checkpoint(raw, temporary, source="raw")
                os.replace(temporary, staged)
            finally:
                temporary.unlink(missing_ok=True)
        provenance = torch.load(
            staged, map_location="cpu", weights_only=False, mmap=True
        ).get("canonical_lejepa_evaluation", {})
        assert provenance.get("encoder_source") == "raw"
        assert provenance.get("source_global_step") == step
        assert provenance.get("source_checkpoint_sha256") == spec["sha256"]

        entry = (str(staged), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry
        run_tsne_module.SHORT_LABEL[checkpoint_id] = spec["label"]
        checkpoint_ids.append(checkpoint_id)
    return checkpoint_ids


def main() -> None:
    checkpoint_ids = register()
    extract(checkpoint_ids=checkpoint_ids, device="cuda")
    run_tsne(
        checkpoint_ids=checkpoint_ids,
        n_points=15_000,
        perplexity=30.0,
        seed=SEED,
        max_iter=1_000,
        force=False,
        output_tag=OUTPUT_TAG,
        grid_ncols=2,
        include_purity=False,
        show_suptitle=True,
    )
    position.SPECS = {
        checkpoint_id: SPECS[step]["label"]
        for checkpoint_id, step in zip(checkpoint_ids, SPECS)
    }
    position.FIGURE = HERE / f"figures/tsne_position_{OUTPUT_TAG}_grid.png"
    position.RESULT = HERE / f"results/tsne_position_metrics_{OUTPUT_TAG}.csv"
    position.main()


if __name__ == "__main__":
    main()
