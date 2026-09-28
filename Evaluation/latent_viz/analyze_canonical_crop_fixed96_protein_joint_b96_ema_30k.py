"""EMA-30k t-SNE and positional audit for cropped joint-SIGReg B96."""

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


CHECKPOINT_ID = (
    "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-ema-step30000"
)
RUN_NAME = "canonical_lejepa_crop_fixed96_protein_joint_sigreg_b96_10k"
RAW = REPO / "store" / RUN_NAME / "checkpoints/step-00030000.ckpt"
STAGED = REPO / "store" / RUN_NAME / "evaluation" / (
    "canonical-lejepa-protein-cropped-fixed96-joint-sigreg-b96-"
    "step30000-ema-staged.ckpt"
)
EXPECTED_SHA256 = "4a895be7e605ef5eff434a2e8c8c8f30cc1e2f682fab4bc26232155c580c11dc"
LABEL = "Protein loss / joint-view SIGReg / B96 / EMA-30k"
OUTPUT_TAG = "canonical-crop-fixed96-protein-joint-b96-ema-30k"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register() -> None:
    if sha256_file(RAW) != EXPECTED_SHA256:
        raise RuntimeError(f"raw checkpoint hash mismatch: {RAW}")
    config_path = RAW.parent / f"exp_config_{RUN_NAME}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["alignment_level"] == "protein"
    assert config["sigreg"]["input"] == "protein_mean"
    assert config["sigreg"]["joint_views"] is True
    assert config["dataset"]["datamodule"]["batch_size"] == 48
    assert config["trainer"]["devices"] == 2
    views = config["views"]
    assert views["cropping"] is True and views["packed_views"] is True
    assert views["global_crop_length"] == 96 and views["local_crop_length"] == 48

    checkpoint = torch.load(RAW, map_location="cpu", weights_only=False, mmap=True)
    assert checkpoint.get("global_step") == 30_000
    meta = checkpoint.get("state_dict_ema_meta", {})
    assert meta.get("num_updates") == 937 and meta.get("last_ema_step") == 29_984
    del checkpoint

    STAGED.parent.mkdir(parents=True, exist_ok=True)
    if not STAGED.is_file():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{STAGED.name}.", suffix=".tmp", dir=STAGED.parent
        )
        os.close(descriptor)
        temporary = pathlib.Path(temporary_name)
        try:
            stage_structtokenbench_checkpoint(RAW, temporary, source="ema")
            os.replace(temporary, STAGED)
        finally:
            temporary.unlink(missing_ok=True)
    staged = torch.load(STAGED, map_location="cpu", weights_only=False, mmap=True)
    provenance = staged.get("canonical_lejepa_evaluation", {})
    assert provenance.get("encoder_source") == "ema"
    assert provenance.get("source_global_step") == 30_000
    assert provenance.get("source_checkpoint_sha256") == EXPECTED_SHA256

    entry = (str(STAGED), "jepa")
    checkpoints.CHECKPOINTS[CHECKPOINT_ID] = entry
    checkpoints.ALL_CHECKPOINTS[CHECKPOINT_ID] = entry
    run_tsne_module.SHORT_LABEL[CHECKPOINT_ID] = LABEL


def main() -> None:
    register()
    extract(checkpoint_ids=[CHECKPOINT_ID], device="cuda")
    run_tsne(
        checkpoint_ids=[CHECKPOINT_ID],
        n_points=15_000,
        perplexity=30.0,
        seed=SEED,
        max_iter=1_000,
        force=False,
        output_tag=OUTPUT_TAG,
        grid_ncols=1,
        include_purity=False,
        show_suptitle=True,
    )
    position.SPECS = {CHECKPOINT_ID: LABEL}
    position.FIGURE = HERE / f"figures/tsne_position_{OUTPUT_TAG}_grid.png"
    position.RESULT = HERE / f"results/tsne_position_metrics_{OUTPUT_TAG}.csv"
    position.main()


if __name__ == "__main__":
    main()
