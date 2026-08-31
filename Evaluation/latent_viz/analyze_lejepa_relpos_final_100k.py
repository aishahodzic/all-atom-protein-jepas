"""Matched 100k t-SNE comparison for the two final relative-position LeJEPA runs."""

from __future__ import annotations

import pathlib
import sys

import torch
from loguru import logger


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(LINEAR_PROBING_DIR), str(HERE)]

CHECKPOINTS = {
    "le-jepa-r50-relpos-step100000": pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/le_jepa_rand50_relative_positions_10k/"
        "store/le_jepa_rand50_relative_positions_10k/checkpoints/"
        "chk_epoch=00000019_step=000000100000.ckpt"
    ),
    "le-jepa-r70-relpos-rotation-step100000": REPO_ROOT / (
        "StructTokenBench/100k-checkpoints/le-jepa/"
        "relative-position-and-rotation/100k.ckpt"
    ),
}


def main() -> None:
    import checkpoints

    for checkpoint_id, path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry

    from data_pool import build_protein_pool
    from extract_latents import LATENTS_DIR, extract

    missing = [
        checkpoint_id
        for checkpoint_id in CHECKPOINTS
        if not (LATENTS_DIR / f"{checkpoint_id}.pt").exists()
    ]
    logger.info(f"Missing 100k latent caches: {missing}")
    if missing:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to extract the missing latent caches")
        pool = build_protein_pool()
        for checkpoint_id in missing:
            extract(pool=pool, checkpoint_ids=[checkpoint_id], device="cuda")
            torch.cuda.empty_cache()

    # Import only after registering the two dynamic checkpoint identifiers.
    import run_tsne

    run_tsne.SHORT_LABEL.update({
        "le-jepa-r50-relpos-step100000": "LeJEPA r50 + relative positions (100k)",
        "le-jepa-r70-relpos-rotation-step100000": (
            "LeJEPA r70 + relative positions + rotations (100k)"
        ),
    })
    run_tsne.run(
        checkpoint_ids=list(CHECKPOINTS),
        n_points=15_000,
        perplexity=30.0,
        seed=42,
        max_iter=1_000,
        force=False,
        output_tag="lejepa-relpos-final-100k",
    )


if __name__ == "__main__":
    main()
