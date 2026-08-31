"""Render the requested matched 100k t-SNE comparison panels."""

from __future__ import annotations

import argparse
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
sys.path[:0] = [str(REPO), str(LINEAR), str(HERE)]

FULL = [
    "random-init",
    "ca-cond",
    "fully-latent",
    "jepa-rand-mask-50",
    "jepa-rand-mask-70",
    "jepa-mutation",
    "le-jepa-rand-mask-70",
    "le-jepa-r70-relpos-step100000",
    "le-jepa-spatial3d-d64",
    "le-jepa-sp3d-relpos-d8-step100000",
    "le-jepa-r50-relpos-step100000",
    "le-jepa-r70-relpos-rotation-step100000",
]

REDUCED = [
    "ca-cond",
    "jepa-rand-mask-70",
    "jepa-mutation",
    "le-jepa-rand-mask-70",
    "le-jepa-r70-relpos-step100000",
    "le-jepa-r70-relpos-rotation-step100000",
]

CUSTOM = {
    "le-jepa-r70-relpos-step100000": pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/le_jepa_rand70_relative_positions_100k/"
        "store/le_jepa_rand70_relative_positions_100k/checkpoints/"
        "chk_epoch=00000019_step=000000100000.ckpt"
    ),
    "le-jepa-r50-relpos-step100000": pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/le_jepa_rand50_relative_positions_10k/"
        "store/le_jepa_rand50_relative_positions_10k/checkpoints/"
        "chk_epoch=00000019_step=000000100000.ckpt"
    ),
    "le-jepa-r70-relpos-rotation-step100000": REPO / (
        "StructTokenBench/100k-checkpoints/le-jepa/"
        "relative-position-and-rotation/100k.ckpt"
    ),
    "le-jepa-sp3d-relpos-d8-step100000": pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/"
        "le_jepa_spatial3d_relative_positions_d8_100k/store/"
        "le_jepa_spatial3d_relative_positions_d8_100k/checkpoints/"
        "chk_epoch=00000019_step=000000100000.ckpt"
    ),
}

LABELS = {
    "random-init": "Random-init, D8",
    "ca-cond": "Cα-conditioned AE, D8",
    "fully-latent": "Fully latent AE, D8",
    "jepa-rand-mask-50": "JEPA-r50, D8",
    "jepa-rand-mask-70": "JEPA-r70, D8",
    "jepa-mutation": "Mutation-JEPA, D8",
    "le-jepa-rand-mask-70": "LeJEPA absolute-position r70, D8",
    "le-jepa-r70-relpos-step100000": "LeJEPA relative-position r70, D8",
    "le-jepa-spatial3d-d64": "LeJEPA spatial-3D, D64",
    "le-jepa-sp3d-relpos-d8-step100000": (
        "LeJEPA spatial-3D + relative-position, D8"
    ),
    "le-jepa-r50-relpos-step100000": "LeJEPA relative-position r50, D8",
    "le-jepa-r70-relpos-rotation-step100000": (
        "LeJEPA relative-position + rotation r70, D8"
    ),
}


def register(require_latents: bool = True, checkpoint_ids=FULL) -> None:
    import checkpoints
    from publication_100k import register_publication_checkpoints

    register_publication_checkpoints()
    for checkpoint_id, path in CUSTOM.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = (str(path), "jepa")

    missing = [
        checkpoint_id for checkpoint_id in checkpoint_ids
        if not (LINEAR / "latents" / f"{checkpoint_id}.pt").is_file()
    ]
    if require_latents and missing:
        raise FileNotFoundError(f"Missing latent caches: {missing}")


def main(extract_spatial: bool = False, reduced_only: bool = False) -> None:
    required = REDUCED if reduced_only else FULL
    register(require_latents=not extract_spatial, checkpoint_ids=required)
    if extract_spatial:
        from extract_latents import extract

        extract(
            checkpoint_ids=["le-jepa-sp3d-relpos-d8-step100000"],
            device="cuda",
        )
    import run_tsne

    run_tsne.SHORT_LABEL.update(LABELS)
    common = dict(
        n_points=15_000,
        perplexity=30.0,
        seed=42,
        max_iter=1_000,
        force=False,
        include_purity=False,
        show_suptitle=False,
    )
    if not reduced_only:
        run_tsne.run(
            checkpoint_ids=FULL,
            output_tag="requested-full-12model-100k",
            grid_ncols=4,
            **common,
        )
    run_tsne.run(
        checkpoint_ids=REDUCED,
        output_tag="requested-reduced-6model-100k",
        grid_ncols=3,
        **common,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract-spatial",
        action="store_true",
        help="Extract the missing spatial-3D relative-position D8 100k cache first",
    )
    parser.add_argument(
        "--reduced-only",
        action="store_true",
        help="Refresh only the six-model 2x3 panel",
    )
    args = parser.parse_args()
    main(
        extract_spatial=args.extract_spatial,
        reduced_only=args.reduced_only,
    )
