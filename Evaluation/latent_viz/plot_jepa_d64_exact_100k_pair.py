"""Exact step-matched JEPA-r70 D64 100k comparison with AA labels."""

from __future__ import annotations

import pathlib

import plot_jepa_d64_exact_60k_pair as comparison


REPO = pathlib.Path(__file__).resolve().parents[2]
RUN_ROOT = pathlib.Path("/mnt/hopper-raid/aishod/runs")

comparison.CHECKPOINT_PATHS = {
    "jepa-r70-d64-step100000": REPO / (
        "StructTokenBench/100k-checkpoints/bottle-inspect/64dim/jepa/100k.ckpt"
    ),
    "jepa-r70-d64-rot-step100000": RUN_ROOT / (
        "jepa_rand70_d64_independent_rotations_10k/store/"
        "jepa_rand70_d64_independent_rotations_10k/checkpoints/"
        "chk_epoch=00000019_step=000000100000.ckpt"
    ),
}
comparison.CHECKPOINT_IDS = tuple(comparison.CHECKPOINT_PATHS)
comparison.ANNOTATE_ID = "jepa-r70-d64-rot-step100000"
comparison.TITLES = {
    "jepa-r70-d64-step100000": "JEPA-r70 D64",
    "jepa-r70-d64-rot-step100000": "JEPA-r70 D64 with rotations",
}
comparison.OUTPUT = comparison.HERE / "figures-extended" / "workshop" / "main" / (
    "tsne_aa20_jepa-r70-d64-100k-v-rotation-100k_labels-on-clusters.png"
)


if __name__ == "__main__":
    comparison.main(from_latents=False)
