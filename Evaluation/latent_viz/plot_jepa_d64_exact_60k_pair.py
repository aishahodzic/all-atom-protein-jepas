"""Exact step-matched JEPA-r70 D64 60k comparison with AA labels."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
sys.path[:0] = [str(REPO), str(LINEAR), str(KNN), str(HERE)]

import checkpoints  # noqa: E402
from extract_latents import extract  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from run_tsne import (  # noqa: E402
    annotate_aa_locations,
    coordinate_path,
    run as run_tsne,
    scatter_panel,
)
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


RUN_ROOT = pathlib.Path("/mnt/hopper-raid/aishod/runs")
CHECKPOINT_PATHS = {
    "jepa-r70-d64-step60000": RUN_ROOT / (
        "jepa_rand70_dim64/store/jepa_rand70_dim64/checkpoints/"
        "chk_epoch=00000011_step=000000060000.ckpt"
    ),
    "jepa-r70-d64-rot-step60000": RUN_ROOT / (
        "jepa_rand70_d64_independent_rotations_10k/store/"
        "jepa_rand70_d64_independent_rotations_10k/checkpoints/"
        "chk_epoch=00000011_step=000000060000.ckpt"
    ),
}
CHECKPOINT_IDS = tuple(CHECKPOINT_PATHS)
TITLES = {
    "jepa-r70-d64-step60000": "JEPA-r70 D64 (60k)",
    "jepa-r70-d64-rot-step60000": "JEPA-r70 D64 with rotations (60k)",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
OUTPUT = HERE / "figures" / "workshop" / "main" / (
    "tsne_aa20_jepa-r70-d64-60k-v-rotation-60k_labels-on-clusters.png"
)


def register() -> None:
    for checkpoint_id, path in CHECKPOINT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def main() -> None:
    register()
    extract(checkpoint_ids=list(CHECKPOINT_IDS), device="cuda")
    run_tsne(
        checkpoint_ids=list(CHECKPOINT_IDS),
        n_points=N_POINTS,
        perplexity=PERPLEXITY,
        seed=SEED,
        max_iter=MAX_ITER,
        force=False,
        output_tag="jepa-r70-d64-exact-60k-pair",
        grid_ncols=2,
        include_purity=False,
        show_suptitle=False,
    )

    labels = load_labels()
    latent_caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in CHECKPOINT_IDS
    }
    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    aa = selection.aa[sampled]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), squeeze=False)
    for ax, checkpoint_id in zip(axes.ravel(), CHECKPOINT_IDS):
        path = coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED)
        embedding = np.load(path, allow_pickle=False)["embedding"]
        if len(embedding) != len(sampled):
            raise ValueError(
                f"{checkpoint_id}: {len(embedding)} coordinates != "
                f"{len(sampled)} sampled residues"
            )
        scatter_panel(ax, embedding, aa, view="aa20", annotate_aa=False)
        for collection in ax.collections:
            collection.set_alpha(0.92)
            collection.set_sizes([4.6])
        if checkpoint_id == "jepa-r70-d64-rot-step60000":
            annotate_aa_locations(ax, embedding, aa, ["black"] * 20)
        ax.set_title(TITLES[checkpoint_id], fontsize=10)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=2.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
