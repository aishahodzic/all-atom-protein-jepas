"""Exact step-matched JEPA-r70 D64 60k comparison with AA labels.

Use --from-latents to render from frozen latent caches without loading models.
"""

from __future__ import annotations

import argparse
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
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from run_tsne import (  # noqa: E402
    annotate_aa_locations,
    coordinate_path,
    load_or_fit,
    run as run_tsne,
    scatter_panel,
)
from thesis_plot_style import BLACK, apply_thesis_style  # noqa: E402
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
ANNOTATE_ID = "jepa-r70-d64-rot-step60000"
TITLES = {
    "jepa-r70-d64-step60000": "JEPA-r70 D64",
    "jepa-r70-d64-rot-step60000": "JEPA-r70 D64 with rotations",
}
N_POINTS = 15_000
PERPLEXITY = 30.0
MAX_ITER = 1_000
FILENAME = "tsne_aa20_jepa-r70-d64-60k-v-rotation-60k_labels-on-clusters.png"
OUTPUTS = (
    HERE / "figures-extended" / "workshop" / "main" / FILENAME,
    REPO / "thesis-latex" / "figures" / "results" / FILENAME,
)


def register() -> None:
    for checkpoint_id, path in CHECKPOINT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry


def main(from_latents: bool = False) -> None:
    lm_family = apply_thesis_style()
    if not from_latents:
        from extract_latents import extract

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
    latent_paths = {
        checkpoint_id: LINEAR / "latents" / f"{checkpoint_id}.pt"
        for checkpoint_id in CHECKPOINT_IDS
    }
    have_latents = all(path.exists() for path in latent_paths.values())
    latent_caches = (
        {
            checkpoint_id: load_encoder_latents(checkpoint_id)
            for checkpoint_id in CHECKPOINT_IDS
        }
        if have_latents else {}
    )
    protein_ids = common_protein_ids(latent_caches) if have_latents else sorted(labels)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    aa = selection.aa[sampled]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), squeeze=False)
    for ax, checkpoint_id in zip(axes.ravel(), CHECKPOINT_IDS):
        if have_latents:
            X = selection.build_X(latent_caches[checkpoint_id])[sampled]
            embedding, _, _ = load_or_fit(
                checkpoint_id, X, perplexity=PERPLEXITY, seed=SEED,
                max_iter=MAX_ITER, force=False,
            )
        else:
            path = coordinate_path(checkpoint_id, N_POINTS, PERPLEXITY, SEED)
            cached = np.load(path, allow_pickle=False)
            embedding = cached["embedding"]
        if len(embedding) != len(sampled):
            raise ValueError(
                f"{checkpoint_id}: {len(embedding)} coordinates != "
                f"{len(sampled)} sampled residues"
            )
        scatter_panel(ax, embedding, aa, view="aa20", annotate_aa=False)
        for collection in ax.collections:
            collection.set_alpha(0.92)
            collection.set_sizes([4.6])
        if checkpoint_id == ANNOTATE_ID:
            annotate_aa_locations(ax, embedding, aa, ["black"] * 20)
            for annotation in ax.texts:
                annotation.set_fontsize(8.5)
                annotation.set_color(BLACK)
                annotation.set_fontfamily(lm_family)
                annotation.set_fontweight(600)
        ax.set_title(TITLES[checkpoint_id], fontsize=14, color=BLACK)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=2.0)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-latents", action="store_true",
        help="Use existing latent caches; do not load checkpoints or re-extract latents.",
    )
    main(from_latents=parser.parse_args().from_latents)
