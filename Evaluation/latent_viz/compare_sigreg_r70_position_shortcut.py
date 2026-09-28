"""Matched 100k JEPA+SIGReg r70 absolute/relative-position shortcut audit."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
sys.path[:0] = [str(REPO), str(LINEAR), str(KNN), str(HERE)]

from thesis_plot_style import BLACK, apply_thesis_style

IDS = ("jepa-sigreg-r70-absolute-100k", "jepa-sigreg-r70-relative-100k")
NAMES = {
    IDS[0]: "JEPA+SIGReg\nusing absolute positions",
    IDS[1]: "JEPA+SIGReg\nusing relative positions",
}
N_POINTS = 15_000
SEED = 42
PERPLEXITY = 30.0
K = 10
RESULT = HERE / "results" / "jepa_sigreg_r70_absolute_relative_100k_position.csv"
COORD_DIR = HERE / "results" / "tsne_coordinates"
FIGURE = (
    REPO
    / "thesis-latex"
    / "figures"
    / "results"
    / "tsne_jepa_sigreg_r70_absolute-v-relative_absolute-index_100k.png"
)


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def external_neighbours(
    values: np.ndarray, protein_index: np.ndarray, k: int = K
) -> np.ndarray:
    """Return neighbours from other proteins so index matches are not trivial."""
    unit_values = normalized(values)
    _, candidates = cKDTree(unit_values).query(unit_values, k=64, workers=-1)
    neighbours = np.empty((len(values), k), dtype=np.int64)
    for row, candidate_row in enumerate(candidates):
        usable = candidate_row[protein_index[candidate_row] != protein_index[row]]
        if len(usable) < k:
            raise RuntimeError("Insufficient cross-protein neighbour buffer")
        neighbours[row] = usable[:k]
    return neighbours


def exact_index_purity(
    residue_index: np.ndarray, neighbours: np.ndarray
) -> tuple[float, float]:
    matches = residue_index[neighbours] == residue_index[:, None]
    return float(matches.mean()), float(matches[:, 0].mean())


def effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values.astype(np.float64), rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / np.maximum(eigenvalues.sum(), 1e-12)
    nonzero = probabilities[probabilities > 0]
    return float(np.exp(-(nonzero * np.log(nonzero)).sum()))


def annotate_indices(
    ax: plt.Axes, embedding: np.ndarray, residue_index: np.ndarray
) -> None:
    for index0 in np.unique(residue_index):
        index1 = int(index0) + 1
        if index1 != 1 and index1 % 25 != 0:
            continue
        mask = residue_index == index0
        if mask.sum() < 3:
            continue
        points = embedding[mask]
        center = np.median(points, axis=0)
        x, y = points[np.argmin(np.square(points - center).sum(axis=1))]
        label = ax.text(
            x, y, str(index1), fontsize=9, weight="bold", ha="center", va="center"
        )
        label.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.4, foreground="white"),
                path_effects.Normal(),
            ]
        )


def render_figure(
    rows: list[dict[str, object]],
    embeddings: dict[str, np.ndarray],
    residue_index: np.ndarray,
) -> None:
    apply_thesis_style()
    index_colors = plt.get_cmap("turbo")(np.linspace(0.0, 1.0, 256))
    luminance = index_colors[:, :3] @ np.array([0.2126, 0.7152, 0.0722])
    index_colors[luminance > 0.62, :3] *= 0.84
    index_cmap = ListedColormap(index_colors, name="turbo_high_contrast")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    scatter = None
    for ax, row in zip(axes, rows):
        checkpoint_id = str(row["checkpoint"])
        embedding = embeddings[checkpoint_id]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=residue_index + 1,
            cmap=index_cmap,
            vmin=1,
            vmax=251,
            s=4,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
        annotate_indices(ax, embedding, residue_index)
        ax.set_title(NAMES[checkpoint_id], fontsize=14.5, color=BLACK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.91, bottom=0.02, top=0.90, wspace=0.03)
    colorbar_ax = fig.add_axes([0.925, 0.14, 0.015, 0.62])
    colorbar = fig.colorbar(scatter, cax=colorbar_ax)
    colorbar.set_label("absolute residue index", fontsize=14, color=BLACK)
    colorbar.ax.tick_params(labelsize=12, colors=BLACK)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_from_caches() -> None:
    import checkpoints
    from knn import common_protein_ids, load_encoder_latents, load_labels
    from label_tsne_worms import local_index_array
    from viz import CombinedSelection, sample_indices

    latent_caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in IDS
    }
    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, load_labels())
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    residue_index = local_index_array(selection)[sampled]
    embeddings = {}
    for checkpoint_id in IDS:
        coordinate = COORD_DIR / (
            f"{checkpoint_id}_n{N_POINTS}_perp30p0_seed{SEED}_cosine.npz"
        )
        embeddings[checkpoint_id] = np.load(coordinate, allow_pickle=False)["embedding"]
    with RESULT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    render_figure(rows, embeddings, residue_index)
    print(f"Wrote {FIGURE}")


def main() -> None:
    absolute_checkpoint = pathlib.Path(os.environ["ABS_CHECKPOINT"])
    relative_checkpoint = pathlib.Path(os.environ["REL_CHECKPOINT"])
    for path in (absolute_checkpoint, relative_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    import checkpoints
    from extract_latents import extract
    from knn import common_protein_ids, load_encoder_latents, load_labels
    from label_tsne_worms import local_index_array
    from viz import CombinedSelection, sample_indices

    checkpoint_paths = dict(zip(IDS, (absolute_checkpoint, relative_checkpoint)))
    for checkpoint_id, path in checkpoint_paths.items():
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry

    extract(checkpoint_ids=list(IDS), device="cuda")

    labels = load_labels()
    latent_caches = {
        checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in IDS
    }
    protein_ids = common_protein_ids(latent_caches)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    residue_index = local_index_array(selection)[sampled]
    protein_index = selection.protein_idx[sampled]

    rows = []
    embeddings = {}
    COORD_DIR.mkdir(parents=True, exist_ok=True)
    for checkpoint_id in IDS:
        values = selection.build_X(latent_caches[checkpoint_id])[sampled].astype(np.float32)
        latent_neighbours = external_neighbours(values, protein_index)
        latent_purity, latent_nn = exact_index_purity(residue_index, latent_neighbours)

        embedding = TSNE(
            n_components=2,
            perplexity=PERPLEXITY,
            early_exaggeration=12.0,
            learning_rate="auto",
            max_iter=1_000,
            metric="cosine",
            init="random",
            random_state=SEED,
            method="barnes_hut",
        ).fit_transform(values)
        embeddings[checkpoint_id] = embedding
        coordinate_path = COORD_DIR / (
            f"{checkpoint_id}_n{N_POINTS}_perp30p0_seed{SEED}_cosine.npz"
        )
        np.savez_compressed(coordinate_path, embedding=embedding)

        _, tsne_candidates = cKDTree(embedding).query(embedding, k=K + 1, workers=-1)
        tsne_purity, tsne_nn = exact_index_purity(
            residue_index, tsne_candidates[:, 1:]
        )
        rows.append(
            {
                "checkpoint": checkpoint_id,
                "checkpoint_path": str(checkpoint_paths[checkpoint_id]),
                "proteins": len(protein_ids),
                "sampled_residues": N_POINTS,
                "latent_dim": values.shape[1],
                "effective_rank": effective_rank(values),
                "latent_cross_protein_exact_index_purity_at_10": latent_purity,
                "latent_cross_protein_nearest_neighbour_exact_index": latent_nn,
                "tsne_exact_index_purity_at_10": tsne_purity,
                "tsne_nearest_neighbour_exact_index": tsne_nn,
            }
        )

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    render_figure(rows, embeddings, residue_index)

    print(f"Wrote {RESULT}")
    print(f"Wrote {FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from existing latent and t-SNE caches.",
    )
    args = parser.parse_args()
    plot_from_caches() if args.plot_only else main()
