"""La-Proteina-style t-SNE views of cached per-residue protein latents.

This is deliberately a visualization-only pass: it reuses the frozen,
full-context residue latents already extracted by Evaluation/linear_probing.
No checkpoint is loaded and no encoder forward pass is repeated.

The default comparison contains the eight primary 100k in-house checkpoints,
the published external AE reference, and the fixed random-initialized encoder.
The same seeded residue indices and the same t-SNE settings are used for every
encoder.  Each encoder's t-SNE coordinates are then recolored without refitting
for four views: 20 amino-acid identities, charge, aromaticity, and side-chain
size.

Usage:
    python Evaluation/latent_viz/run_tsne.py
    python Evaluation/latent_viz/run_tsne.py --n-points 15000 --device cpu
    python Evaluation/latent_viz/run_tsne.py --checkpoints ca-cond jepa-rand-mask-50
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.manifold import TSNE

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
KNN_PURITY_DIR = REPO_ROOT / "Evaluation" / "knn_purity"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LINEAR_PROBING_DIR))
sys.path.insert(0, str(KNN_PURITY_DIR))

from checkpoints import ALL_CHECKPOINTS  # noqa: E402
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from viz import (  # noqa: E402
    AA_LETTERS,
    CombinedSelection,
    DEFAULT_N_POINTS,
    SEED,
SHORT_LABEL,
    sample_indices,
)

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
COORD_DIR = RESULTS_DIR / "tsne_coordinates"
PURITY_CSV = KNN_PURITY_DIR / "results" / "knn_purity_results.csv"
METRICS_CSV = RESULTS_DIR / "tsne_panel_metrics.csv"
MANIFEST_JSON = RESULTS_DIR / "tsne_manifest.json"

MAIN_CHECKPOINTS = [
    "ca-cond",
    "fully-latent",
    "jepa-mutation",
    "jepa-rand-mask-50",
    "jepa-rand-mask-70",
    "le-jepa-mutation",
    "le-jepa-rand-mask-50",
    "le-jepa-rand-mask-70",
    "ae1-external",
    "random-init",
]

ALL_CACHED_CHECKPOINTS = list(ALL_CHECKPOINTS)
SHORT_LABEL.update({
    "jepa-rand50-control-30k": "JEPA-50 control (30k)",
    "jepa-rand50-ema-30k": "JEPA-50 + EMA (30k)",
    "jepa-rand50-layernorm-30k": "JEPA-50 + LayerNorm (30k)",
    "jepa-rand50-predictor-30k": "JEPA-50 + predictor (30k)",
})

AA_THREE = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

VIEW_SPECS = {
    "charge": {
        "labels": ["Basic / positive (R,K,H)", "Acidic / negative (D,E)", "Neutral"],
        "colors": ["#2166ac", "#b2182b", "#bdbdbd"],
        "sets": [set("RKH"), set("DE")],
    },
    "aromatic": {
        "labels": ["Aromatic (F,W,Y)", "Non-aromatic"],
        "colors": ["#7b3294", "#c2a5cf"],
        "sets": [set("FWY")],
    },
    "size": {
        "labels": ["Small (A,C,D,G,N,P,S,T)", "Larger (remaining 12)"],
        "colors": ["#1b9e77", "#d95f02"],
        "sets": [set("ACDGNPST")],
    },
}


def load_aa_purity_reference(k: int = 10) -> Dict[str, float]:
    result: Dict[str, float] = {}
    with open(PURITY_CSV) as f:
        for row in csv.DictReader(f):
            if row["property"] == "aa" and int(row["k"]) == k:
                result[row["encoder"]] = float(row["purity"])
    return result


def project_tsne(
    X: np.ndarray,
    perplexity: float,
    seed: int,
    max_iter: int,
) -> np.ndarray:
    """Fit a deterministic 2D t-SNE using cosine distance.

    Cosine matches the primary geometry used by the quantitative k-NN purity
    analysis.  Coordinates from separate fits have arbitrary orientation and
    scale, so panel axes are intentionally hidden.
    """
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=max_iter,
        metric="cosine",
        # Avoid injecting an Euclidean PCA initialization into an analysis
        # whose declared neighbourhood geometry is cosine.
        init="random",
        random_state=seed,
        method="barnes_hut",
        angle=0.5,
        n_jobs=-1,
        verbose=1,
    )
    return model.fit_transform(X).astype(np.float32)


def input_digest(X: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(X, dtype=np.float32)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def coordinate_path(checkpoint_id: str, n_points: int, perplexity: float, seed: int) -> pathlib.Path:
    p = str(perplexity).replace(".", "p")
    return COORD_DIR / f"{checkpoint_id}_n{n_points}_perp{p}_seed{seed}_cosine.npz"


def load_or_fit(
    checkpoint_id: str,
    X: np.ndarray,
    perplexity: float,
    seed: int,
    max_iter: int,
    force: bool,
) -> Tuple[np.ndarray, pathlib.Path, str]:
    digest = input_digest(X)
    path = coordinate_path(checkpoint_id, len(X), perplexity, seed)
    if path.exists() and not force:
        cached = np.load(path, allow_pickle=False)
        cached_digest = str(cached["input_sha256"].item())
        cached_max_iter = int(cached["max_iter"].item())
        if cached_digest == digest and cached_max_iter == max_iter:
            logger.info(f"{checkpoint_id}: reusing {path}")
            return cached["embedding"], path, digest
        logger.warning(f"{checkpoint_id}: coordinate cache settings/input changed; refitting")

    logger.info(f"{checkpoint_id}: fitting t-SNE on {len(X)} x {X.shape[1]} latents")
    emb = project_tsne(X, perplexity=perplexity, seed=seed, max_iter=max_iter)
    COORD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embedding=emb,
        input_sha256=np.array(digest),
        max_iter=np.array(max_iter),
        perplexity=np.array(perplexity),
        seed=np.array(seed),
        metric=np.array("cosine"),
    )
    logger.info(f"{checkpoint_id}: wrote reusable coordinates -> {path}")
    return emb, path, digest


def class_indices(aa: np.ndarray, view: str) -> np.ndarray:
    if view == "aa20":
        return aa.astype(np.int64)
    letters = np.asarray(AA_LETTERS)[aa]
    spec = VIEW_SPECS[view]
    result = np.full(len(letters), len(spec["labels"]) - 1, dtype=np.int64)
    for group_idx, members in enumerate(spec["sets"]):
        result[np.isin(letters, list(members))] = group_idx
    return result


def view_labels_colors(view: str) -> Tuple[List[str], List[str]]:
    if view == "aa20":
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20")
        return AA_THREE, [cmap(i / 20) for i in range(20)]
    spec = VIEW_SPECS[view]
    return list(spec["labels"]), list(spec["colors"])


def annotate_aa_locations(ax, embedding: np.ndarray, aa: np.ndarray, colors: Sequence):
    """Place each 3-letter AA label on a real point nearest its class median.

    A real representative point is preferable to the raw centroid, which can
    fall in an empty region when a class is split into several t-SNE islands.
    """
    import matplotlib.patheffects as path_effects

    for aa_idx, name in enumerate(AA_THREE):
        member_idx = np.flatnonzero(aa == aa_idx)
        if member_idx.size == 0:
            continue
        member_xy = embedding[member_idx]
        median = np.median(member_xy, axis=0)
        representative = member_idx[np.argmin(np.sum((member_xy - median) ** 2, axis=1))]
        text = ax.text(
            embedding[representative, 0],
            embedding[representative, 1],
            name,
            color=colors[aa_idx],
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])


def scatter_panel(ax, embedding: np.ndarray, aa: np.ndarray, view: str, annotate_aa: bool = False):
    labels, colors = view_labels_colors(view)
    groups = class_indices(aa, view)
    for group_idx, (label, color) in enumerate(zip(labels, colors)):
        idx = groups == group_idx
        ax.scatter(
            embedding[idx, 0], embedding[idx, 1], s=2.2, alpha=0.55,
            linewidths=0, c=[color], label=label, rasterized=True,
        )
    if annotate_aa and view == "aa20":
        annotate_aa_locations(ax, embedding, aa, colors)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_grid(
    embeddings: Mapping[str, np.ndarray],
    checkpoint_ids: Sequence[str],
    aa: np.ndarray,
    purity: Mapping[str, float],
    view: str,
    out_path: pathlib.Path,
    grid_ncols=None,
    show_suptitle: bool = True,
):
    import matplotlib.pyplot as plt

    ncols = min(grid_ncols or 5, len(checkpoint_ids))
    nrows = math.ceil(len(checkpoint_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 3.65 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, checkpoint_id in zip(flat_axes, checkpoint_ids):
        scatter_panel(ax, embeddings[checkpoint_id], aa, view=view, annotate_aa=False)
        panel_title = SHORT_LABEL.get(checkpoint_id, checkpoint_id)
        if view == "aa20" and checkpoint_id in purity:
            panel_title += f"\nlatent-space AA purity@10={purity[checkpoint_id]:.3f}"
        ax.set_title(panel_title, fontsize=9)
    for ax in flat_axes[len(checkpoint_ids):]:
        ax.axis("off")

    labels, colors = view_labels_colors(view)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=6, label=label)
        for label, color in zip(labels, colors)
    ]
    legend_cols = 10 if view == "aa20" else len(labels)
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols, fontsize=8, frameon=False)
    title = {
        "aa20": "t-SNE of per-residue latents — amino-acid identity",
        "charge": "Same t-SNE coordinates — residue charge",
        "aromatic": "Same t-SNE coordinates — aromatic residues",
        "size": "Same t-SNE coordinates — small versus larger side chains",
    }[view]
    if show_suptitle:
        fig.suptitle(title, fontsize=13)
    bottom = 0.075 if view == "aa20" else 0.06
    top = 0.95 if show_suptitle else 1.0
    fig.tight_layout(rect=[0, bottom, 1, top])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def render_individual_aa_panels(
    embeddings: Mapping[str, np.ndarray],
    checkpoint_ids: Sequence[str],
    aa: np.ndarray,
    purity: Mapping[str, float],
):
    import matplotlib.pyplot as plt

    out_dir = FIG_DIR / "tsne_individual"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, colors = view_labels_colors("aa20")
    for checkpoint_id in checkpoint_ids:
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        scatter_panel(ax, embeddings[checkpoint_id], aa, view="aa20", annotate_aa=True)
        purity_line = (
            f"; latent-space AA purity@10 = {purity[checkpoint_id]:.3f}"
            if checkpoint_id in purity else ""
        )
        ax.set_title(
            f"{SHORT_LABEL.get(checkpoint_id, checkpoint_id)} residue latents\n"
            f"t-SNE (cosine){purity_line}",
            fontsize=12,
        )
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5, label=label)
            for label, color in zip(labels, colors)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, frameon=False)
        fig.tight_layout(rect=[0, 0.075, 1, 1])
        path = out_dir / f"{checkpoint_id}_aa20_labeled.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {path}")


def run(
    checkpoint_ids: Sequence[str],
    n_points: int,
    perplexity: float,
    seed: int,
    max_iter: int,
    force: bool,
    output_tag: str,
    grid_ncols=None,
    include_purity: bool = True,
    show_suptitle: bool = True,
):
    unknown = [c for c in checkpoint_ids if c not in ALL_CHECKPOINTS]
    if unknown:
        raise ValueError(f"Unknown checkpoint ids: {unknown}; choices={list(ALL_CHECKPOINTS)}")

    labels_by_protein = load_labels()
    per_protein = {c: load_encoder_latents(c) for c in checkpoint_ids}
    protein_ids = common_protein_ids(per_protein)
    selection = CombinedSelection(protein_ids, labels_by_protein)
    sampled = sample_indices(selection.n, n_points=n_points, seed=seed)
    aa = selection.aa[sampled]
    logger.info(
        f"Shared protein count={len(protein_ids)}; AA+SS-valid residue pool={selection.n}; "
        f"matched plotted residues={len(sampled)}; seed={seed}"
    )

    embeddings: Dict[str, np.ndarray] = {}
    manifest_rows = []
    for checkpoint_id in checkpoint_ids:
        X = selection.build_X(per_protein[checkpoint_id])[sampled]
        embedding, path, digest = load_or_fit(
            checkpoint_id, X, perplexity=perplexity, seed=seed,
            max_iter=max_iter, force=force,
        )
        embeddings[checkpoint_id] = embedding
        manifest_rows.append({
            "checkpoint_id": checkpoint_id,
            "latent_cache": str(LINEAR_PROBING_DIR / "latents" / f"{checkpoint_id}.pt"),
            "latent_dim": int(X.shape[1]),
            "n_points": int(X.shape[0]),
            "input_sha256": digest,
            "coordinate_cache": str(path),
        })

    purity = load_aa_purity_reference(k=10) if include_purity else {}
    file_tag = f"_{output_tag}" if output_tag else ""
    for view in ("aa20", "charge", "aromatic", "size"):
        render_grid(
            embeddings, checkpoint_ids, aa, purity, view,
            FIG_DIR / f"tsne_{view}{file_tag}_grid.png",
            grid_ncols=grid_ncols,
            show_suptitle=show_suptitle,
        )
    render_individual_aa_panels(embeddings, checkpoint_ids, aa, purity)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv = RESULTS_DIR / f"tsne_panel_metrics{file_tag}.csv"
    manifest_json = RESULTS_DIR / f"tsne_manifest{file_tag}.json"
    with open(metrics_csv, "w", newline="") as f:
        fieldnames = ["encoder", "aa_purity_k10_original_8d", "n_points", "seed", "perplexity", "max_iter", "metric"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for checkpoint_id in checkpoint_ids:
            writer.writerow({
                "encoder": checkpoint_id,
                "aa_purity_k10_original_8d": purity.get(checkpoint_id, ""),
                "n_points": len(sampled),
                "seed": seed,
                "perplexity": perplexity,
                "max_iter": max_iter,
                "metric": "cosine",
            })

    manifest = {
        "description": "t-SNE of cached frozen full-context per-residue mean latents",
        "checkpoint_note": (
            "AE caches use the saved EMA encoders; JEPA/LeJEPA caches use online/student encoders. "
            "The exact checkpoint identities and latent dimensions are listed per encoder below."
        ),
        "selection": {
            "shared_proteins": len(protein_ids),
            "aa_and_ss_valid_residue_pool": selection.n,
            "sampled_residues": len(sampled),
            "same_physical_residues_for_every_encoder": True,
            "seed": seed,
        },
        "tsne": {
            "metric": "cosine",
            "perplexity": perplexity,
            "max_iter": max_iter,
            "learning_rate": "auto",
            "early_exaggeration": 12.0,
            "init": "random",
            "method": "barnes_hut",
        },
        "views": {
            "aa20": AA_THREE,
            **{name: spec["labels"] for name, spec in VIEW_SPECS.items()},
        },
        "encoders": manifest_rows,
        "interpretation_warning": (
            "t-SNE axes, orientation, scale, inter-cluster distances, and island shapes are not directly comparable "
            "between independently fitted encoder panels. Use original-8D k-NN purity/probes as quantitative evidence."
        ),
    }
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote {metrics_csv} and {manifest_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=MAIN_CHECKPOINTS)
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Ignore matching coordinate caches and refit t-SNE")
    parser.add_argument(
        "--output-tag", default="",
        help="Optional filename suffix for a comparison set, e.g. all-cached",
    )
    parser.add_argument("--device", default="cpu", help="Kept for CLI clarity; sklearn t-SNE runs on CPU")
    parser.add_argument("--grid-ncols", type=int, default=None)
    parser.add_argument("--hide-purity", action="store_true")
    args = parser.parse_args()
    run(
        checkpoint_ids=args.checkpoints,
        n_points=args.n_points,
        perplexity=args.perplexity,
        seed=args.seed,
        max_iter=args.max_iter,
        force=args.force,
        output_tag=args.output_tag,
        grid_ncols=args.grid_ncols,
        include_purity=not args.hide_purity,
    )
