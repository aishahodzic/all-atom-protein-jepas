"""Follow-up diagnostic: distinguishes three hypotheses for the "worm"
pattern seen in LeJEPA's UMAP panels (thin curly strands, very different
from the round blobs in CA-cond/Fully-latent/AE1-external/Random-init):

  1. Each worm = one protein               -> per-protein threading
  2. Worms show smooth internal gradients   -> sequence-position threading
  3. Neither (worms look colorless-random)  -> UMAP isotropy artifact, not real structure

Same UMAP embeddings as run.py's grids -- same seed=42, same n_points=15000,
same subsample. umap-learn with n_jobs=1 + a fixed random_state is
deterministic, so recomputing here reproduces the exact same 2D layout
run.py already showed; only the point COLORS are new:
  - protein ID: ~20 randomly sampled proteins get distinct colors, every
    other point is neutral gray -- tests hypothesis 1 directly (do a
    highlighted protein's residues, wherever they land, fall in one worm?)
  - sequence position: each residue's index within its own protein,
    normalized 0..1, continuous colormap -- tests hypothesis 2 (do worms
    show a smooth gradient along their length, i.e. is adjacency in
    sequence preserved as adjacency in the embedding?)

Usage: python worm_test.py [--n-points 15000] [--device cuda] [--seed 42]
"""

import argparse
import math
import pathlib
import sys

import numpy as np
import torch
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "knn_purity"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from checkpoints import ALL_CHECKPOINTS  # noqa: E402
from viz import (  # noqa: E402
    CombinedSelection,
    DEFAULT_N_POINTS,
    SEED,
    SHORT_LABEL,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    common_protein_ids,
    load_encoder_latents,
    load_labels,
    project_umap,
    sample_indices,
)

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"
N_HIGHLIGHT_PROTEINS = 20


def build_local_idx_flat(selection: CombinedSelection) -> np.ndarray:
    """Local residue index (within its own protein's full residue array)
    for each of selection.n rows, in the same flattened order as
    selection.aa/ss/protein_idx -- reconstructed from the same (pid,
    idx_array) keys CombinedSelection built its label arrays from."""
    return np.concatenate([idx for _, idx in selection.keys])


def build_protein_lengths(protein_ids, per_protein_any_checkpoint) -> dict:
    """{p_int -> total residue count for that protein}, p_int matching
    CombinedSelection's own enumerate(protein_ids) indexing exactly (same
    protein_ids list, same order) -- total residue count (not just
    aa+ss-valid residues) is what "position in the full sequence" means."""
    lengths = {}
    for p_int, pid in enumerate(protein_ids):
        z = per_protein_any_checkpoint.get(pid)
        if z is not None:
            lengths[p_int] = z.shape[0]
    return lengths


def protein_id_colors(protein_idx_sub: np.ndarray, seed: int = SEED):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    unique_proteins = np.unique(protein_idx_sub)
    n_highlight = min(N_HIGHLIGHT_PROTEINS, len(unique_proteins))
    highlighted = rng.choice(unique_proteins, size=n_highlight, replace=False)
    cmap = plt.get_cmap("tab20")
    color_of = {p: cmap(i / n_highlight) for i, p in enumerate(highlighted)}
    gray = (0.85, 0.85, 0.85, 0.5)
    colors = np.array([color_of.get(p, gray) for p in protein_idx_sub])
    return colors, highlighted


def plot_categorical_grid(embeddings, color_arrays, checkpoint_ids, title, out_path, ncols=5):
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(checkpoint_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, checkpoint_id in zip(axes, checkpoint_ids):
        emb = embeddings[checkpoint_id]
        ax.scatter(emb[:, 0], emb[:, 1], c=color_arrays[checkpoint_id], s=6, alpha=1.0, linewidths=0)
        ax.set_title(SHORT_LABEL.get(checkpoint_id, checkpoint_id), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(checkpoint_ids):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_continuous_grid(embeddings, value_arrays, checkpoint_ids, title, out_path, ncols=5):
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(checkpoint_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    sc = None
    for ax, checkpoint_id in zip(axes, checkpoint_ids):
        emb = embeddings[checkpoint_id]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=value_arrays[checkpoint_id], cmap="viridis",
                         vmin=0, vmax=1, s=4, alpha=0.7, linewidths=0)
        ax.set_title(SHORT_LABEL.get(checkpoint_id, checkpoint_id), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(checkpoint_ids):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="residue position within protein (0=N-term, 1=C-term)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def run_protein_dense_test(per_protein_by_checkpoint, protein_ids, labels_by_protein,
                            checkpoint_ids, n_proteins: int = 25, device: str = "cuda", seed: int = SEED):
    """The main worm_test subsample (uniform over ~750k residues) gives each
    protein only ~3 points on average (max 12 anywhere) -- far too sparse to
    tell whether a cluster is "one protein" no matter how it's colored (a
    500-point blob would need 150+ different proteins to explain it at that
    density). This is a DELIBERATELY DIFFERENT, protein-dense subsample built
    just for hypothesis 1: pick a handful of proteins and keep ALL of their
    aa+ss-valid residues (not a uniform residue subsample), so each protein
    actually has enough points to see as a shape. Not comparable point-for-
    point with run.py's or run_all()'s embeddings above -- a fresh UMAP fit,
    same params/seed, different input set.
    """
    selection = CombinedSelection(protein_ids, labels_by_protein)
    rng = np.random.default_rng(seed)
    protein_counts = {}
    for pid, idx in selection.keys:
        protein_counts[pid] = idx.size
    eligible = [pid for pid, n in protein_counts.items() if n >= 30]  # skip tiny fragments
    chosen = rng.choice(eligible, size=min(n_proteins, len(eligible)), replace=False)
    chosen_set = set(chosen)

    keep_mask = np.zeros(selection.n, dtype=bool)
    pos = 0
    for pid, idx in selection.keys:
        n = idx.size
        if pid in chosen_set:
            keep_mask[pos:pos + n] = True
        pos += n
    keep_idx = np.nonzero(keep_mask)[0]
    rng.shuffle(keep_idx)
    protein_idx_dense = selection.protein_idx[keep_idx]
    local_idx_dense = build_local_idx_flat(selection)[keep_idx]
    protein_length = build_protein_lengths(protein_ids, per_protein_by_checkpoint[checkpoint_ids[0]])
    seq_pos_dense = np.array([
        local_idx_dense[i] / max(protein_length[protein_idx_dense[i]] - 1, 1)
        for i in range(len(keep_idx))
    ])
    logger.info(f"Protein-dense subsample: {len(keep_idx)} residues from {len(chosen)} proteins "
                f"(min 30 residues each, mean {len(keep_idx) / len(chosen):.0f} pts/protein)")

    import matplotlib.pyplot as plt
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    protein_int_to_color = {p: palette[i % len(palette)] for i, p in enumerate(np.unique(protein_idx_dense))}

    umap_embeddings = {}
    for checkpoint_id in checkpoint_ids:
        X_full = selection.build_X(per_protein_by_checkpoint[checkpoint_id])
        X_dense = X_full[keep_idx]
        umap_embeddings[checkpoint_id] = project_umap(X_dense, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, seed=seed)
        logger.info(f"{checkpoint_id:22s} protein-dense UMAP done")

    color_dict = {c: np.array([protein_int_to_color[p] for p in protein_idx_dense]) for c in checkpoint_ids}
    seq_pos_dict = {c: seq_pos_dense for c in checkpoint_ids}

    plot_categorical_grid(
        umap_embeddings, color_dict, checkpoint_ids,
        f"UMAP on a PROTEIN-DENSE subsample ({len(chosen)} proteins, all their residues, "
        f"{len(keep_idx)} points total -- NOT the same subsample as the other grids) colored by protein ID "
        f"-- tests 'is each worm one protein?' properly powered this time",
        FIG_DIR / "umap_protein_dense_grid.png",
    )
    plot_continuous_grid(
        umap_embeddings, seq_pos_dict, checkpoint_ids,
        f"Same protein-dense subsample, colored by sequence position within protein",
        FIG_DIR / "umap_protein_dense_seq_position_grid.png",
    )


def run_all(n_points: int = DEFAULT_N_POINTS, device: str = None, seed: int = SEED):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    checkpoint_ids = list(ALL_CHECKPOINTS)
    labels_by_protein = load_labels()
    per_protein_by_checkpoint = {c: load_encoder_latents(c) for c in checkpoint_ids}
    protein_ids = common_protein_ids(per_protein_by_checkpoint)

    selection = CombinedSelection(protein_ids, labels_by_protein)
    subsample_idx = sample_indices(selection.n, n_points=n_points, seed=seed)
    protein_idx_sub = selection.protein_idx[subsample_idx]
    local_idx_sub = build_local_idx_flat(selection)[subsample_idx]
    protein_length = build_protein_lengths(protein_ids, per_protein_by_checkpoint[checkpoint_ids[0]])
    seq_pos_sub = np.array([
        local_idx_sub[i] / max(protein_length[protein_idx_sub[i]] - 1, 1)
        for i in range(len(subsample_idx))
    ])
    logger.info(f"Subsample: {len(subsample_idx)} residues from {len(np.unique(protein_idx_sub))} distinct proteins "
                f"(seed={seed}, same subsample as run.py's grids)")

    pid_colors, highlighted_proteins = protein_id_colors(protein_idx_sub, seed=seed)
    logger.info(f"Highlighted {len(highlighted_proteins)} proteins by distinct color; all others gray")

    umap_embeddings = {}
    for checkpoint_id in checkpoint_ids:
        X_full = selection.build_X(per_protein_by_checkpoint[checkpoint_id])
        X_sub = X_full[subsample_idx]
        umap_embeddings[checkpoint_id] = project_umap(X_sub, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, seed=seed)
        logger.info(f"{checkpoint_id:22s} UMAP done")

    pid_color_dict = {c: pid_colors for c in checkpoint_ids}
    seq_pos_dict = {c: seq_pos_sub for c in checkpoint_ids}

    plot_categorical_grid(
        umap_embeddings, pid_color_dict, checkpoint_ids,
        f"UMAP (cosine, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by protein ID "
        f"({len(highlighted_proteins)} proteins highlighted, rest gray) -- tests 'is each worm one protein?'",
        FIG_DIR / "umap_protein_id_grid.png",
    )
    plot_continuous_grid(
        umap_embeddings, seq_pos_dict, checkpoint_ids,
        f"UMAP (cosine, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by sequence position "
        f"within protein -- tests 'are worms sequence-contiguous threads?'",
        FIG_DIR / "umap_seq_position_grid.png",
    )

    print_worm_verdict(checkpoint_ids, umap_embeddings, protein_idx_sub, seq_pos_sub)

    run_protein_dense_test(per_protein_by_checkpoint, protein_ids, labels_by_protein,
                            checkpoint_ids, n_proteins=25, device=device, seed=seed)


def print_worm_verdict(checkpoint_ids, umap_embeddings, protein_idx_sub, seq_pos_sub):
    """Quantitative companion to the visual grids: for each encoder, mean
    within-protein pairwise embedding distance (small -> protein clumping)
    and Spearman correlation between embedding position along the local
    UMAP-1 axis and sequence position (large |r| within same-protein
    clusters -> sequence threading). Kept simple/approximate -- the grids
    are the primary evidence, this is just a number to anchor "worm or not"
    without eyeballing."""
    from scipy.stats import spearmanr

    lines = ["", "=" * 78, "WORM TEST -- quantitative companion to the two grids", "=" * 78]
    for checkpoint_id in checkpoint_ids:
        emb = umap_embeddings[checkpoint_id]
        proteins, counts = np.unique(protein_idx_sub, return_counts=True)
        keep = proteins[counts >= 5]
        if len(keep) == 0:
            continue
        centered = np.zeros_like(emb)
        mask = np.isin(protein_idx_sub, keep)
        for p in keep:
            sel = protein_idx_sub == p
            centered[sel] = emb[sel] - emb[sel].mean(axis=0)
        r0, _ = spearmanr(centered[mask, 0], seq_pos_sub[mask])
        r1, _ = spearmanr(centered[mask, 1], seq_pos_sub[mask])
        threadiness = max(abs(r0), abs(r1))
        lines.append(f"{checkpoint_id:22s} within-protein seq-position vs UMAP-axis |Spearman r| = {threadiness:.3f}"
                     f"  (axis0={r0:+.3f}, axis1={r1:+.3f})")
    lines.append("Read: near 0 -> no sequence-gradient signal within a protein's own points (rules out hypothesis "
                 "2 for that encoder); large |r| (roughly >0.3) -> within-protein points ARE ordered by sequence "
                 "position along some direction in the embedding -- consistent with sequence-contiguous threading. "
                 "Combine with the two grids above, not in isolation.")
    lines.append("=" * 78)
    for line in lines:
        logger.info(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_all(n_points=args.n_points, device=args.device, seed=args.seed)
