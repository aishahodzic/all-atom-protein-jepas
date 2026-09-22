"""Orchestrates the latent visualization: for every cached encoder (8
in-house + ae1-external + random-init -- same 10 as knn_purity), project the
same ~15k-residue subsample (PCA + UMAP), render two grid figures per
projection (colored by AA-group, colored by SS) with each panel annotated by
its k-NN purity number, and write a caption CSV.

UMAP is the primary/thesis-ready output (see viz.py: cosine metric, matches
knn_purity's distance metric). PCA is kept for the variance-explained
numbers and as an honest "here's what a linear projection shows" reference
-- 8-dim PCA is expected to be a fairly uninformative blob.

Usage: python run.py [--n-points 15000] [--device cuda] [--seed 42]
"""

import argparse
import csv
import math
import pathlib
import sys

import numpy as np
import torch
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "knn_purity"))

from checkpoints import ALL_CHECKPOINTS  # noqa: E402
from viz import (  # noqa: E402
    AA_GROUP,
    AA_GROUP_COLORS,
    AA_GROUP_NAMES,
    AA_LETTERS,
    CombinedSelection,
    DEFAULT_N_POINTS,
    SEED,
    SHORT_LABEL,
    SS_COLORS,
    SS_STATE_NAMES,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    common_protein_ids,
    embedding_purity,
    load_encoder_latents,
    load_labels,
    project_pca,
    project_umap,
    sample_indices,
)

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CAPTIONS_CSV = RESULTS_DIR / "panel_captions.csv"
PURITY_CSV = pathlib.Path(__file__).resolve().parents[1] / "knn_purity" / "results" / "knn_purity_results.csv"

ARTIFACT_THRESHOLD = 0.15  # |8-dim-on-subsample purity - projected purity| above this -> flag as possible artifact


def load_purity_reference(k: int = 10):
    """{(encoder, property): purity} at k=10 from the official 20k-query /
    full-pool knn_purity sweep -- what panels are annotated with."""
    ref = {}
    with open(PURITY_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["k"]) == k:
                ref[(row["encoder"], row["property"])] = float(row["purity"])
    return ref


def aa_group_colors(aa_indices: np.ndarray) -> np.ndarray:
    groups = AA_GROUP[aa_indices]
    return np.array([AA_GROUP_COLORS[g] for g in groups])


def aa_full_colors(aa_indices: np.ndarray, cmap) -> np.ndarray:
    return np.array([cmap(i / 20) for i in aa_indices])


def ss_colors(ss_indices: np.ndarray) -> np.ndarray:
    return np.array([SS_COLORS[s] for s in ss_indices])


def plot_grid(embeddings, color_arrays, purity_lookup, checkpoint_order, prop_label, k,
              legend_swatches, title, out_path, ncols=5):
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(checkpoint_order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, checkpoint_id in zip(axes, checkpoint_order):
        emb = embeddings[checkpoint_id]
        c = color_arrays[checkpoint_id]
        ax.scatter(emb[:, 0], emb[:, 1], c=c, s=3, alpha=0.5, linewidths=0)
        purity = purity_lookup.get((checkpoint_id, prop_label))
        purity_str = f"{purity:.3f}" if purity is not None else "n/a"
        ax.set_title(f"{SHORT_LABEL.get(checkpoint_id, checkpoint_id)}\n{prop_label.upper()} purity@{k}={purity_str}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(checkpoint_order):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in legend_swatches
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(legend_swatches), 6), bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def run_all(n_points: int = DEFAULT_N_POINTS, device: str = None, seed: int = SEED):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    checkpoint_ids = list(ALL_CHECKPOINTS)
    labels_by_protein = load_labels()
    per_protein_by_checkpoint = {c: load_encoder_latents(c) for c in checkpoint_ids}
    protein_ids = common_protein_ids(per_protein_by_checkpoint)

    selection = CombinedSelection(protein_ids, labels_by_protein)
    subsample_idx = sample_indices(selection.n, n_points=n_points, seed=seed)
    n_sub = len(subsample_idx)
    y_aa = selection.aa[subsample_idx]
    y_ss = selection.ss[subsample_idx]
    protein_idx_sub = selection.protein_idx[subsample_idx]
    logger.info(f"Residue pool valid for both aa+ss: {selection.n}; plotting subsample: {n_sub} "
                f"(seed={seed}, same indices reused for every encoder)")

    purity_ref = load_purity_reference(k=10)
    aa_full_cmap = __import__("matplotlib.pyplot", fromlist=["get_cmap"]).get_cmap("tab20")

    pca_embeddings, umap_embeddings = {}, {}
    caption_rows = []

    for checkpoint_id in checkpoint_ids:
        X_full = selection.build_X(per_protein_by_checkpoint[checkpoint_id])
        X_sub = X_full[subsample_idx]

        pca_emb, pca_var = project_pca(X_sub, seed=seed)
        umap_emb = project_umap(X_sub, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, seed=seed)
        pca_embeddings[checkpoint_id] = pca_emb
        umap_embeddings[checkpoint_id] = umap_emb

        purity_8dim_sub = {
            "aa": embedding_purity(X_sub, y_aa, protein_idx_sub, k=10, device=device),
            "ss": embedding_purity(X_sub, y_ss, protein_idx_sub, k=10, device=device),
        }
        purity_pca = {
            "aa": embedding_purity(pca_emb, y_aa, protein_idx_sub, k=10, device=device),
            "ss": embedding_purity(pca_emb, y_ss, protein_idx_sub, k=10, device=device),
        }
        purity_umap = {
            "aa": embedding_purity(umap_emb, y_aa, protein_idx_sub, k=10, device=device),
            "ss": embedding_purity(umap_emb, y_ss, protein_idx_sub, k=10, device=device),
        }

        logger.info(
            f"{checkpoint_id:22s} pca_var=({pca_var[0]:.3f},{pca_var[1]:.3f}) "
            f"8dim_sub(aa/ss)=({purity_8dim_sub['aa']:.3f}/{purity_8dim_sub['ss']:.3f}) "
            f"pca(aa/ss)=({purity_pca['aa']:.3f}/{purity_pca['ss']:.3f}) "
            f"umap(aa/ss)=({purity_umap['aa']:.3f}/{purity_umap['ss']:.3f})"
        )

        for prop, p8, ppca, pumap in [("aa", purity_8dim_sub["aa"], purity_pca["aa"], purity_umap["aa"]),
                                       ("ss", purity_8dim_sub["ss"], purity_pca["ss"], purity_umap["ss"])]:
            caption_rows.append(dict(
                encoder=checkpoint_id, property=prop,
                purity_reference_k10=purity_ref.get((checkpoint_id, prop), ""),
                purity_8dim_subsample=p8, purity_pca_2d=ppca, purity_umap_2d=pumap,
                pca_var_pc1=pca_var[0], pca_var_pc2=pca_var[1],
                umap_n_neighbors=UMAP_N_NEIGHBORS, umap_min_dist=UMAP_MIN_DIST,
                n_points=n_sub, seed=seed,
            ))

    # --- color arrays (fixed once, reused across all four grids that need them) ---
    aa_group_c = {c: aa_group_colors(y_aa) for c in checkpoint_ids}  # same coloring, kept per-encoder key for plot_grid's interface
    aa_full_c = {c: aa_full_colors(y_aa, aa_full_cmap) for c in checkpoint_ids}
    ss_c = {c: ss_colors(y_ss) for c in checkpoint_ids}

    ss_legend = [(SS_STATE_NAMES[i], SS_COLORS[i]) for i in range(3)]
    aa_group_legend = [(AA_GROUP_NAMES[i], AA_GROUP_COLORS[i]) for i in range(4)]
    aa_full_legend = [(AA_LETTERS[i], aa_full_cmap(i / 20)) for i in range(20)]

    plot_grid(umap_embeddings, ss_c, purity_ref, checkpoint_ids, "ss", 10, ss_legend,
              f"UMAP (cosine, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by secondary structure",
              FIG_DIR / "umap_ss_grid.png")
    plot_grid(umap_embeddings, aa_group_c, purity_ref, checkpoint_ids, "aa", 10, aa_group_legend,
              f"UMAP (cosine, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by AA group",
              FIG_DIR / "umap_aa_grouped_grid.png")
    plot_grid(umap_embeddings, aa_full_c, purity_ref, checkpoint_ids, "aa", 10, aa_full_legend,
              f"UMAP (cosine, n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) colored by AA identity (20-class)",
              FIG_DIR / "umap_aa_full20_grid.png")
    plot_grid(pca_embeddings, ss_c, purity_ref, checkpoint_ids, "ss", 10, ss_legend,
              "PCA (2D) colored by secondary structure -- secondary/diagnostic, see variance-explained in captions CSV",
              FIG_DIR / "pca_ss_grid.png")
    plot_grid(pca_embeddings, aa_group_c, purity_ref, checkpoint_ids, "aa", 10, aa_group_legend,
              "PCA (2D) colored by AA group -- secondary/diagnostic, see variance-explained in captions CSV",
              FIG_DIR / "pca_aa_grouped_grid.png")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(caption_rows[0].keys())
    with open(CAPTIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(caption_rows)
    logger.info(f"Wrote {len(caption_rows)} caption rows -> {CAPTIONS_CSV}")

    print_interpretation_notes(caption_rows)
    return caption_rows


def print_interpretation_notes(caption_rows):
    by_key = {(r["encoder"], r["property"]): r for r in caption_rows}
    lines = ["", "=" * 78, "INTERPRETATION NOTES -- latent visualization", "=" * 78,
             "Caution: 8-dim -> 2D loses most structure. These figures are illustrative, "
             "not proof -- the k-NN purity / linear-probe numbers are the evidence; the "
             "viz is the intuition. Don't over-interpret exact cluster shapes (UMAP is "
             f"parameter-sensitive: n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}).",
             "Caution (per-protein clumping): visible discrete blobs in a panel are often ONE "
             "PROTEIN each, not property-based clusters -- same-protein residues are usually "
             "each other's nearest neighbours in raw 8-dim space (shared context), and UMAP's "
             "local-neighbourhood preservation reflects that directly. The purity numbers "
             "explicitly EXCLUDE same-protein neighbours (that's the whole point -- see "
             "knn_purity's module docstring), so a tight-looking blob with mixed colors inside "
             "it is not a contradiction: check whether COLOR is homogeneous *within* clumps, "
             "not just whether clumps exist."]

    def g(enc, prop, field):
        r = by_key.get((enc, prop))
        return r[field] if r else None

    lejepa_ss = [g(f"le-jepa-{v}", "ss", "purity_reference_k10") for v in ("mutation", "rand-mask-50", "rand-mask-70")]
    jepa_ss = [g(f"jepa-{v}", "ss", "purity_reference_k10") for v in ("mutation", "rand-mask-50", "rand-mask-70")]
    if all(v is not None for v in lejepa_ss + jepa_ss):
        lines.append(f"Expected-pattern check (SS): LeJEPA {[round(v,3) for v in lejepa_ss]} vs "
                      f"JEPA {[round(v,3) for v in jepa_ss]} -- "
                      f"{'CONFIRMED tighter LeJEPA SS clustering' if min(lejepa_ss) > max(jepa_ss) else 'mixed / not clean-cut'}.")

    ae_aa = [g("ca-cond", "aa", "purity_reference_k10"), g("fully-latent", "aa", "purity_reference_k10")]
    jepa_aa = [g(f"jepa-{v}", "aa", "purity_reference_k10") for v in ("mutation", "rand-mask-50", "rand-mask-70")]
    lejepa_aa = [g(f"le-jepa-{v}", "aa", "purity_reference_k10") for v in ("mutation", "rand-mask-50", "rand-mask-70")]
    if all(v is not None for v in ae_aa + jepa_aa + lejepa_aa):
        lines.append(f"Expected-pattern check (AA): AE {[round(v,3) for v in ae_aa]} (tight) vs "
                      f"JEPA {[round(v,3) for v in jepa_aa]} (diffuse) vs LeJEPA {[round(v,3) for v in lejepa_aa]} "
                      f"(intermediate) -- {'CONFIRMED AE > LeJEPA > JEPA ordering' if min(ae_aa) > max(lejepa_aa) > min(lejepa_aa) and max(lejepa_aa) > max(jepa_aa) else 'see numbers'}.")

    any_artifact = False
    for r in caption_rows:
        gap_umap = r["purity_8dim_subsample"] - r["purity_umap_2d"]
        if abs(gap_umap) > ARTIFACT_THRESHOLD:
            any_artifact = True
            direction = "UMAP UNDER-represents" if gap_umap > 0 else "UMAP OVER-represents"
            lines.append(
                f"[POSSIBLE ARTIFACT] {r['encoder']:22s} {r['property']:3s}: 8-dim(subsample) purity="
                f"{r['purity_8dim_subsample']:.3f} vs UMAP-2D purity={r['purity_umap_2d']:.3f} "
                f"(gap {gap_umap:+.3f}, >{ARTIFACT_THRESHOLD}) -- {direction} the true clustering; "
                f"don't read the panel at face value for this encoder/property."
            )
    if not any_artifact:
        lines.append(f"No panel's UMAP-2D purity diverges from its 8-dim(subsample) purity by more than "
                      f"{ARTIFACT_THRESHOLD} -- the pictures are a fair visual proxy for the metric everywhere checked.")
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
