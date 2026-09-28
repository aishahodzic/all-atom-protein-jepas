"""Audit what the matched 768D block-12 spaces organize around."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from Evaluation.latent_viz.label_tsne_worms import local_index_array
from Evaluation.latent_viz.run_layerwise_tsne import (
    N_POINTS,
    SEED,
    load_selection,
    sample_sha256,
)
from Evaluation.latent_viz.run_tsne import scatter_panel, view_labels_colors
from Evaluation.latent_viz.analyze_current_four_health import aligned_property


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "Evaluation/latent_viz/results/layerwise_tsne"
RESULT = ROOT / "block12-organization-metrics.csv"
MANIFEST = ROOT / "block12-organization-manifest.json"
FIGURE = REPO / "iclr2027/figures/block12-organization-uncropped-lejepa-vs-jepa-rand70.png"
K = 10

SPECS = {
    "uncropped-le-jepa": {
        "label": "Uncropped LeJEPA",
        "sample": ROOT / "layerwise-tsne-uncropped-le-jepa-sampled-768d.pt",
        "coordinates": ROOT / "layerwise-tsne-uncropped-le-jepa-block12-n15000-cosine-tsne.npz",
    },
    "jepa-rand70": {
        "label": "JEPA random-mask 70%",
        "sample": ROOT / "layerwise-tsne-jepa-rand70-sampled-768d.pt",
        "coordinates": ROOT / "layerwise-tsne-jepa-rand70-block12-n15000-cosine-tsne.npz",
    },
}


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def cross_protein_neighbours(
    values: np.ndarray, protein_index: np.ndarray, *, cosine: bool
) -> np.ndarray:
    tree_values = normalize(values) if cosine else np.asarray(values, dtype=np.float32)
    _, candidates = cKDTree(tree_values).query(tree_values, k=64, workers=-1)
    result = np.empty((len(values), K), dtype=np.int64)
    for row, candidate_row in enumerate(candidates):
        external = candidate_row[protein_index[candidate_row] != protein_index[row]]
        if len(external) < K:
            raise RuntimeError(f"insufficient cross-protein neighbours for row {row}")
        result[row] = external[:K]
    return result


def same_protein_fraction(values: np.ndarray, protein_index: np.ndarray) -> float:
    normalized_values = normalize(values)
    _, neighbours = cKDTree(normalized_values).query(
        normalized_values, k=K + 1, workers=-1
    )
    return float((protein_index[neighbours[:, 1:]] == protein_index[:, None]).mean())


def purity(labels: np.ndarray, neighbours: np.ndarray) -> float:
    return float((labels[neighbours] == labels[:, None]).mean())


def chance_purity(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    frequencies = counts / counts.sum()
    return float(np.square(frequencies).sum())


def neighbour_spearman(
    target: np.ndarray, valid: np.ndarray, neighbours: np.ndarray
) -> tuple[float, int]:
    usable = valid & np.isfinite(target) & np.all(valid[neighbours], axis=1)
    prediction = target[neighbours[usable]].mean(axis=1)
    statistic = spearmanr(target[usable], prediction).statistic
    return float(statistic), int(usable.sum())


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def hide_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def continuous_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    embedding: np.ndarray,
    color: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plot = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=2.2,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=8.5)
    hide_axes(ax)
    colorbar = fig.colorbar(plot, ax=ax, fraction=0.047, pad=0.015)
    colorbar.set_label(colorbar_label, fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)


def main() -> None:
    selection, sampled, _ = load_selection()
    labels = torch.load(
        REPO / "Evaluation/linear_probing/latents/labels.pt",
        map_location="cpu",
        weights_only=False,
    )
    selection_digest = sample_sha256(selection, sampled)
    aa = selection.aa[sampled].astype(np.int64, copy=False)
    ss = selection.ss[sampled].astype(np.int64, copy=False)
    protein_index = selection.protein_idx[sampled].astype(np.int64, copy=False)
    residue_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate(
        [
            np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    normalized_position = (residue_index + 1.0) / lengths
    log_length = np.log(lengths)
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    bfactor, bfactor_valid = aligned_property(
        selection, labels, "bfactor_z", "bfactor_valid"
    )
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]
    bfactor, bfactor_valid = bfactor[sampled], bfactor_valid[sampled]
    all_valid = np.ones(N_POINTS, dtype=bool)

    fig, axes = plt.subplots(len(SPECS), 6, figsize=(20.0, 7.4), squeeze=False)
    rows: list[dict[str, object]] = []
    artifact_records = []
    for row_index, (model, spec) in enumerate(SPECS.items()):
        cached = torch.load(spec["sample"], map_location="cpu", weights_only=False)
        if cached["selection_sha256"] != selection_digest:
            raise AssertionError(f"sample mismatch for {model}")
        latent = cached["states"][12].float().numpy()
        coordinates = np.load(spec["coordinates"], allow_pickle=False)
        embedding = coordinates["embedding"].astype(np.float32, copy=False)
        if latent.shape != (N_POINTS, 768) or embedding.shape != (N_POINTS, 2):
            raise AssertionError(f"shape mismatch for {model}: {latent.shape}, {embedding.shape}")

        latent_neighbours = cross_protein_neighbours(
            latent, protein_index, cosine=True
        )
        tsne_neighbours = cross_protein_neighbours(
            embedding, protein_index, cosine=False
        )
        metrics: dict[str, object] = {
            "model": model,
            "label": spec["label"],
            "block": 12,
            "dimension": 768,
            "sampled_residues": N_POINTS,
            "sample_sha256": selection_digest,
            "aa_random_pair_purity": chance_purity(aa),
            "ss_random_pair_purity": chance_purity(ss),
            "latent_aa_purity_at_10_cross_protein": purity(aa, latent_neighbours),
            "latent_ss_purity_at_10_cross_protein": purity(ss, latent_neighbours),
            "latent_same_protein_fraction_at_10": same_protein_fraction(
                latent, protein_index
            ),
            "tsne_aa_purity_at_10_cross_protein": purity(aa, tsne_neighbours),
            "tsne_ss_purity_at_10_cross_protein": purity(ss, tsne_neighbours),
        }
        continuous = {
            "rsa": (rsa, rsa_valid),
            "bfactor": (bfactor, bfactor_valid),
            "absolute_position": (residue_index, all_valid),
            "normalized_position": (normalized_position, all_valid),
            "log_chain_length": (log_length, all_valid),
        }
        for name, (target, valid) in continuous.items():
            latent_rho, latent_n = neighbour_spearman(
                target, valid, latent_neighbours
            )
            tsne_rho, tsne_n = neighbour_spearman(target, valid, tsne_neighbours)
            metrics[f"latent_{name}_knn10_spearman"] = latent_rho
            metrics[f"latent_{name}_n"] = latent_n
            metrics[f"tsne_{name}_knn10_spearman"] = tsne_rho
            metrics[f"tsne_{name}_n"] = tsne_n
        rows.append(metrics)

        scatter_panel(axes[row_index, 0], embedding, aa, view="aa20")
        axes[row_index, 0].set_title(
            "Amino-acid identity\n"
            f"768D cross-protein purity@10={metrics['latent_aa_purity_at_10_cross_protein']:.3f}",
            fontsize=8.5,
        )
        ss_colors = np.asarray(["#d62728", "#1f77b4", "#7f7f7f"])
        for state, name in enumerate(("helix", "sheet", "loop")):
            mask = ss == state
            axes[row_index, 1].scatter(
                embedding[mask, 0], embedding[mask, 1], s=2.2, alpha=0.6,
                c=[ss_colors[state]], linewidths=0, rasterized=True, label=name,
            )
        axes[row_index, 1].set_title(
            "Secondary structure\n"
            f"768D cross-protein purity@10={metrics['latent_ss_purity_at_10_cross_protein']:.3f}",
            fontsize=8.5,
        )
        hide_axes(axes[row_index, 1])
        continuous_panel(
            fig, axes[row_index, 2], embedding, rsa,
            f"Relative solvent accessibility\n768D kNN rho={metrics['latent_rsa_knn10_spearman']:.3f}",
            "RSA", "viridis", 0.0, 1.0,
        )
        continuous_panel(
            fig, axes[row_index, 3], embedding, residue_index + 1.0,
            f"Absolute residue position\n768D kNN rho={metrics['latent_absolute_position_knn10_spearman']:.3f}",
            "one-based index i", "turbo", 1.0, 256.0,
        )
        continuous_panel(
            fig, axes[row_index, 4], embedding, normalized_position,
            f"Normalized position\n768D kNN rho={metrics['latent_normalized_position_knn10_spearman']:.3f}",
            "i/L", "viridis", 0.0, 1.0,
        )
        continuous_panel(
            fig, axes[row_index, 5], embedding, lengths,
            f"Chain length\n768D kNN rho={metrics['latent_log_chain_length_knn10_spearman']:.3f}",
            "chain length L", "magma", 50.0, 256.0,
        )
        axes[row_index, 0].set_ylabel(
            f"{spec['label']}\nsame-protein@10={metrics['latent_same_protein_fraction_at_10']:.3f}",
            fontsize=9.5,
        )
        artifact_records.append(
            {
                "model": model,
                "sample_cache": str(spec["sample"]),
                "coordinate_cache": str(spec["coordinates"]),
                "checkpoint_sha256": cached["checkpoint_sha256"],
            }
        )

    aa_labels, aa_colors = view_labels_colors("aa20")
    aa_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=4, label=label)
        for label, color in zip(aa_labels, aa_colors)
    ]
    fig.legend(
        handles=aa_handles, loc="lower center", ncol=20, fontsize=6.5,
        frameon=False, bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "What organizes block-12 768D residue representations?\n"
        "Identical 15,000 residues; panel coordinates are the precomputed cosine t-SNEs",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.93])
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    temporary_figure = FIGURE.with_name(f".{FIGURE.name}.tmp-{os.getpid()}.png")
    fig.savefig(temporary_figure, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    os.replace(temporary_figure, FIGURE)

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary_result = RESULT.with_name(f".{RESULT.name}.tmp-{os.getpid()}")
    with temporary_result.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_result, RESULT)
    atomic_json(
        {
            "schema_version": 1,
            "artifact": "matched_block12_organization_audit",
            "sampled_residues": N_POINTS,
            "sample_sha256": selection_digest,
            "latent_neighbours": "exact cosine kNN; same-protein candidates excluded",
            "tsne_neighbours": "exact Euclidean kNN in plotted coordinates; same-protein candidates excluded",
            "k": K,
            "artifacts": artifact_records,
            "result": str(RESULT),
            "figure": str(FIGURE),
        },
        MANIFEST,
    )
    print(f"wrote {RESULT}")
    print(f"wrote {MANIFEST}")
    print(f"wrote {FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
