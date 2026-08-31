"""Fast cross-protein neighbourhood screen for LeJEPA latent organization."""

import csv
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


REPO = pathlib.Path(__file__).resolve().parents[2]
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
VIZ = REPO / "Evaluation/latent_viz"
for path in (LINEAR, KNN, VIZ):
    sys.path.insert(0, str(path))

from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from label_tsne_worms import local_index_array  # noqa: E402
from run_tsne import coordinate_path  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


IDS = (
    "le-jepa-rand-mask-50",
    "le-jepa-rand50-relative-pos-10k",
    "le-jepa-rand50-relative-pos-30k",
)
OUT = REPO / "Evaluation/latent_viz/results/lejepa_relative_position_organization_screen.csv"
FIGURE = REPO / "Evaluation/latent_viz/figures/tsne_organization/lejepa_relative_position_organization_screen.png"
N = 15_000
K = 10


def aligned_property(selection, labels, key, valid_key=None):
    values = []
    valid = []
    for protein_id, indices in selection.keys:
        label = labels[protein_id]
        values.append(np.asarray(label[key])[indices])
        if valid_key is None:
            valid.append(np.ones(len(indices), dtype=bool))
        else:
            valid.append(np.asarray(label[valid_key])[indices])
    return np.concatenate(values), np.concatenate(valid)


def external_neighbours(values, protein_index, k=K):
    values = np.asarray(values, dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    # Fetch a buffer, then discard neighbours from the query's own protein.
    _, candidates = cKDTree(values).query(values, k=64, workers=-1)
    result = np.empty((len(values), k), dtype=np.int64)
    for row, candidate_row in enumerate(candidates):
        external = candidate_row[protein_index[candidate_row] != protein_index[row]]
        if len(external) < k:
            raise RuntimeError("Neighbour buffer was insufficient")
        result[row] = external[:k]
    return result


def classification_purity(y, neighbours):
    return float((y[neighbours] == y[:, None]).mean())


def regression_scores(y, valid, neighbours):
    usable = valid & np.all(valid[neighbours], axis=1) & np.isfinite(y)
    prediction = y[neighbours[usable]].mean(axis=1)
    target = y[usable]
    return float(r2_score(target, prediction)), float(spearmanr(target, prediction).statistic), int(usable.sum())


def main():
    labels = load_labels()
    latents = {checkpoint_id: load_encoder_latents(checkpoint_id) for checkpoint_id in IDS}
    protein_ids = common_protein_ids(latents)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, N, SEED)

    aa = selection.aa[sampled]
    ss = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    absolute_index = local_index_array(selection)[sampled].astype(np.float64)
    lengths = np.concatenate([
        np.full(len(indices), len(labels[protein_id]["aa"]), dtype=np.float64)
        for protein_id, indices in selection.keys
    ])[sampled]
    denominator = np.maximum(lengths - 1.0, 1.0)
    normalized_position = absolute_index / denominator
    terminal_distance = np.minimum(absolute_index, lengths - 1.0 - absolute_index) / denominator
    log_length = np.log(lengths)
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    bfactor, bfactor_valid = aligned_property(selection, labels, "bfactor_z", "bfactor_valid")
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]
    bfactor, bfactor_valid = bfactor[sampled], bfactor_valid[sampled]

    rows = []
    for checkpoint_id in IDS:
        values = selection.build_X(latents[checkpoint_id])[sampled]
        neighbours = external_neighbours(values, protein_index)
        rows.extend([
            {"checkpoint": checkpoint_id, "property": "aa20", "metric": "purity_at_10", "value": classification_purity(aa, neighbours), "n": len(aa)},
            {"checkpoint": checkpoint_id, "property": "secondary_structure", "metric": "purity_at_10", "value": classification_purity(ss, neighbours), "n": len(ss)},
        ])
        for name, target, valid in (
            ("normalized_chain_position", normalized_position, np.ones(N, dtype=bool)),
            ("normalized_terminal_distance", terminal_distance, np.ones(N, dtype=bool)),
            ("log_chain_length", log_length, np.ones(N, dtype=bool)),
            ("rsa", rsa, rsa_valid),
            ("bfactor_z", bfactor, bfactor_valid),
        ):
            r2, rho, n = regression_scores(target, valid, neighbours)
            rows.extend([
                {"checkpoint": checkpoint_id, "property": name, "metric": "knn10_r2", "value": r2, "n": n},
                {"checkpoint": checkpoint_id, "property": name, "metric": "knn10_spearman", "value": rho, "n": n},
            ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("checkpoint", "property", "metric", "value", "n"))
        writer.writeheader()
        writer.writerows(rows)

    score = {
        (row["checkpoint"], row["property"], row["metric"]): row["value"]
        for row in rows
    }
    properties = (
        ("normalized_chain_position", normalized_position, np.ones(N, dtype=bool), "Normalized chain position"),
        ("log_chain_length", log_length, np.ones(N, dtype=bool), "Log chain length"),
        ("rsa", rsa, rsa_valid, "Relative solvent accessibility (RSA)"),
    )
    names = {
        IDS[0]: "Original absolute positions (100k)",
        IDS[1]: "Relative positions (10k)",
        IDS[2]: "Relative positions (30k)",
    }
    fig, axes = plt.subplots(len(properties), len(IDS), figsize=(17, 14))
    for row_index, (property_name, target, valid, label) in enumerate(properties):
        finite = valid & np.isfinite(target)
        low, high = np.quantile(target[finite], [0.01, 0.99])
        for column_index, checkpoint_id in enumerate(IDS):
            ax = axes[row_index, column_index]
            embedding = np.load(
                coordinate_path(checkpoint_id, N, 30.0, SEED), allow_pickle=False
            )["embedding"]
            ax.scatter(
                embedding[finite, 0], embedding[finite, 1], c=target[finite],
                cmap="viridis", vmin=low, vmax=high, s=3, alpha=0.72,
                linewidths=0, rasterized=True,
            )
            rho = score[(checkpoint_id, property_name, "knn10_spearman")]
            ax.set_title(f"{names[checkpoint_id]}\n10-NN Spearman = {rho:.3f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if column_index == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(
        "What organizes the LeJEPA latent space?\n"
        "Same 15,000 residues; quantitative scores use cross-protein neighbours in the original 8D space",
        fontsize=15,
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE, dpi=210, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"wrote {FIGURE}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
