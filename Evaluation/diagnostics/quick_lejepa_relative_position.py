"""Fast CPU diagnostic for the relative-position LeJEPA checkpoint.

This is an early directional result on a fixed 128-protein subset.  The full
publication analysis remains ``latent_viz/analyze_lejepa_relative_positions.py``.
"""

import csv
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm


REPO = pathlib.Path(__file__).resolve().parents[2]
LINEAR = REPO / "Evaluation" / "linear_probing"
sys.path[:0] = [str(REPO), str(LINEAR)]

import checkpoints  # noqa: E402
from data_pool import build_protein_pool, eval_transform_compose  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import (  # noqa: E402
    dense_padded_from_data_list,
)


CONTROL = "le-jepa-rand-mask-50"
INTERVENTION = "le-jepa-rand50-relative-pos-10k"
SUFFIX = pathlib.Path(
    "aishod/runs/le_jepa_rand50_relative_positions_10k/"
    "store/le_jepa_rand50_relative_positions_10k/checkpoints/"
    "chk_epoch=00000001_step=000000010000.ckpt"
)
CHECKPOINT = next(
    (p / SUFFIX for p in (pathlib.Path("/mnt/hopper-raid"), pathlib.Path("/mnt/raid"))
     if (p / SUFFIX).exists()),
    pathlib.Path("/mnt/hopper-raid") / SUFFIX,
)
N_PROTEINS = 128
SEED = 42
MAX_TSNE_POINTS = 15_000
OUT_DIR = REPO / "Evaluation" / "latent_viz" / "results"
FIG_DIR = REPO / "Evaluation" / "latent_viz" / "figures" / "tsne_absolute_index"


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def neighbour_stats(values: np.ndarray, indices: np.ndarray, proteins: np.ndarray) -> dict:
    normalized_values = normalized(values)
    _, neighbours = cKDTree(normalized_values).query(
        normalized_values, k=11, workers=-1
    )
    neighbours = neighbours[:, 1:]
    exact = indices[neighbours] == indices[:, None]
    same_protein = proteins[neighbours] == proteins[:, None]
    return {
        "exact_index_purity_at_10": float(exact.mean()),
        "nearest_neighbor_exact_index": float(exact[:, 0].mean()),
        "same_protein_fraction_at_10": float(same_protein.mean()),
    }


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    entry = (str(CHECKPOINT), "jepa")
    checkpoints.CHECKPOINTS[INTERVENTION] = entry
    checkpoints.ALL_CHECKPOINTS[INTERVENTION] = entry

    old_cache = torch.load(
        LINEAR / "latents" / f"{CONTROL}.pt", weights_only=False
    )
    pool = build_protein_pool()
    pool_index = {pid: i for i, pid in enumerate(pool.protein_ids)}
    eligible = sorted(set(old_cache) & set(pool_index))
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(eligible, size=N_PROTEINS, replace=False).tolist())

    encoder = checkpoints.load_frozen_encoder(INTERVENTION, device="cpu")
    transform = eval_transform_compose()
    old_chunks, new_chunks, index_chunks, protein_chunks = [], [], [], []
    with torch.inference_mode():
        for protein_number, pid in enumerate(tqdm(chosen, desc="CPU quick diagnostic")):
            graph = transform(pool.dataset[pool_index[pid]])
            batch = dense_padded_from_data_list([graph])
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
            mean = encoder(batch)["mean"][0][batch["mask"][0]].float().numpy()
            old = old_cache[pid].float().numpy()
            n = min(len(mean), len(old))
            old_chunks.append(old[:n])
            new_chunks.append(mean[:n])
            index_chunks.append(np.arange(n, dtype=np.int64))
            protein_chunks.append(np.full(n, protein_number, dtype=np.int64))

    arrays = {
        CONTROL: np.concatenate(old_chunks),
        INTERVENTION: np.concatenate(new_chunks),
    }
    indices = np.concatenate(index_chunks)
    proteins = np.concatenate(protein_chunks)
    counts = np.bincount(indices)
    chance = float((counts * (counts - 1)).sum() / (len(indices) * (len(indices) - 1)))

    rows = []
    for name, values in arrays.items():
        rows.append({
            "checkpoint": name,
            "proteins": N_PROTEINS,
            "residues": len(values),
            "random_pair_exact_index_probability": chance,
            **neighbour_stats(values, indices, proteins),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = OUT_DIR / "lejepa_relative_position_quick128_summary.csv"
    with open(result, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sampled = (
        np.arange(len(indices)) if len(indices) <= MAX_TSNE_POINTS else
        np.sort(rng.choice(len(indices), MAX_TSNE_POINTS, replace=False))
    )
    embeddings = {
        name: TSNE(
            n_components=2, perplexity=30, metric="cosine", init="random",
            learning_rate="auto", max_iter=1000, random_state=SEED,
        ).fit_transform(values[sampled])
        for name, values in arrays.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    scatter = None
    titles = {
        CONTROL: "Original LeJEPA rand50 (100k)",
        INTERVENTION: "Relative-position LeJEPA rand50 (10k)",
    }
    for ax, row in zip(axes, rows):
        name = row["checkpoint"]
        scatter = ax.scatter(
            embeddings[name][:, 0], embeddings[name][:, 1],
            c=indices[sampled] + 1, cmap="turbo", s=5, alpha=0.7,
            linewidths=0, rasterized=True,
        )
        ax.set_title(
            f"{titles[name]}\n8D absolute-index purity@10="
            f"{row['exact_index_purity_at_10']:.3f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Preliminary relative-position shortcut test: {N_PROTEINS} held-out proteins, "
        f"{len(sampled):,} matched residues"
    )
    fig.subplots_adjust(right=0.91, top=0.86, wspace=0.04)
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    fig.colorbar(scatter, cax=cax, label="absolute residue index (one-based)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure = FIG_DIR / "lejepa_relative_position_quick128.png"
    fig.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {result}")
    print(f"wrote {figure}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
