"""Matched latent audit of the D8 JEPA-r70 absolute/relative-position pilots."""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree


REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
LINEAR = REPO / "Evaluation" / "linear_probing"
KNN = REPO / "Evaluation" / "knn_purity"
sys.path[:0] = [str(REPO), str(LINEAR), str(KNN), str(HERE)]

CHECKPOINTS = {
    "jepa-r70-d8-abspos-step30000": pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/jepa_rand70_d8_abspos_ema0p996_30k_s42/"
        "store/jepa_rand70_d8_abspos_ema0p996_30k_s42/checkpoints/"
        "chk_epoch=00000005_step=000000030000.ckpt"
    ),
    "jepa-r70-d8-relpos-step28500": pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/jepa_rand70_d8_relpos128_ema0p996_30k_s42/"
        "store/jepa_rand70_d8_relpos128_ema0p996_30k_s42/checkpoints/last.ckpt"
    ),
}
N_POINTS = 15_000
SEED = 42
RESULT = HERE / "results" / "jepa_d8_abs_rel_30k_health.csv"


def effective_rank(values: np.ndarray) -> float:
    covariance = np.cov(values.astype(np.float64), rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = eigenvalues / np.maximum(eigenvalues.sum(), 1e-12)
    nonzero = probabilities[probabilities > 0]
    return float(np.exp(-(nonzero * np.log(nonzero)).sum()))


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def external_neighbours(values: np.ndarray, proteins: np.ndarray, k: int = 10) -> np.ndarray:
    values = normalized(values)
    _, candidates = cKDTree(values).query(values, k=64, workers=-1)
    neighbours = np.empty((len(values), k), dtype=np.int64)
    for row, candidate_row in enumerate(candidates):
        usable = candidate_row[proteins[candidate_row] != proteins[row]]
        if len(usable) < k:
            raise RuntimeError("Insufficient cross-protein neighbour buffer")
        neighbours[row] = usable[:k]
    return neighbours


def main() -> None:
    import checkpoints
    from extract_latents import extract
    from knn import common_protein_ids, load_encoder_latents, load_labels
    from label_tsne_worms import local_index_array
    from run_tsne import run as run_tsne
    from viz import CombinedSelection, sample_indices

    for checkpoint_id, path in CHECKPOINTS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[checkpoint_id] = entry
        checkpoints.ALL_CHECKPOINTS[checkpoint_id] = entry

    extract(checkpoint_ids=list(CHECKPOINTS), device="cuda")

    labels = load_labels()
    latent_cache = {
        checkpoint_id: load_encoder_latents(checkpoint_id)
        for checkpoint_id in CHECKPOINTS
    }
    protein_ids = common_protein_ids(latent_cache)
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=N_POINTS, seed=SEED)
    amino_acid = selection.aa[sampled]
    protein_index = selection.protein_idx[sampled]
    absolute_index = local_index_array(selection)[sampled]

    rows = []
    for checkpoint_id, path in CHECKPOINTS.items():
        values = selection.build_X(latent_cache[checkpoint_id])[sampled].astype(np.float32)
        neighbours = external_neighbours(values, protein_index)
        rows.append({
            "checkpoint": checkpoint_id,
            "checkpoint_path": str(path),
            "sampled_proteins": len(protein_ids),
            "sampled_residues": len(sampled),
            "latent_dim": values.shape[1],
            "effective_rank": effective_rank(values),
            "aa_purity_at_10_cross_protein": float(
                (amino_acid[neighbours] == amino_acid[:, None]).mean()
            ),
            "absolute_index_purity_at_10_cross_protein": float(
                (absolute_index[neighbours] == absolute_index[:, None]).mean()
            ),
        })

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import run_tsne as tsne_module
    tsne_module.SHORT_LABEL.update({
        "jepa-r70-d8-abspos-step30000": "JEPA-r70 D8 absolute positions (30k)",
        "jepa-r70-d8-relpos-step28500": "JEPA-r70 D8 relative positions (28.5k)",
    })
    run_tsne(
        checkpoint_ids=list(CHECKPOINTS),
        n_points=N_POINTS,
        perplexity=30.0,
        seed=SEED,
        max_iter=1_000,
        force=False,
        output_tag="jepa-r70-d8-absolute-v-relative-30k",
        grid_ncols=2,
        include_purity=False,
    )
    print(f"Wrote {RESULT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the two missing latent caches")
    main()
