"""k-NN fold-retrieval purity: the training-free counterpart to StructTokenBench's
trained classifier on the Homo task. For each protein in a TEST split (query),
find its k nearest neighbours by Euclidean distance among the mean/max-pooled
8-dim protein vectors, restricted to the TRAIN+VALID proteins (the "database"
a retrieval system would search / what a classifier learns from) -- mirrors
STB's train-on-train, evaluate-on-test-split structure exactly, so purity is
directly comparable to STB's per-split AUROC on the identical proteins.

Same-protein exclusion: a protein_key ("{pdb_id}_{chain_id}") can appear in
more than one split's rows if that chain hosts multiple SCOP domains
assigned to different remoteness buckets (e.g. one domain's fold in train,
another's in test_family_holdout) -- its POOLED VECTOR is identical either
way (extract.py pools over the whole chain, not per-domain sub-ranges), so
without exclusion a query could trivially retrieve its own vector out of the
"database" at distance 0. Excluded by protein_key, not just row identity.
"""

import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

LATENTS_DIR = pathlib.Path(__file__).resolve().parent / "latents_homo"
SEED = 42
K_VALUES = (10,)
TEST_SPLITS = ["test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"]
DATABASE_SPLITS = ["train", "valid"]


def load_data():
    pooled = torch.load(LATENTS_DIR / "pooled_latents.pt", weights_only=False)
    rows = torch.load(LATENTS_DIR / "protein_rows.pt", weights_only=False)
    for r in rows:
        r["protein_key"] = f"{r['pdb_id']}_{r['chain_id']}"
    # keep only rows whose structure was actually encoded (some fail to parse -- see extract.py)
    rows = [r for r in rows if r["protein_key"] in pooled]
    return pooled, rows


def _vectors_for_split(pooled, rows, split: str, encoder: str, pool_mode: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    split_rows = [r for r in rows if r["split"] == split]
    X = np.stack([pooled[r["protein_key"]][encoder][pool_mode] for r in split_rows]).astype(np.float32)
    y = np.array([r["fold_label45"] for r in split_rows], dtype=np.int64)
    keys = [r["protein_key"] for r in split_rows]
    return X, y, keys


def knn_purity_against_database(
    query_X: np.ndarray, query_y: np.ndarray, query_keys: Sequence[str],
    db_X: np.ndarray, db_y: np.ndarray, db_keys: Sequence[str],
    k_values: Sequence[int] = K_VALUES, device: str = "cpu",
) -> Dict[int, float]:
    """Exact brute-force Euclidean k-NN of each query against the database,
    excluding database entries that share the query's protein_key."""
    max_k = max(k_values)
    qX = torch.from_numpy(query_X).to(device)
    dX = torch.from_numpy(db_X).to(device)
    qy = torch.from_numpy(query_y).to(device)
    dy = torch.from_numpy(db_y).to(device)

    # squared Euclidean distance via ||q||^2 + ||d||^2 - 2 q.d
    d2 = (qX * qX).sum(dim=1, keepdim=True) + (dX * dX).sum(dim=1).unsqueeze(0) - 2 * qX @ dX.T  # [n_q, n_db]

    same_protein = np.array(query_keys)[:, None] == np.array(db_keys)[None, :]
    same_protein_t = torch.from_numpy(same_protein).to(device)
    d2 = d2.masked_fill(same_protein_t, float("inf"))

    eff_k = min(max_k, db_X.shape[0] - 1)
    _, nn_idx = torch.topk(d2, eff_k, dim=1, largest=False)
    nn_labels = dy[nn_idx]  # [n_q, eff_k]
    matches = (nn_labels == qy.unsqueeze(1)).float()
    cum = matches.cumsum(dim=1)

    return {k: (cum[:, min(k, eff_k) - 1] / min(k, eff_k)).mean().item() for k in k_values}


def majority_baseline(y: np.ndarray) -> float:
    counts = np.bincount(y.astype(np.int64))
    return float(counts.max() / counts.sum())


def random_control(n_query: int, n_db: int, dim: int, query_y: np.ndarray, db_y: np.ndarray,
                    k_values: Sequence[int] = K_VALUES, device: str = "cpu", seed: int = SEED) -> Dict[int, float]:
    rng = np.random.default_rng(seed)
    qX = rng.standard_normal((n_query, dim)).astype(np.float32)
    dX = rng.standard_normal((n_db, dim)).astype(np.float32)
    fake_q_keys = [f"q{i}" for i in range(n_query)]
    fake_d_keys = [f"d{i}" for i in range(n_db)]
    return knn_purity_against_database(qX, query_y, fake_q_keys, dX, db_y, fake_d_keys, k_values, device)
