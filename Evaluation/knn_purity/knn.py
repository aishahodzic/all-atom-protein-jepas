"""k-NN purity: a non-linear complement to Evaluation/linear_probing's linear
probes, on the exact same cached 8-dim per-residue latents (no re-extraction
-- see Evaluation/linear_probing/latents/{checkpoint_id}.pt, produced by
extract_latents.py, and latents/labels.pt for the shared per-residue labels).

For each residue, do its k nearest neighbours in latent space (cosine) share
its label? Purity = fraction of the k neighbours that do, averaged over
query residues. High purity = the latent clusters residues by that property
even where a *linear* probe can't separate them.

Query/neighbour pool design (see run.py for the orchestration that builds
these): unlike the linear probes, k-NN purity has no fitting step, so there
is no train/test split here -- queries and neighbour candidates are drawn
from the SAME pool: every valid residue across every protein in the shared
val+test protein pool (Evaluation/linear_probing/data_pool.py), exactly
"same test+val protein pool, same residues as the probes". Held-out
generalization isn't the concern a train/test split would guard against;
same-protein leakage is, so that's excluded explicitly below.

Same-protein exclusion (non-negotiable -- see module docstring in run.py):
a residue's neighbours must come from OTHER proteins, or purity is inflated
by trivially-similar adjacent residues from the same chain. This also
subsumes "exclude the query residue itself" (a residue is always in the
same protein as itself), so one mask does both.

Distance metric: cosine similarity (normalize, then dot product / matmul).
"""

import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path.insert(0, str(LINEAR_PROBING_DIR))

from checkpoints import ALL_CHECKPOINTS  # noqa: E402

LATENTS_DIR = LINEAR_PROBING_DIR / "latents"
LABELS_CACHE_PATH = LATENTS_DIR / "labels.pt"

SEED = 42
K_VALUES = (5, 10, 50)
DEFAULT_MAX_QUERIES = 20_000  # subsample cap on queries; neighbour pool always stays full (see run.py)
DEFAULT_BATCH_SIZE = 256      # query rows per batched similarity matmul

LABEL_SPECS = {
    "aa": dict(label_key="aa", valid_key="aa_valid"),
    "ss": dict(label_key="ss", valid_key="ss_valid"),
}


def load_labels() -> Dict[str, dict]:
    return torch.load(LABELS_CACHE_PATH, weights_only=False)


def load_encoder_latents(checkpoint_id: str) -> Dict[str, torch.Tensor]:
    return torch.load(LATENTS_DIR / f"{checkpoint_id}.pt", weights_only=False)


def common_protein_ids(per_protein_by_checkpoint: Dict[str, Dict[str, torch.Tensor]]) -> List[str]:
    """Sorted intersection of protein ids cached across all requested encoders --
    defensive (in practice every checkpoint_id's cache comes from the same
    extract() pool), and sorting makes residue ordering below deterministic
    and independent of any one encoder's dict insertion order, which is what
    lets query_indices mean "the same physical residues" across encoders."""
    ids = None
    for per_protein in per_protein_by_checkpoint.values():
        keys = set(per_protein.keys())
        ids = keys if ids is None else (ids & keys)
    return sorted(ids or set())


class ResidueSelection:
    """Encoder-agnostic: which (protein_id, local residue index) pairs are
    valid for one label property, in a fixed deterministic order, plus their
    labels and an integer protein id per residue (for same-protein masking).
    Computed once per property and reused for every encoder + the random
    control, so X arrays built from it are guaranteed index-aligned."""

    def __init__(self, protein_ids: List[str], labels_by_protein: Dict[str, dict], label_key: str, valid_key: str):
        self.keys: List[Tuple[str, np.ndarray]] = []  # (protein_id, local indices) per protein, valid only
        ys, protein_idx = [], []
        for p_int, pid in enumerate(protein_ids):
            lab = labels_by_protein.get(pid)
            if lab is None:
                continue
            valid = lab[valid_key]
            idx = np.nonzero(valid)[0]
            if idx.size == 0:
                continue
            self.keys.append((pid, idx))
            ys.append(lab[label_key][idx])
            protein_idx.append(np.full(idx.size, p_int, dtype=np.int64))
        self.y = np.concatenate(ys, axis=0)
        self.protein_idx = np.concatenate(protein_idx, axis=0)
        self.n = self.y.shape[0]

    def build_X(self, per_protein: Dict[str, torch.Tensor]) -> torch.Tensor:
        xs = [per_protein[pid][idx] for pid, idx in self.keys]
        return torch.cat(xs, dim=0)


def sample_query_indices(n_pool: int, max_queries: int = DEFAULT_MAX_QUERIES, seed: int = SEED) -> np.ndarray:
    """A fixed, seeded subsample of query indices into the pool -- reused
    identically across every encoder and the random-latent control for this
    property, so all of them are evaluated on the exact same queries."""
    rng = np.random.default_rng(seed)
    if n_pool <= max_queries:
        return np.arange(n_pool)
    return np.sort(rng.choice(n_pool, size=max_queries, replace=False))


def majority_baseline(y_query: np.ndarray) -> float:
    """Chance purity: majority-class frequency of the query set -- the same
    definition probe.py uses for its classification baseline (a constant
    "always predict the majority class" rule), reused here so the two
    baselines are directly comparable rather than two different notions of
    "chance"."""
    counts = np.bincount(y_query.astype(np.int64))
    return float(counts.max() / counts.sum())


def knn_purity(
    X: torch.Tensor,
    y: np.ndarray,
    protein_idx: np.ndarray,
    query_indices: np.ndarray,
    k_values: Sequence[int] = K_VALUES,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[int, float]:
    """Exact brute-force cosine k-NN, batched over queries so the full
    [n_queries, n_pool] similarity matrix is never materialized at once.
    Same-protein candidates (including the query itself) are masked to -inf
    before topk; since a protein has at most a couple hundred residues and
    the pool has ~750k, there are always far more than max(k_values) valid
    (other-protein) candidates left, so topk(max(k_values)) after masking
    can never fall back to a masked entry -- no extra "fetch buffer" needed.
    """
    max_k = max(k_values)
    X = X.to(device=device, dtype=torch.float32)
    X_norm = torch.nn.functional.normalize(X, dim=-1)
    y_t = torch.from_numpy(y).to(device)
    protein_idx_t = torch.from_numpy(protein_idx).to(device)
    query_indices_t = torch.from_numpy(query_indices).to(device)

    purity_sums = {k: 0.0 for k in k_values}
    n_queries = len(query_indices)

    for start in range(0, n_queries, batch_size):
        batch_idx = query_indices_t[start : start + batch_size]
        q = X_norm[batch_idx]  # [B, 8]
        sims = q @ X_norm.T  # [B, n_pool]
        same_protein = protein_idx_t[batch_idx].unsqueeze(1) == protein_idx_t.unsqueeze(0)  # [B, n_pool]
        sims.masked_fill_(same_protein, float("-inf"))

        _, nn_idx = torch.topk(sims, max_k, dim=1)  # sorted descending by similarity
        nn_labels = y_t[nn_idx]  # [B, max_k]
        query_labels = y_t[batch_idx].unsqueeze(1)  # [B, 1]
        matches = (nn_labels == query_labels).float()  # [B, max_k]
        cum_matches = matches.cumsum(dim=1)  # [B, max_k]

        for k in k_values:
            purity_sums[k] += (cum_matches[:, k - 1] / k).sum().item()

    return {k: purity_sums[k] / n_queries for k in k_values}


def random_latent_knn_purity(
    n_pool: int,
    latent_dim: int,
    y: np.ndarray,
    protein_idx: np.ndarray,
    query_indices: np.ndarray,
    k_values: Sequence[int] = K_VALUES,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = SEED,
) -> Dict[int, float]:
    """Same procedure, same query set, same labels/protein grouping -- but
    the "latents" are iid standard-normal noise of the same shape. Encoder-
    independent by construction; call once per property, reuse across every
    encoder's row (mirrors probe.compute_random_control)."""
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.standard_normal((n_pool, latent_dim)).astype(np.float32))
    return knn_purity(X, y, protein_idx, query_indices, k_values, device=device, batch_size=batch_size)
