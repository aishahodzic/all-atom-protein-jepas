"""2D visualization of cached 8-dim per-residue latents (PCA + UMAP), colored
by AA identity and secondary structure -- the visual counterpart to
Evaluation/knn_purity's purity numbers. Reuses the exact same cached latents
(Evaluation/linear_probing/latents/{checkpoint_id}.pt) and labels
(latents/labels.pt) -- no re-extraction.

Residue selection, unlike knn_purity (which handles aa/ss separately since
their validity masks differ slightly): here we want ONE fixed 2D embedding
per encoder that gets colored two different ways (by AA, by SS), so the two
colorings are directly comparable point-for-point. That means residues must
be valid for BOTH properties -- see CombinedSelection below. The loss from
this intersection is negligible (~99.95% overlap between aa_valid and
ss_valid; ss_valid is the more restrictive of the two).

AA grouping (primary coloring -- 20 raw classes is unreadable in a scatter):
standard 4-way biochemical grouping. Hydrophobic (aliphatic + aromatic
nonpolar): A,V,L,I,M,F,W,Y. Polar (uncharged): S,T,N,Q,C. Charged (acidic +
basic): D,E,K,R,H. Special (backbone-flexibility outliers): G,P. Index order
matches openfold.np.residue_constants.restypes (0=A,1=R,2=N,...19=V) --
graph.residue_type in labels.py uses this order directly.
"""

import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.decomposition import PCA

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
KNN_PURITY_DIR = REPO_ROOT / "Evaluation" / "knn_purity"
sys.path.insert(0, str(LINEAR_PROBING_DIR))
sys.path.insert(0, str(KNN_PURITY_DIR))

from checkpoints import ALL_CHECKPOINTS  # noqa: E402
from knn import common_protein_ids, knn_purity, load_encoder_latents, load_labels  # noqa: E402

LATENTS_DIR = LINEAR_PROBING_DIR / "latents"
SEED = 42
DEFAULT_N_POINTS = 15_000  # subsample size for plotting (within the 10-20k range requested)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

AA_LETTERS = list("ARNDCQEGHILKMFPSTWYV")
AA_GROUP_NAMES = ["Hydrophobic", "Polar", "Charged", "Special (Gly/Pro)"]
_GROUP_OF_LETTER = {
    **{l: 0 for l in "AVLIMFWY"},   # Hydrophobic
    **{l: 1 for l in "STNQC"},      # Polar
    **{l: 2 for l in "DEKRH"},      # Charged
    **{l: 3 for l in "GP"},         # Special
}
AA_GROUP = np.array([_GROUP_OF_LETTER[l] for l in AA_LETTERS], dtype=np.int64)  # [20], index-aligned with restypes

SS_STATE_NAMES = ["Helix", "Sheet", "Loop"]

SS_COLORS = {0: "#d62728", 1: "#1f77b4", 2: "#7f7f7f"}       # helix red, sheet blue, loop gray
AA_GROUP_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#9467bd", 3: "#8c564b"}  # green/orange/purple/brown

SHORT_LABEL = {
    "ca-cond": "CA-cond", "fully-latent": "Fully-latent",
    "jepa-mutation": "JEPA-mut", "jepa-rand-mask-50": "JEPA-50", "jepa-rand-mask-70": "JEPA-70",
    "le-jepa-mutation": "LeJEPA-mut", "le-jepa-rand-mask-50": "LeJEPA-50", "le-jepa-rand-mask-70": "LeJEPA-70",
    "ae1-external": "AE1-ext", "random-init": "Random-init",
}


class CombinedSelection:
    """Encoder-agnostic: residues valid for BOTH aa and ss, in a fixed
    deterministic order (sorted protein ids), with both label arrays and an
    integer protein id per residue. One instance is built once and reused
    for every encoder, so subsampling the same positional indices means the
    same physical residues everywhere -- required for a fair visual
    comparison across encoders."""

    def __init__(self, protein_ids: List[str], labels_by_protein: Dict[str, dict]):
        self.keys: List[Tuple[str, np.ndarray]] = []
        aa_list, ss_list, protein_idx = [], [], []
        for p_int, pid in enumerate(protein_ids):
            lab = labels_by_protein.get(pid)
            if lab is None:
                continue
            valid = lab["aa_valid"] & lab["ss_valid"]
            idx = np.nonzero(valid)[0]
            if idx.size == 0:
                continue
            self.keys.append((pid, idx))
            aa_list.append(lab["aa"][idx])
            ss_list.append(lab["ss"][idx])
            protein_idx.append(np.full(idx.size, p_int, dtype=np.int64))
        self.aa = np.concatenate(aa_list, axis=0)
        self.ss = np.concatenate(ss_list, axis=0)
        self.protein_idx = np.concatenate(protein_idx, axis=0)
        self.n = self.aa.shape[0]

    def build_X(self, per_protein: Dict[str, torch.Tensor]) -> np.ndarray:
        xs = [per_protein[pid][idx] for pid, idx in self.keys]
        return torch.cat(xs, dim=0).numpy()


def sample_indices(n_pool: int, n_points: int = DEFAULT_N_POINTS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_pool <= n_points:
        return np.arange(n_pool)
    return np.sort(rng.choice(n_pool, size=n_points, replace=False))


def project_pca(X: np.ndarray, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (embedding [n,2], explained_variance_ratio for PC1/PC2)."""
    pca = PCA(n_components=2, random_state=seed)
    emb = pca.fit_transform(X)
    return emb, pca.explained_variance_ratio_[:2]


def project_umap(X: np.ndarray, n_neighbors: int = UMAP_N_NEIGHBORS, min_dist: float = UMAP_MIN_DIST,
                  seed: int = SEED) -> np.ndarray:
    """metric='cosine' deliberately, NOT UMAP's euclidean default -- matches the
    distance metric knn_purity.py used to compute the purity numbers these
    figures are annotated with, so the 2D picture and the number it's paired
    with are at least asking the same geometric question."""
    import umap  # deferred: heavy import (numba JIT), only needed here

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
        random_state=seed, n_jobs=1,  # n_jobs=1 required for reproducibility when random_state is set
    )
    return reducer.fit_transform(X)


def embedding_purity(embedding: np.ndarray, y: np.ndarray, protein_idx: np.ndarray, k: int = 10,
                      device: str = "cpu") -> float:
    """k-NN purity of the 2D embedding itself (same same-protein-exclusion
    rule as the real 8-dim metric, reusing knn_purity unmodified) -- lets
    run.py flag panels where the 2D picture disagrees with the true 8-dim
    purity number it's annotated with (a projection artifact) instead of
    relying on eyeballing."""
    X = torch.from_numpy(embedding.astype(np.float32))
    all_idx = np.arange(embedding.shape[0])
    result = knn_purity(X, y, protein_idx, all_idx, k_values=(k,), device=device, batch_size=len(all_idx))
    return result[k]
