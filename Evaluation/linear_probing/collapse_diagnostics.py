"""Script 1: collapse diagnostics on cached per-residue latents.

Reuses Evaluation/linear_probing/latents/{checkpoint_id}.pt as-is -- no
encoder re-runs. That cache is already a dict[protein_id -> tensor[n_i, 8]]
(see extract_latents.py), so protein-id-per-residue is already recoverable
losslessly by flattening the dict; no cache format change was needed.

Metrics (all computed exactly, no sampling -- see note below):
  - eff_rank: participation ratio (Sum(lambda))^2 / Sum(lambda^2) of the 8x8
    covariance over ALL pooled residues (every residue weighted equally,
    so longer proteins contribute proportionally more).
  - dim_stds: std of each of the 8 dims across all pooled residues.
  - within_protein_cos: per protein, mean pairwise cosine similarity between
    its own residue latents; averaged across proteins (mean of per-protein
    means, not pooled).
  - across_protein_cos: mean pairwise cosine between per-protein mean
    latent vectors (one point per protein).

Sampling note: the brief asks to cap pairwise cosine at N pairs if full
pairwise is too expensive. It isn't, here -- max residues/protein is 256
(dataselector max_length), so within-protein is at most 32,640 pairs; there
are ~5,224 proteins, so across-protein is ~13.6M pairs. Both are cheap
(sub-second). MAX_PAIRS_PER_PROTEIN / MAX_PAIRS_ACROSS below are kept as
defensive ceilings (comfortably above current data) rather than removed, so
this doesn't silently blow up if run on a much larger pool later.
"""

import argparse
import csv
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from loguru import logger

from checkpoints import CHECKPOINTS

LATENTS_DIR = pathlib.Path(__file__).resolve().parent / "latents"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "collapse_diagnostics.csv"

MAX_PAIRS_PER_PROTEIN = 50_000     # true max ~32,640 (256-residue protein) -- never triggers today
MAX_PAIRS_ACROSS = 20_000_000      # true count ~13.6M (5,224 proteins) -- never triggers today
DEAD_DIM_STD_THRESHOLD = 1e-3
RNG_SEED = 42


def load_checkpoint_latents(checkpoint_id: str) -> Tuple[np.ndarray, List[str], Dict[str, torch.Tensor]]:
    """Returns (X [N_total, 8] float64, protein_ids [N_total] parallel to X, per_protein dict)."""
    per_protein: Dict[str, torch.Tensor] = torch.load(LATENTS_DIR / f"{checkpoint_id}.pt", weights_only=False)
    xs, protein_ids = [], []
    for pid, z in per_protein.items():
        xs.append(z)
        protein_ids.extend([pid] * z.shape[0])
    X = torch.cat(xs, dim=0).numpy().astype(np.float64)
    return X, protein_ids, per_protein


def effective_rank(X: np.ndarray) -> float:
    cov = np.cov(X, rowvar=False)  # [8, 8]
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)  # guard tiny negative numerical noise
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def per_dim_std(X: np.ndarray) -> np.ndarray:
    return X.std(axis=0)


def _mean_pairwise_cosine(vecs: np.ndarray, rng: np.random.Generator, max_pairs: int) -> float:
    m = vecs.shape[0]
    if m < 2:
        return float("nan")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    unit = vecs / norms
    n_pairs_full = m * (m - 1) // 2
    if n_pairs_full <= max_pairs:
        sims = unit @ unit.T
        iu = np.triu_indices(m, k=1)
        return float(sims[iu].mean())
    i = rng.integers(0, m, size=max_pairs)
    j = rng.integers(0, m, size=max_pairs)
    keep = i != j
    sims = (unit[i[keep]] * unit[j[keep]]).sum(axis=1)
    return float(sims.mean())


def within_protein_cosine(per_protein: Dict[str, torch.Tensor], rng: np.random.Generator) -> float:
    scores = []
    for z in per_protein.values():
        vecs = z.numpy().astype(np.float64)
        if vecs.shape[0] >= 2:
            scores.append(_mean_pairwise_cosine(vecs, rng, MAX_PAIRS_PER_PROTEIN))
    return float(np.mean(scores)) if scores else float("nan")


def across_protein_cosine(per_protein: Dict[str, torch.Tensor], rng: np.random.Generator) -> float:
    means = np.stack([z.numpy().astype(np.float64).mean(axis=0) for z in per_protein.values()])
    return _mean_pairwise_cosine(means, rng, MAX_PAIRS_ACROSS)


def classify(eff_rank: float, within_cos: float) -> str:
    if eff_rank < 2 or within_cos > 0.98:
        return "collapsed"
    if eff_rank > 4 and within_cos < 0.9:
        return "healthy"
    return "partial collapse"


@dataclass
class CollapseResult:
    checkpoint_id: str
    eff_rank: float
    dim_stds: np.ndarray
    within_protein_cos: float
    across_protein_cos: float
    flag: str


def run_diagnostics(checkpoint_ids: Optional[List[str]] = None, use_wandb: bool = True) -> List[CollapseResult]:
    checkpoint_ids = list(CHECKPOINTS) if checkpoint_ids is None else checkpoint_ids
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for checkpoint_id in checkpoint_ids:
        X, _protein_ids, per_protein = load_checkpoint_latents(checkpoint_id)
        eff_rank = effective_rank(X)
        dim_stds = per_dim_std(X)
        within_cos = within_protein_cosine(per_protein, rng)
        across_cos = across_protein_cosine(per_protein, rng)
        flag = classify(eff_rank, within_cos)
        dead_dims = [i for i, s in enumerate(dim_stds) if s < DEAD_DIM_STD_THRESHOLD]

        result = CollapseResult(checkpoint_id, eff_rank, dim_stds, within_cos, across_cos, flag)
        results.append(result)

        logger.info(
            f"{checkpoint_id:22s} eff_rank={eff_rank:.3f} within_cos={within_cos:.4f} "
            f"across_cos={across_cos:.4f} dead_dims={dead_dims or 'none'} -> {flag.upper()}"
        )

        if use_wandb:
            run = wandb.init(
                project="protein_linear_probing", name=f"{checkpoint_id}_collapse",
                config=dict(encoder=checkpoint_id), job_type="collapse_diagnostics", reinit=True,
            )
            log_dict = dict(
                eff_rank=eff_rank, within_protein_cos=within_cos, across_protein_cos=across_cos,
                n_dead_dims=len(dead_dims),
            )
            log_dict.update({f"dim_std_{i}": float(s) for i, s in enumerate(dim_stds)})
            run.log(log_dict)
            run.finish()

    _write_csv(results)
    return results


def _write_csv(results: List[CollapseResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["checkpoint", "eff_rank", "dim_std_min", "dim_std_max"]
        + [f"dim_std_{i}" for i in range(8)]
        + ["within_protein_cos", "across_protein_cos", "flag"]
    )
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(
                checkpoint=r.checkpoint_id, eff_rank=r.eff_rank,
                dim_std_min=float(r.dim_stds.min()), dim_std_max=float(r.dim_stds.max()),
                within_protein_cos=r.within_protein_cos, across_protein_cos=r.across_protein_cos,
                flag=r.flag,
            )
            row.update({f"dim_std_{i}": float(s) for i, s in enumerate(r.dim_stds)})
            writer.writerow(row)
    logger.info(f"Wrote {len(results)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_diagnostics(use_wandb=not args.no_wandb)
