"""Script 1: tests whether a high across_protein_cos from collapse_diagnostics
is a benign shared DC offset (a constant vector added to every residue's
latent, regardless of protein) rather than genuine protein-level collapse.

Subtracts the global per-dimension mean latent (over ALL residues, ALL
proteins) from every latent, then recomputes across_cos/within_cos/eff_rank
on the centered latents using the exact same methods as collapse_diagnostics.
If across_cos drops a lot after centering, proteins were already separated
up to a shared offset -- benign. If it stays high, proteins genuinely map to
near-identical points even after removing any common offset.

Cached latents only, no encoder re-runs. Takes checkpoint_id as a parameter
(defaults to fully-latent, the one flagged) so it runs on any encoder.
"""

import argparse

import numpy as np
import torch
from loguru import logger

from collapse_diagnostics import (
    RNG_SEED,
    across_protein_cosine,
    effective_rank,
    load_checkpoint_latents,
    within_protein_cosine,
)


def run_check(checkpoint_id: str = "fully-latent", seed: int = RNG_SEED) -> dict:
    X, _protein_ids, per_protein = load_checkpoint_latents(checkpoint_id)

    rng = np.random.default_rng(seed)
    eff_rank_orig = effective_rank(X)
    within_orig = within_protein_cosine(per_protein, rng)
    across_orig = across_protein_cosine(per_protein, rng)

    global_mean = X.mean(axis=0)  # [8], over ALL residues, ALL proteins
    global_mean_t = torch.from_numpy(global_mean)
    per_protein_centered = {pid: (z.double() - global_mean_t).float() for pid, z in per_protein.items()}
    X_centered = X - global_mean

    rng = np.random.default_rng(seed)  # fresh, matches original's sampling path if ever sample-capped
    eff_rank_c = effective_rank(X_centered)
    within_c = within_protein_cosine(per_protein_centered, rng)
    across_c = across_protein_cosine(per_protein_centered, rng)

    logger.info(f"{checkpoint_id}: global_mean latent = {global_mean}")
    header = f"{'':16s} {'original':>10s} {'centered':>10s}"
    row_across = f"{'across_cos':16s} {across_orig:10.4f} {across_c:10.4f}"
    row_within = f"{'within_cos':16s} {within_orig:10.4f} {within_c:10.4f}"
    row_rank = f"{'eff_rank':16s} {eff_rank_orig:10.4f} {eff_rank_c:10.4f}"
    for line in (header, row_across, row_within, row_rank):
        print(line)
        logger.info(line)

    if across_c < 0.4:
        verdict = (
            f"BENIGN SHARED OFFSET -- across_cos dropped from {across_orig:.4f} to {across_c:.4f} "
            f"after centering; proteins do separate once the common DC component is removed."
        )
    else:
        verdict = (
            f"NOT a shared-offset artifact -- across_cos stays high ({across_c:.4f}) after centering; "
            f"proteins genuinely don't separate in this latent space. Matters for per-protein tasks."
        )
    logger.info(verdict)
    print(verdict)

    return dict(
        checkpoint_id=checkpoint_id, global_mean=global_mean,
        across_orig=across_orig, across_centered=across_c,
        within_orig=within_orig, within_centered=within_c,
        eff_rank_orig=eff_rank_orig, eff_rank_centered=eff_rank_c,
        verdict=verdict,
    )


def run_all(seed: int = RNG_SEED) -> None:
    from checkpoints import ALL_CHECKPOINTS

    results = [run_check(checkpoint_id=c, seed=seed) for c in ALL_CHECKPOINTS]

    print()
    header = f"{'checkpoint':22s} {'across_orig':>12s} {'across_centered':>16s} {'offset_norm':>12s}"
    print(header)
    logger.info(header)
    for r in results:
        offset_norm = float(np.linalg.norm(r["global_mean"]))
        line = f"{r['checkpoint_id']:22s} {r['across_orig']:12.4f} {r['across_centered']:16.4f} {offset_norm:12.4f}"
        print(line)
        logger.info(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="fully-latent")
    parser.add_argument("--all", action="store_true", help="run on every cached checkpoint (my 8 + ae1-external)")
    args = parser.parse_args()
    if args.all:
        run_all()
    else:
        run_check(checkpoint_id=args.checkpoint)
