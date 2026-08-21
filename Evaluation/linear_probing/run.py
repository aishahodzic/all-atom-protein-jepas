"""Orchestrates B1-B5: sanity gate first, then the full 8-encoder x
4-property sweep, writing CSV + W&B.

Usage: python run.py [--no-wandb] [--seed SEED]
"""

import argparse
import csv
import pathlib

import wandb
from loguru import logger

from checkpoints import CHECKPOINTS
from extract_latents import extract
from probe import PROPERTIES, compute_random_control, run_probe

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "linear_probe_results.csv"
CSV_FIELDS = ["encoder", "property", "seed", "probe_score", "baseline", "random_control", "n_train", "n_test"]


def _write_csv_rows(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanity_gate(seed: int = 42, margin_threshold: float = 0.10):
    """B5: ca-cond AE must clearly beat chance on AA (trained decoder gets
    ~100%; linear may be lower but must beat both baselines by a clear
    margin). If not, stop -- pipeline is broken."""
    pool, labels, latents = extract(checkpoint_ids=["ca-cond"])
    result = run_probe("aa", pool.protein_ids, pool.probe_split, labels, latents["ca-cond"], seed=seed)
    logger.info(
        f"[Sanity gate] ca-cond AA probe: accuracy={result.probe_score:.4f}, "
        f"majority_baseline={result.baseline:.4f}, random_control={result.random_control:.4f} "
        f"(n_train={result.n_train}, n_test={result.n_test})"
    )
    margin = result.probe_score - max(result.baseline, result.random_control)
    if margin < margin_threshold:
        raise RuntimeError(
            f"Sanity gate FAILED: ca-cond AA probe accuracy ({result.probe_score:.4f}) is not clearly "
            f"above baseline/chance (majority={result.baseline:.4f}, random={result.random_control:.4f}, "
            f"margin={margin:.4f} < {margin_threshold}). Pipeline is likely broken -- stopping before the full run."
        )
    logger.info(f"[Sanity gate] PASSED (margin over best baseline: {margin:.4f}).")
    return result


def run_all(seed: int = 42, use_wandb: bool = True):
    sanity_gate(seed=seed)  # raises RuntimeError if this fails

    pool, labels, latents = extract(checkpoint_ids=list(CHECKPOINTS))

    random_controls = {
        prop: compute_random_control(prop, pool.protein_ids, pool.probe_split, labels, latent_dim=8, seed=seed)
        for prop in PROPERTIES
    }
    logger.info(f"Random-latent controls (once per property, reused across all encoders): {random_controls}")

    rows = []
    for checkpoint_id in CHECKPOINTS:
        for prop in PROPERTIES:
            result = run_probe(
                prop, pool.protein_ids, pool.probe_split, labels, latents[checkpoint_id],
                random_control_score=random_controls[prop], seed=seed,
            )
            row = dict(
                encoder=checkpoint_id, property=prop, seed=seed,
                probe_score=result.probe_score,
                baseline="" if result.baseline is None else result.baseline,
                random_control=result.random_control,
                n_train=result.n_train, n_test=result.n_test,
            )
            rows.append(row)
            logger.info(
                f"{checkpoint_id:22s} {prop:8s} score={result.probe_score:.4f} "
                f"baseline={row['baseline']} random={result.random_control:.4f}"
            )

            if use_wandb:
                run = wandb.init(
                    project="protein_linear_probing",
                    name=f"{checkpoint_id}_{prop}_seed{seed}",
                    config=dict(encoder=checkpoint_id, property=prop, seed=seed),
                    reinit=True,
                )
                run.log(dict(
                    probe_score=result.probe_score,
                    baseline=result.baseline if result.baseline is not None else float("nan"),
                    random_control=result.random_control,
                    n_train=result.n_train, n_test=result.n_test,
                ))
                run.finish()

    _write_csv_rows(rows)
    logger.info(f"Wrote {len(rows)} rows -> {CSV_PATH}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_all(seed=args.seed, use_wandb=not args.no_wandb)
