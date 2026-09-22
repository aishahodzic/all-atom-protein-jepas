"""Script 2, Phase B: random-init (untrained) EncoderTransformer floor.

Same architecture/config as all 8 in-house checkpoints -- verified in Phase A
that every one of them shares a byte-for-byte identical EncoderTransformer
config, so one floor covers all 8 (not ae1-external, which uses a different
feats_seq/feats_pair_repr -- see Phase A note). Skips weight loading only;
everything else (transform stack, batching, caching format) is identical to
the trained checkpoints, reusing extract_latents.extract() unmodified.

"""

import argparse
import csv

import numpy as np
import torch
import wandb
from loguru import logger

from checkpoints import RANDOM_INIT_CHECKPOINTS
from collapse_diagnostics import (
    CSV_PATH as COLLAPSE_CSV_PATH,
    RNG_SEED as COLLAPSE_RNG_SEED,
    across_protein_cosine,
    classify,
    effective_rank,
    load_checkpoint_latents,
    per_dim_std,
    within_protein_cosine,
)
from extract_latents import extract
from probe import PROPERTIES, compute_random_control, run_probe
from run import CSV_FIELDS as PROBE_CSV_FIELDS, CSV_PATH as PROBE_CSV_PATH, RESULTS_DIR

CHECKPOINT_ID = "random-init"
SEED = 42


def append_probe_rows(rows) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PROBE_CSV_PATH.exists()
    with open(PROBE_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBE_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"Appended {len(rows)} probe rows for {CHECKPOINT_ID} -> {PROBE_CSV_PATH}")


def append_collapse_row(row) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["checkpoint", "eff_rank", "dim_std_min", "dim_std_max"]
        + [f"dim_std_{i}" for i in range(8)]
        + ["within_protein_cos", "across_protein_cos", "flag"]
    )
    write_header = not COLLAPSE_CSV_PATH.exists()
    with open(COLLAPSE_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Appended 1 collapse row for {CHECKPOINT_ID} -> {COLLAPSE_CSV_PATH}")


def run_probes(seed: int = SEED, use_wandb: bool = True):
    assert CHECKPOINT_ID in RANDOM_INIT_CHECKPOINTS, f"{CHECKPOINT_ID} not registered"
    pool, labels, latents = extract(checkpoint_ids=[CHECKPOINT_ID])
    checkpoint_latents = latents[CHECKPOINT_ID]
    latent_dim = next(iter(checkpoint_latents.values())).shape[-1]
    logger.info(f"{CHECKPOINT_ID}: latent_dim={latent_dim}")

    rows = []
    for prop in PROPERTIES:
        random_control = compute_random_control(
            prop, pool.protein_ids, pool.probe_split, labels, latent_dim=latent_dim, seed=seed
        )
        result = run_probe(
            prop, pool.protein_ids, pool.probe_split, labels, checkpoint_latents,
            random_control_score=random_control, seed=seed,
        )
        row = dict(
            encoder=CHECKPOINT_ID, property=prop, seed=seed,
            probe_score=result.probe_score,
            baseline="" if result.baseline is None else result.baseline,
            random_control=result.random_control,
            n_train=result.n_train, n_test=result.n_test,
        )
        rows.append(row)
        logger.info(
            f"{CHECKPOINT_ID:22s} {prop:8s} score={result.probe_score:.4f} "
            f"baseline={row['baseline']} random={result.random_control:.4f}"
        )
        if use_wandb:
            run = wandb.init(
                project="protein_linear_probing", name=f"{CHECKPOINT_ID}_{prop}_seed{seed}",
                config=dict(encoder=CHECKPOINT_ID, property=prop, seed=seed, floor_baseline=True),
                reinit=True,
            )
            run.log(dict(
                probe_score=result.probe_score,
                baseline=result.baseline if result.baseline is not None else float("nan"),
                random_control=result.random_control,
                n_train=result.n_train, n_test=result.n_test,
            ))
            run.finish()

    append_probe_rows(rows)
    return rows, latent_dim


def run_collapse(use_wandb: bool = True):
    rng = np.random.default_rng(COLLAPSE_RNG_SEED)
    X, _protein_ids, per_protein = load_checkpoint_latents(CHECKPOINT_ID)
    eff_rank = effective_rank(X)
    dim_stds = per_dim_std(X)
    within_cos = within_protein_cosine(per_protein, rng)
    across_cos = across_protein_cosine(per_protein, rng)
    flag = classify(eff_rank, within_cos)
    dead_dims = [i for i, s in enumerate(dim_stds) if s < 1e-3]

    logger.info(
        f"{CHECKPOINT_ID:22s} eff_rank={eff_rank:.3f} within_cos={within_cos:.4f} "
        f"across_cos={across_cos:.4f} dead_dims={dead_dims or 'none'} -> {flag.upper()}"
    )

    row = dict(
        checkpoint=CHECKPOINT_ID, eff_rank=eff_rank,
        dim_std_min=float(dim_stds.min()), dim_std_max=float(dim_stds.max()),
        within_protein_cos=within_cos, across_protein_cos=across_cos, flag=flag,
    )
    row.update({f"dim_std_{i}": float(s) for i, s in enumerate(dim_stds)})
    append_collapse_row(row)

    if use_wandb:
        run = wandb.init(
            project="protein_linear_probing", name=f"{CHECKPOINT_ID}_collapse",
            config=dict(encoder=CHECKPOINT_ID, floor_baseline=True), job_type="collapse_diagnostics", reinit=True,
        )
        log_dict = dict(eff_rank=eff_rank, within_protein_cos=within_cos,
                         across_protein_cos=across_cos, n_dead_dims=len(dead_dims))
        log_dict.update({f"dim_std_{i}": float(s) for i, s in enumerate(dim_stds)})
        run.log(log_dict)
        run.finish()

    return row, flag


def print_interpretation_note(probe_rows) -> None:
    lines = ["", "=" * 70, "RANDOM-INIT FLOOR -- INTERPRETATION NOTE", "=" * 70]
    any_above = False
    for r in probe_rows:
        prop, score = r["property"], r["probe_score"]
        baseline = r["baseline"] if r["baseline"] != "" else None
        reference = baseline if baseline is not None else r["random_control"]
        margin = score - reference
        if margin > 0.02:
            any_above = True
            lines.append(
                f"{prop}: random-init scores {score:.4f} vs reference {reference:.4f} "
                f"(margin {margin:+.4f}) -- ABOVE baseline. Some of {prop} is readable from raw "
                f"input featurization alone; discount trained-model scores on {prop} by roughly "
                f"this margin before crediting them to learning."
            )
        else:
            lines.append(
                f"{prop}: random-init scores {score:.4f} vs reference {reference:.4f} "
                f"(margin {margin:+.4f}) -- at chance, as expected."
            )
    if not any_above:
        lines.append(
            "Random-init sits at baseline on every property -- your trained models' above-baseline "
            "probe scores reflect genuine learning, not input featurization alone."
        )
    lines.append("=" * 70)
    for line in lines:
        logger.info(line)


def main(seed: int = SEED, use_wandb: bool = True):
    torch.manual_seed(seed)  # reproducible "random" init -- consumed by EncoderTransformer's
    probe_rows, _latent_dim = run_probes(seed=seed, use_wandb=use_wandb)
    _collapse_row, _flag = run_collapse(use_wandb=use_wandb)
    print_interpretation_note(probe_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, use_wandb=not args.no_wandb)
