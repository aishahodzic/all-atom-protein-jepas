"""Script 2, Phase B: runs NVIDIA's published La-Proteina AE1 (unconditional,
<=512 residues, checkpoints_laproteina/AE1_ucond_512.ckpt) through the exact
same probe + collapse pipeline as the 8 in-house checkpoints, and appends it
as one extra row to both existing result tables.

Not folded into run.py's main sweep (CHECKPOINTS) on purpose -- AE1 is an
external reference point (different training corpus: AFDB, not PDB; 640k
cumulative steps, not 100k).

Reuses, unmodified: extract_latents.extract() (same protein pool, same
eval_transform_compose -- verified in Phase A that AE1's encoder needs
nothing beyond what that transform stack already produces), probe.run_probe
/ compute_random_control (same splits, same baselines), and
collapse_diagnostics' metric functions.
"""

import argparse
import csv

import numpy as np
import wandb
from loguru import logger

from checkpoints import EXTERNAL_CHECKPOINTS
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

CHECKPOINT_ID = "ae1-external"
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
    assert CHECKPOINT_ID in EXTERNAL_CHECKPOINTS, f"{CHECKPOINT_ID} not registered"
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
                config=dict(encoder=CHECKPOINT_ID, property=prop, seed=seed, external=True),
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
            config=dict(encoder=CHECKPOINT_ID, external=True), job_type="collapse_diagnostics", reinit=True,
        )
        log_dict = dict(eff_rank=eff_rank, within_protein_cos=within_cos,
                         across_protein_cos=across_cos, n_dead_dims=len(dead_dims))
        log_dict.update({f"dim_std_{i}": float(s) for i, s in enumerate(dim_stds)})
        run.log(log_dict)
        run.finish()

    return row, flag


def print_interpretation_note(probe_rows, latent_dim: int, collapse_flag: str) -> None:
    by_prop = {r["property"]: r for r in probe_rows}
    aa, ss = by_prop["aa"], by_prop["ss"]
    aa_nails_it = aa["probe_score"] > aa["baseline"] + 0.5
    ss_at_baseline = abs(ss["probe_score"] - ss["baseline"]) < 0.03

    lines = ["", "=" * 70, "AE1 EXTERNAL COMPARISON -- INTERPRETATION NOTE", "=" * 70]
    if latent_dim != 8:
        lines.append(
            f"AE1 latent_dim={latent_dim} != 8 -- EVERY cross-comparison below is "
            f"capacity-confounded. The 8-dim rows (your own 8 checkpoints) remain the "
            f"fair comparison among each other; AE1 numbers are not directly comparable."
        )
    else:
        lines.append(
            f"AE1 latent_dim=8, matching your 8 checkpoints exactly -- NOT capacity-confounded. "
            f"This is a fair probe comparison, differing only in training corpus (AFDB vs PDB) "
            f"and training length (640k vs 100k steps)."
        )
    if aa_nails_it and ss_at_baseline:
        lines.append(
            f"AE1 (SOTA, 640k steps) also nails AA (acc={aa['probe_score']:.4f} vs "
            f"baseline={aa['baseline']:.4f}) and sits at the SS baseline "
            f"(acc={ss['probe_score']:.4f} vs baseline={ss['baseline']:.4f}). This confirms the "
            f"sequence-vs-structure dissociation your ca-cond/fully-latent checkpoints showed is "
            f"an OBJECTIVE property of this AE architecture/objective -- not an artifact of your "
            f"100k-step training budget."
        )
    else:
        lines.append(
            f"AE1 does NOT cleanly reproduce the 'nails AA, sits at SS baseline' pattern "
            f"(AA acc={aa['probe_score']:.4f} vs baseline={aa['baseline']:.4f}, "
            f"SS acc={ss['probe_score']:.4f} vs baseline={ss['baseline']:.4f}) -- worth digging "
            f"into rather than assuming your under-training explains ca-cond/fully-latent's pattern."
        )
    lines.append(f"Collapse diagnostic flag: {collapse_flag}.")
    lines.append("=" * 70)
    for line in lines:
        logger.info(line)


def main(seed: int = SEED, use_wandb: bool = True):
    probe_rows, latent_dim = run_probes(seed=seed, use_wandb=use_wandb)
    _collapse_row, collapse_flag = run_collapse(use_wandb=use_wandb)
    print_interpretation_note(probe_rows, latent_dim, collapse_flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, use_wandb=not args.no_wandb)
