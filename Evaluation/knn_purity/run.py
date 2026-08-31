"""Orchestrates the k-NN purity sweep: every cached encoder (the 8 in-house
checkpoints + random-init + ae1-external -- same 10 rows as
linear_probing/results/collapse_diagnostics.csv) x {aa, ss} x k in
{5, 10, 50}, writing CSV + W&B, then cross-checking against the existing
8-dim linear-probe results and flagging any encoder/property where the two
methods disagree about whether the property is decodable.

Usage: python run.py [--no-wandb] [--seed SEED] [--max-queries N]
                     [--device cuda:2] [--checkpoints ID [ID ...]]
"""

import argparse
import csv
import pathlib
import sys

import torch
import wandb
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path[:0] = [str(REPO_ROOT), str(LINEAR_PROBING_DIR)]

from checkpoints import ALL_CHECKPOINTS  # noqa: E402
from knn import (  # noqa: E402
    K_VALUES,
    LABEL_SPECS,
    ResidueSelection,
    SEED,
    common_protein_ids,
    knn_purity,
    load_encoder_latents,
    load_labels,
    majority_baseline,
    random_latent_knn_purity,
    sample_query_indices,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "knn_purity_results.csv"
CSV_FIELDS = ["encoder", "property", "k", "purity", "chance_baseline", "random_control", "n_queries", "n_pool"]

# Linear-probe reference for the cross-check note (8-dim, same latents this sweep uses).
PROBE_CSV_PATH = pathlib.Path(__file__).resolve().parents[1] / "linear_probing" / "results" / "linear_probe_results.csv"
DECODABLE_MARGIN = 0.05  # "clearly above chance/baseline" threshold, applied identically to both methods


def _write_csv_rows(rows) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_probe_reference():
    """{(encoder, property): (probe_score, baseline)} from the 8-dim linear
    probe CSV, for encoders/properties this sweep can cross-check against.
    Missing on purpose for ae1-external (not part of run.py's main sweep --
    see linear_probing/run_ae1_comparison.py's docstring)."""
    ref = {}
    if not PROBE_CSV_PATH.exists():
        logger.warning(f"No linear-probe reference CSV at {PROBE_CSV_PATH} -- cross-check will be skipped.")
        return ref
    with open(PROBE_CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["property"] not in LABEL_SPECS:
                continue
            baseline = row["baseline"]
            ref[(row["encoder"], row["property"])] = (float(row["probe_score"]), float(baseline) if baseline else None)
    return ref


def run_all(
    seed: int = SEED,
    max_queries: int = 20_000,
    batch_size: int = 256,
    device: str = None,
    use_wandb: bool = True,
    checkpoint_ids=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    checkpoint_ids = list(ALL_CHECKPOINTS) if checkpoint_ids is None else list(checkpoint_ids)
    missing_caches = [
        checkpoint_id
        for checkpoint_id in checkpoint_ids
        if not (LINEAR_PROBING_DIR / "latents" / f"{checkpoint_id}.pt").exists()
    ]
    if missing_caches:
        raise FileNotFoundError(f"Missing residue-latent caches: {missing_caches}")

    labels_by_protein = load_labels()
    per_protein_by_checkpoint = {c: load_encoder_latents(c) for c in checkpoint_ids}
    latent_dims = {
        checkpoint_id: int(next(iter(per_protein.values())).shape[-1])
        for checkpoint_id, per_protein in per_protein_by_checkpoint.items()
    }
    protein_ids = common_protein_ids(per_protein_by_checkpoint)
    logger.info(f"Encoders: {checkpoint_ids}")
    logger.info(f"Latent dimensions: {latent_dims}")
    logger.info(f"Common protein pool: {len(protein_ids)} proteins.")

    rows = []
    purity_table = {}  # (encoder, property) -> {k: purity}
    chance_by_property, random_control_by_property_dim = {}, {}

    for prop, spec in LABEL_SPECS.items():
        selection = ResidueSelection(protein_ids, labels_by_protein, spec["label_key"], spec["valid_key"])
        query_indices = sample_query_indices(selection.n, max_queries=max_queries, seed=seed)
        subsampled = selection.n > max_queries
        chance = majority_baseline(selection.y[query_indices])
        chance_by_property[prop] = chance

        for latent_dim in sorted(set(latent_dims.values())):
            random_control_by_property_dim[(prop, latent_dim)] = random_latent_knn_purity(
                n_pool=selection.n, latent_dim=latent_dim, y=selection.y,
                protein_idx=selection.protein_idx, query_indices=query_indices,
                k_values=K_VALUES, device=device, batch_size=batch_size, seed=seed,
            )

        logger.info(
            f"[{prop}] n_pool={selection.n} n_queries={len(query_indices)} "
            f"(subsampled={subsampled}) chance={chance:.4f} "
            f"random_controls={{{', '.join(f'D{dim}: {random_control_by_property_dim[(prop, dim)]}' for dim in sorted(set(latent_dims.values())))}}}"
        )

        for checkpoint_id in checkpoint_ids:
            latent_dim = latent_dims[checkpoint_id]
            random_control = random_control_by_property_dim[(prop, latent_dim)]
            X = selection.build_X(per_protein_by_checkpoint[checkpoint_id])
            purity = knn_purity(
                X, selection.y, selection.protein_idx, query_indices,
                k_values=K_VALUES, device=device, batch_size=batch_size,
            )
            purity_table[(checkpoint_id, prop)] = purity

            logger.info(
                f"{checkpoint_id:22s} {prop:8s} " + " ".join(f"purity@{k}={purity[k]:.4f}" for k in K_VALUES)
            )

            for k in K_VALUES:
                rows.append(dict(
                    encoder=checkpoint_id, property=prop, k=k,
                    purity=purity[k], chance_baseline=chance, random_control=random_control[k],
                    n_queries=len(query_indices), n_pool=selection.n,
                ))

            if use_wandb:
                run = wandb.init(
                    project="protein_linear_probing", name=f"{checkpoint_id}_{prop}_knn",
                    config=dict(encoder=checkpoint_id, property=prop, seed=seed, max_queries=max_queries),
                    job_type="knn_purity", reinit=True,
                )
                log_dict = {f"purity_k{k}": purity[k] for k in K_VALUES}
                log_dict.update({f"random_control_k{k}": random_control[k] for k in K_VALUES})
                log_dict.update(chance_baseline=chance, n_queries=len(query_indices), n_pool=selection.n)
                run.log(log_dict)
                run.finish()

    _write_csv_rows(rows)
    logger.info(f"Wrote {len(rows)} rows -> {CSV_PATH}")

    print_summary_table(
        checkpoint_ids, purity_table, chance_by_property,
        random_control_by_property_dim, latent_dims,
    )
    print_interpretation_note(checkpoint_ids, purity_table, chance_by_property)
    return rows


def print_summary_table(
    checkpoint_ids, purity_table, chance_by_property,
    random_control_by_property_dim, latent_dims, k: int = 10,
):
    props = list(LABEL_SPECS)
    header = ["encoder"] + [f"{p}_purity@{k}" for p in props] + [f"{p}_chance" for p in props]
    widths = [max(len(header[i]), 22 if i == 0 else 10) for i in range(len(header))]
    lines = ["", "=" * 70, f"k-NN PURITY SUMMARY (k={k})", "=" * 70]
    lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for checkpoint_id in checkpoint_ids:
        cells = [checkpoint_id]
        for p in props:
            cells.append(f"{purity_table[(checkpoint_id, p)][k]:.4f}")
        for p in props:
            cells.append(f"{chance_by_property[p]:.4f}")
        lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    lines.append("-" * 70)
    for latent_dim in sorted(set(latent_dims.values())):
        rc_cells = [f"random-latent D{latent_dim}"] + [
            f"{random_control_by_property_dim[(p, latent_dim)][k]:.4f}"
            for p in props
        ] + [""] * len(props)
        lines.append("  ".join(c.ljust(w) for c, w in zip(rc_cells, widths)))
    lines.append("=" * 70)
    lines.append(
        "Note: random-latent control usually sits BELOW the chance line above, not at it -- chance here "
        "is the majority-class frequency (matching probe.py's baseline, for direct comparability), but "
        "purely random neighbours reproduce sum(class_freq^2) (a Gini-style expectation), which is <= the "
        "majority frequency whenever a label has >2 classes with any spread. Judge 'is this encoder doing "
        "anything real' against the random-latent control, not the chance line; use the chance line only "
        "to cross-reference against the linear-probe baselines."
    )
    lines.append("=" * 70)
    for line in lines:
        logger.info(line)


def print_interpretation_note(checkpoint_ids, purity_table, chance_by_property, k: int = 10):
    probe_ref = _load_probe_reference()
    lines = ["", "=" * 70, "INTERPRETATION NOTE -- k-NN purity vs linear probe", "=" * 70]
    any_flag = False
    for checkpoint_id in checkpoint_ids:
        for prop in LABEL_SPECS:
            purity = purity_table[(checkpoint_id, prop)][k]
            knn_margin = purity - chance_by_property[prop]
            knn_decodable = knn_margin > DECODABLE_MARGIN

            ref = probe_ref.get((checkpoint_id, prop))
            if ref is None:
                continue  # no linear-probe reference for this encoder (e.g. ae1-external)
            probe_score, probe_baseline = ref
            if probe_baseline is None:
                continue
            probe_margin = probe_score - probe_baseline
            probe_decodable = probe_margin > DECODABLE_MARGIN

            if knn_decodable != probe_decodable:
                any_flag = True
                lines.append(
                    f"[ENTANGLEMENT] {checkpoint_id:22s} {prop:4s}: k-NN purity@{k}={purity:.4f} "
                    f"(margin {knn_margin:+.4f}, {'decodable' if knn_decodable else 'at chance'}) vs "
                    f"linear probe={probe_score:.4f} (margin {probe_margin:+.4f}, "
                    f"{'decodable' if probe_decodable else 'at chance'}) -- disagree."
                )
    if not any_flag:
        lines.append("No disagreements above the +0.05 margin threshold -- k-NN purity and linear-probe "
                      "decodability agree everywhere checked.")
    lines.append(f"(Decodability threshold: margin over chance/baseline > {DECODABLE_MARGIN})")
    lines.append("=" * 70)
    for line in lines:
        logger.info(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-queries", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Residue-cache IDs to evaluate; defaults to the historical registry.",
    )
    args = parser.parse_args()
    run_all(
        seed=args.seed, max_queries=args.max_queries, batch_size=args.batch_size,
        device=args.device, use_wandb=not args.no_wandb,
        checkpoint_ids=args.checkpoints,
    )
