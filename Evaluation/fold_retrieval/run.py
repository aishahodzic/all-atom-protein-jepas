"""Orchestrates Part 3 (training-free fold retrieval) and prints the combined
Part 1 + 2 + 3 picture: STB's trained-classifier AUROC (per remoteness split,
including the random-init floor from Part 1) vs. this training-free k-NN
retrieval purity on the identical proteins -- same pooling (mean primary,
max secondary), same 45-class SCOP fold labels, same train/valid -> test
split structure.

Usage: python run.py [--no-wandb] [--device cuda]
"""

import argparse
import csv
import pathlib

import torch
import wandb
from loguru import logger

from retrieval import (
    DATABASE_SPLITS,
    K_VALUES,
    SEED,
    TEST_SPLITS,
    _vectors_for_split,
    knn_purity_against_database,
    load_data,
    majority_baseline,
    random_control,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "fold_retrieval_results.csv"
CSV_FIELDS = ["encoder", "pool_mode", "test_split", "k", "purity", "chance_baseline",
              "random_control", "n_query", "n_database"]

STB_SUMMARY_CSV = pathlib.Path(__file__).resolve().parents[2] / "StructTokenBench" / "struct_token_bench_logs" / "summary_results.csv"
STB_BRANCH_NAME = {
    "ca-cond": "ca-cond-baseline",
}  # everything else matches 1:1


def load_stb_auroc():
    """{(encoder, split): auroc} for task=Homo, metric=auroc, from STB's own summary CSV."""
    out = {}
    if not STB_SUMMARY_CSV.exists():
        logger.warning(f"No STB summary CSV at {STB_SUMMARY_CSV} -- Part1/2 comparison will be skipped.")
        return out
    with open(STB_SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            if row["task"] != "Homo" or row["metric"] != "auroc":
                continue
            branch = row["branch"]
            for split in TEST_SPLITS:
                val = row.get(split, "")
                if val:
                    out[(branch, split)] = float(val)
    return out


def run_all(device: str = None, use_wandb: bool = True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    pooled, rows = load_data()
    checkpoint_ids = sorted({e for r in rows for e in pooled[r["protein_key"]].keys()})
    logger.info(f"Encoders with pooled latents: {checkpoint_ids}")
    logger.info(f"Total usable protein rows (structure successfully encoded): {len(rows)}")
    for split in DATABASE_SPLITS + TEST_SPLITS:
        n = sum(1 for r in rows if r["split"] == split)
        logger.info(f"  {split}: {n} rows")

    stb_auroc = load_stb_auroc()

    csv_rows = []
    results_table = {}  # (encoder, pool_mode, split) -> purity@10

    for pool_mode in ["mean", "max"]:
        for encoder in checkpoint_ids:
            db_X_parts, db_y_parts, db_keys_parts = [], [], []
            for split in DATABASE_SPLITS:
                X, y, keys = _vectors_for_split(pooled, rows, split, encoder, pool_mode)
                db_X_parts.append(X); db_y_parts.append(y); db_keys_parts.extend(keys)
            import numpy as np
            db_X = np.concatenate(db_X_parts, axis=0)
            db_y = np.concatenate(db_y_parts, axis=0)
            db_keys = db_keys_parts

            for split in TEST_SPLITS:
                qX, qy, qkeys = _vectors_for_split(pooled, rows, split, encoder, pool_mode)
                if len(qy) == 0 or len(db_y) == 0:
                    continue
                purity = knn_purity_against_database(qX, qy, qkeys, db_X, db_y, db_keys, K_VALUES, device=device)
                chance = majority_baseline(qy)
                rc = random_control(len(qy), len(db_y), qX.shape[-1], qy, db_y, K_VALUES, device=device, seed=SEED)

                for k in K_VALUES:
                    csv_rows.append(dict(
                        encoder=encoder, pool_mode=pool_mode, test_split=split, k=k,
                        purity=purity[k], chance_baseline=chance, random_control=rc[k],
                        n_query=len(qy), n_database=len(db_y),
                    ))
                results_table[(encoder, pool_mode, split)] = purity[K_VALUES[0]]

                logger.info(f"{encoder:22s} {pool_mode:4s} {split:26s} purity@10={purity[K_VALUES[0]]:.4f} "
                            f"chance={chance:.4f} random_control={rc[K_VALUES[0]]:.4f} "
                            f"(n_query={len(qy)}, n_db={len(db_y)})")

                if use_wandb:
                    run = wandb.init(
                        project="protein_linear_probing", name=f"{encoder}_{pool_mode}_{split}_foldretrieval",
                        config=dict(encoder=encoder, pool_mode=pool_mode, split=split, seed=SEED),
                        job_type="fold_retrieval", reinit=True,
                    )
                    run.log(dict(purity_k10=purity[K_VALUES[0]], chance_baseline=chance,
                                  random_control_k10=rc[K_VALUES[0]], n_query=len(qy), n_database=len(db_y)))
                    run.finish()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Wrote {len(csv_rows)} rows -> {CSV_PATH}")

    print_summary(checkpoint_ids, results_table, stb_auroc)
    return csv_rows


def print_summary(checkpoint_ids, results_table, stb_auroc):
    lines = ["", "=" * 100, "PART 3 vs PART 1/2 -- training-free retrieval purity@10 (mean-pool) vs STB classifier AUROC", "=" * 100]
    header = f"{'encoder':22s} " + "".join(f"{s.replace('test_','').replace('_holdout',''):>16s}_ret {s.replace('test_','').replace('_holdout',''):>10s}_auroc" for s in TEST_SPLITS)
    lines.append(header)
    for encoder in checkpoint_ids:
        branch = STB_BRANCH_NAME.get(encoder, encoder)
        cells = [f"{encoder:22s}"]
        for split in TEST_SPLITS:
            ret = results_table.get((encoder, "mean", split))
            auroc = stb_auroc.get((branch, split))
            ret_s = f"{ret:.4f}" if ret is not None else "n/a"
            auroc_s = f"{auroc:.4f}" if auroc is not None else "n/a"
            cells.append(f"{ret_s:>20s} {auroc_s:>15s}")
        lines.append(" ".join(cells))
    lines.append("=" * 100)

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 100)
    for encoder in checkpoint_ids:
        branch = STB_BRANCH_NAME.get(encoder, encoder)
        for split in TEST_SPLITS:
            ret = results_table.get((encoder, "mean", split))
            auroc = stb_auroc.get((branch, split))
            if ret is None or auroc is None:
                continue
        # per-encoder summary line using fold_holdout (most remote) as headline
    fold_ret = {e: results_table.get((e, "mean", "test_fold_holdout")) for e in checkpoint_ids}
    fold_auroc = {e: stb_auroc.get((STB_BRANCH_NAME.get(e, e), "test_fold_holdout")) for e in checkpoint_ids}
    if fold_ret.get("random-init") is not None:
        lines.append(f"random-init floor (fold_holdout): retrieval={fold_ret['random-init']:.4f}, "
                      f"STB AUROC={fold_auroc.get('random-init', float('nan')):.4f}")
        for e in checkpoint_ids:
            if e == "random-init" or fold_ret.get(e) is None:
                continue
            above_floor = fold_ret[e] - fold_ret["random-init"]
            lines.append(f"  {e:22s} retrieval margin over random-init floor: {above_floor:+.4f} "
                         f"({'ABOVE floor -- some learned fold structure' if above_floor > 0.02 else 'AT/below floor -- no fold signal beyond chance geometry'})")
    lines.append("=" * 100)
    for line in lines:
        logger.info(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_all(device=args.device, use_wandb=not args.no_wandb)
