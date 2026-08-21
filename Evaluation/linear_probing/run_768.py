"""Phase B: linear probes (all 4 properties) on the 768-dim pre-bottleneck
hidden state, for the 8 in-house checkpoints + random-init floor
(ae1-external excluded per explicit instruction -- it has a different
feats_seq). Reuses probe.py and extract_768.py unmodified -- the only thing
that changed is the tap point (768 vs 8) and the resulting input width,
which probe.py never hardcoded (verified in Phase A: no literal "8"
anywhere in its logic).

random-init here loads the SAME persisted weights as the 8-dim floor run
(checkpoints.RANDOM_INIT_WEIGHTS_PATH) -- one weight realization, reused,
not a second independently-seeded model.

CAUTION baked into the output below: the 768-dim probe has ~96x the
trainable parameters of the 8-dim probe (768*20 + 768*3 vs 8*20 + 8*3
weights, roughly).
"""

import argparse
import csv
import pathlib

import torch
import wandb
from loguru import logger

from checkpoints import CHECKPOINTS, RANDOM_INIT_CHECKPOINTS
from extract_768 import extract_768
from extract_latents import LABELS_CACHE_PATH
from probe import compute_random_control, run_probe

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "linear_probe_results_768.csv"
CSV_FIELDS = ["encoder", "property", "seed", "probe_score", "baseline", "random_control", "n_train", "n_test"]

CHECKPOINT_IDS = list(CHECKPOINTS) + list(RANDOM_INIT_CHECKPOINTS)  # 8 mine + random-init, NOT ae1-external
PROPERTIES_768 = ["aa", "ss", "rsa", "bfactor"]
SEED = 42


def _write_csv_rows(rows) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_8dim_results() -> dict:
    """encoder -> property -> probe_score, from the existing 8-dim CSV."""
    path = RESULTS_DIR / "linear_probe_results.csv"
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["encoder"], {})[row["property"]] = float(row["probe_score"])
    return out


def run_all(seed: int = SEED, use_wandb: bool = True):
    labels = torch.load(LABELS_CACHE_PATH, weights_only=False)
    pool, latents_768 = extract_768(checkpoint_ids=CHECKPOINT_IDS)

    random_controls = {
        prop: compute_random_control(prop, pool.protein_ids, pool.probe_split, labels, latent_dim=768, seed=seed)
        for prop in PROPERTIES_768
    }
    logger.info(f"768-dim random-latent controls (once per property): {random_controls}")

    rows = []
    for checkpoint_id in CHECKPOINT_IDS:
        for prop in PROPERTIES_768:
            result = run_probe(
                prop, pool.protein_ids, pool.probe_split, labels, latents_768[checkpoint_id],
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
                f"{checkpoint_id:22s} {prop:8s} [768-dim] score={result.probe_score:.4f} "
                f"baseline={row['baseline']}"
            )
            if use_wandb:
                run = wandb.init(
                    project="protein_linear_probing", name=f"{checkpoint_id}_{prop}_768dim_seed{seed}",
                    config=dict(encoder=checkpoint_id, property=prop, seed=seed, dim=768),
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

    existing_8dim = load_existing_8dim_results()
    rows_768 = {(r["encoder"], r["property"]): r["probe_score"] for r in rows}

    print()
    col_names = [f"{p.upper()}_8dim" for p in PROPERTIES_768] + [f"{p.upper()}_768dim" for p in PROPERTIES_768]
    header = f"{'encoder':22s} " + " ".join(f"{c:>12s}" for c in col_names)
    print(header)
    logger.info(header)
    for checkpoint_id in CHECKPOINT_IDS:
        vals_8 = [existing_8dim.get(checkpoint_id, {}).get(p, float("nan")) for p in PROPERTIES_768]
        vals_768 = [rows_768.get((checkpoint_id, p), float("nan")) for p in PROPERTIES_768]
        line = f"{checkpoint_id:22s} " + " ".join(f"{v:12.4f}" for v in vals_8 + vals_768)
        print(line)
        logger.info(line)

    note = (
        "\nNOTE: the 768-dim probe has ~96x the trainable parameters of the 8-dim probe "
        "(768*20 + 768*3 vs 8*20 + 8*3 weights). Only compare model ordering WITHIN each "
        "dimensionality (which encoder scores highest at 768-dim, which at 8-dim) -- 768 vs 8 "
        "raw magnitude differences are NOT a representation-quality claim, just evidence for "
        "whether a property survives the 8-dim bottleneck."
    )
    print(note)
    logger.info(note)

    print()
    for checkpoint_id in CHECKPOINT_IDS:
        aa_8 = existing_8dim.get(checkpoint_id, {}).get("aa")
        aa_768 = rows_768.get((checkpoint_id, "aa"))
        if aa_8 is None or aa_768 is None:
            continue
        if aa_8 < 0.3 and aa_768 > 0.5:
            line = (
                f"{checkpoint_id}: AA jumps from {aa_8:.4f} (8-dim) to {aa_768:.4f} (768-dim) -- "
                f"info exists pre-bottleneck, the 8-dim projection discards it. Reframes to "
                f"'objective allocates 8 dims away from identity', not 'identity absent'."
            )
        elif aa_8 < 0.3 and aa_768 < 0.3:
            line = (
                f"{checkpoint_id}: AA stays near baseline at 768-dim too ({aa_768:.4f}) -- "
                f"identity genuinely absent from the representation, not a bottleneck effect."
            )
        else:
            continue
        print(line)
        logger.info(line)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    run_all(seed=args.seed, use_wandb=not args.no_wandb)
