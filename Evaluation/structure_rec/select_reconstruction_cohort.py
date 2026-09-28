"""Select a deterministic random reconstruction cohort.

The historical ten-protein visualization set is retained.  Proteins that do
not contain enough native atoms for the structural metrics are exported and
recorded, but are not counted towards the requested number of scored chains.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))

from data_pool import build_protein_pool, eval_transform_compose  # noqa: E402


HISTORICAL_IDS = (
    "1kfm_A",
    "1nkw_M",
    "3d3r_A",
    "5a13_G",
    "5fli_A",
    "5l0o_A",
    "5xr7_A",
    "6o14_A",
    "7cnw_A",
    "8cbv_A",
)
BACKBONE_ATOMS = (0, 1, 2, 4)  # N, CA, C and O in Atom37 order.


def _is_structurally_scorable(graph) -> tuple[bool, str]:
    mask = graph.coord_mask.bool()
    length = int(mask.shape[0])
    if not bool(mask[:, list(BACKBONE_ATOMS)].all()):
        return False, "incomplete N/CA/C/O backbone"
    if int(mask.sum()) <= 4 * length:
        return False, "no native side-chain heavy atoms"
    return True, ""


def select(args: argparse.Namespace) -> None:
    pool = build_protein_pool(seed=args.pool_seed)
    id_to_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    transform = eval_transform_compose()

    missing = sorted(set(HISTORICAL_IDS) - set(id_to_index))
    if missing:
        raise ValueError(f"historical proteins missing from held-out pool: {missing}")

    scored_ids: list[str] = []
    retained_unscored: dict[str, str] = {}
    for protein_id in HISTORICAL_IDS:
        graph = transform(pool.dataset[id_to_index[protein_id]])
        valid, reason = _is_structurally_scorable(graph)
        if valid:
            scored_ids.append(protein_id)
        else:
            retained_unscored[protein_id] = reason

    candidates = [
        protein_id
        for protein_id in pool.protein_ids
        if protein_id not in HISTORICAL_IDS
    ]
    random.Random(args.selection_seed).shuffle(candidates)
    rejected_random: dict[str, str] = {}
    for protein_id in candidates:
        if len(scored_ids) >= args.scored_count:
            break
        graph = transform(pool.dataset[id_to_index[protein_id]])
        valid, reason = _is_structurally_scorable(graph)
        if valid:
            scored_ids.append(protein_id)
        else:
            rejected_random[protein_id] = reason

    if len(scored_ids) != args.scored_count:
        raise RuntimeError(
            f"only found {len(scored_ids)} structurally scorable proteins; "
            f"requested {args.scored_count}"
        )

    export_ids = scored_ids + [
        protein_id
        for protein_id in HISTORICAL_IDS
        if protein_id in retained_unscored
    ]
    result = {
        "protocol": (
            "Historical visualization proteins retained; remaining proteins "
            "sampled without replacement from the frozen held-out pool. "
            "Structural scoring requires complete N/CA/C/O and at least one "
            "native side-chain heavy atom."
        ),
        "pool_seed": args.pool_seed,
        "selection_seed": args.selection_seed,
        "requested_scored_proteins": args.scored_count,
        "historical_ids": list(HISTORICAL_IDS),
        "scored_ids": scored_ids,
        "export_ids": export_ids,
        "retained_unscored": retained_unscored,
        "random_candidates_rejected_before_completion": rejected_random,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-count", type=int, default=100)
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20260920)
    return parser.parse_args()


if __name__ == "__main__":
    select(parse_args())
