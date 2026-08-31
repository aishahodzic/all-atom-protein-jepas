"""Step-matched 128-protein 8D shortcut test: absolute vs relative predictor."""

import csv
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm


REPO = pathlib.Path(__file__).resolve().parents[2]
LINEAR = REPO / "Evaluation" / "linear_probing"
sys.path[:0] = [str(REPO), str(LINEAR)]

import checkpoints  # noqa: E402
from data_pool import build_protein_pool, eval_transform_compose  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import (  # noqa: E402
    dense_padded_from_data_list,
)
from quick_lejepa_relative_position import neighbour_stats  # noqa: E402


ABSOLUTE = "le-jepa-rand50-absolute-pos-10k"
RELATIVE = "le-jepa-rand50-relative-pos-10k"
CHECKPOINTS = {
    ABSOLUTE: pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/le_jepa_rand50_v1.3/store/"
        "le_jepa_rand50_v1.3/checkpoints/"
        "chk_epoch=00000001_step=000000010000.ckpt"
    ),
    RELATIVE: pathlib.Path(
        "/mnt/hopper-raid/aishod/runs/le_jepa_rand50_relative_positions_10k/"
        "store/le_jepa_rand50_relative_positions_10k/checkpoints/"
        "chk_epoch=00000001_step=000000010000.ckpt"
    ),
}
N_PROTEINS = 128
SEED = 42
RESULT = (
    REPO / "Evaluation" / "latent_viz" / "results" /
    "lejepa_relative_position_step_matched_quick128.csv"
)


def main() -> None:
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        entry = (str(path), "jepa")
        checkpoints.CHECKPOINTS[name] = entry
        checkpoints.ALL_CHECKPOINTS[name] = entry

    pool = build_protein_pool()
    pool_index = {pid: i for i, pid in enumerate(pool.protein_ids)}
    rng = np.random.default_rng(SEED)
    chosen = sorted(
        rng.choice(sorted(pool_index), size=N_PROTEINS, replace=False).tolist()
    )
    encoders = {
        name: checkpoints.load_frozen_encoder(name, device="cpu")
        for name in CHECKPOINTS
    }
    transform = eval_transform_compose()
    chunks = {name: [] for name in CHECKPOINTS}
    index_chunks, protein_chunks = [], []

    with torch.inference_mode():
        for protein_number, pid in enumerate(tqdm(chosen, desc="Step-matched shortcut test")):
            graph = transform(pool.dataset[pool_index[pid]])
            batch = dense_padded_from_data_list([graph])
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
            mask = batch["mask"][0]
            encoded = {
                name: encoder(batch)["mean"][0][mask].float().numpy()
                for name, encoder in encoders.items()
            }
            n = min(map(len, encoded.values()))
            for name in CHECKPOINTS:
                chunks[name].append(encoded[name][:n])
            index_chunks.append(np.arange(n, dtype=np.int64))
            protein_chunks.append(np.full(n, protein_number, dtype=np.int64))

    indices = np.concatenate(index_chunks)
    proteins = np.concatenate(protein_chunks)
    counts = np.bincount(indices)
    chance = float((counts * (counts - 1)).sum() / (len(indices) * (len(indices) - 1)))
    rows = []
    for name in CHECKPOINTS:
        values = np.concatenate(chunks[name])
        rows.append({
            "checkpoint": name,
            "checkpoint_path": str(CHECKPOINTS[name]),
            "training_steps": 10_000,
            "proteins": N_PROTEINS,
            "residues": len(values),
            "random_pair_exact_index_probability": chance,
            **neighbour_stats(values, indices, proteins),
        })

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
