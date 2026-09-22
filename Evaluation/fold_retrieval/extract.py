"""Extracts 8-dim per-residue latents (8 in-house encoders + random-init) for
StructTokenBench's OWN Homo (remote homology) protein set -- NOT the
LaProteina val+test pool used by Evaluation/linear_probing,knn_purity,
latent_viz. Per the corrected Part 3 instructions: the training-free
retrieval test must be directly comparable to STB's own trained-classifier
AUROC, which means using STB's own proteins, SCOP fold labels (45-class
scheme), and train/valid/test splits -- not a different, uncontrolled pool.

Protein set + fold labels: reuses StructTokenBench's own already-resolved
cache (struct_token_bench_release_data/data/structural/remote_homology/
processed_structured_{split}, for each of the 5 TAPE splits: train, valid,
test_fold_holdout, test_family_holdout, test_superfamily_holdout) -- the
exact structures (pdb_id/chain_id, already parsed to atom37) and labels the
real StructTokenBench Homo runs used. The 45-class fold filter/remap
(labels_to_filter/labels_mapping below) is copied verbatim from
StructTokenBench/src/dataset/base.py's
additional_label_filtering_for_TAPE_homo, so the label space matches exactly
(STB's num_labels=45 classifier head).

Structure parsing: WrappedLaProteinaTokenizer._parse_graph(pdb_path, chain_id)
-- LaProteina's own pipeline, the exact code path the real Homo runs used to
go from a raw mmCIF file to an encoder-ready graph (already nm-scaled,
centered, chain-breaks added -- see its docstring). Parsed once per unique
(pdb_id, chain_id) even though a chain can carry multiple SCOP domain
entries with different fold labels (each such entry is still recorded as
its own "protein" row below, sharing the one whole-chain extraction --
domain sub-ranges aren't separated out; flagged in the results metadata).

Streamed, bounded-concurrency multiprocessing (imap_unordered, each parsed
graph encoded by every model and discarded immediately) -- NOT a blocking
Pool.starmap() gathering thousands of graphs at once, which is what broke
StructTokenBench's own BindBio precompute earlier this session (mmap-region /
ancdata exhaustion under the "file_system"/"file_descriptor" sharing
strategies respectively -- see StructTokenBench/src/dataset/base.py's
_resolve_and_parse_worker for the actual fix). Here only ~9 encoders x 2
pooling modes x 8 floats accumulates per protein, never full graphs.
"""

import argparse
import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np
import torch
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STB_ROOT = REPO_ROOT / "StructTokenBench"
sys.path.insert(0, str(REPO_ROOT))  # proteinfoundation, openfold
sys.path.insert(0, str(STB_ROOT / "src"))
sys.path.insert(0, str(STB_ROOT / "src" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "Evaluation" / "linear_probing"))

os.chdir(STB_ROOT)  # relative paths below (struct_token_bench_release_data/...) match run_eval.sh's cwd

from checkpoints import ALL_CHECKPOINTS, load_frozen_encoder  # noqa: E402
from dataset.base import resolve_chain_id_cached  # noqa: E402
from tokenizer import WrappedLaProteinaTokenizer  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parent
LATENTS_DIR = OUT_DIR / "latents_homo"
PROCESSED_DIR = STB_ROOT / "struct_token_bench_release_data" / "data" / "structural" / "remote_homology"
PDB_DATA_DIR = STB_ROOT / "pdb_data" / "mmcif_files"  # matches run_eval.sh's data.pdb_data_dir=$PDB_DATA_ROOT/mmcif_files/;

SPLITS = ["train", "valid", "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"]

LABELS_TO_FILTER = set([
    22, 36, 47, 51, 73, 77, 78, 84, 88, 90, 126, 153, 176, 295,
    0, 3, 21, 39, 45, 59, 70, 97, 179,
    26, 49, 60, 81, 95, 113, 124, 133, 143, 178,
    13, 14, 18, 42, 52, 56, 61, 91, 132, 135, 180, 246,
])
LABELS_MAPPING = {x: i for i, x in enumerate(sorted(LABELS_TO_FILTER))}


def load_protein_list():
    """Returns list of dicts: {sid, pdb_id, chain_id, fold_label45, split}."""
    rows = []
    for split in SPLITS:
        data = torch.load(str(PROCESSED_DIR / f"processed_structured_{split}"), weights_only=False)
        for item in data:
            raw_fold = item["fold_label"]
            if raw_fold not in LABELS_TO_FILTER:
                continue
            rows.append(dict(
                sid=item["id"].decode("utf-8") if isinstance(item["id"], bytes) else item["id"],
                pdb_id=item["pdb_id"], chain_id=item["chain_id"],
                fold_label45=LABELS_MAPPING[raw_fold], split=split,
            ))
    return rows


def _parse_one(args):
    """Runs in a worker process. Returns (pdb_id, chain_id, graph_or_None, error_or_None).

    Non-standard residues are excluded here (not just caught-and-skipped) --
    they don't fail gracefully downstream: the encoder's feature factory
    indexes residue-type embeddings assuming the standard-20 vocabulary, and
    an out-of-range index triggers a CUDA device-side assert that poisons
    the whole CUDA context (unrecoverable in-process, unlike a normal
    exception) -- same failure class noted for AE1 on other tasks
    (tokenizer.py's force_remove_non_standard_residues). SCOP domains are
    old (2009 SCOP release) and include some non-standard/modified residues
    the 8 in-house checkpoints were never trained to handle either.
    """
    pdb_id, chain_id = args
    try:
        pdb_path = str(PDB_DATA_DIR / "mmcif_files" / f"{pdb_id}.cif")
        resolved_chain = resolve_chain_id_cached(pdb_path, chain_id, str(PDB_DATA_DIR))
        graph = WrappedLaProteinaTokenizer._parse_graph(pdb_path, resolved_chain)
        if any(r not in WrappedLaProteinaTokenizer.STANDARD_RESIDUES for r in graph.residues):
            return pdb_id, chain_id, None, "non-standard residue(s) present -- excluded"
        return pdb_id, chain_id, graph, None
    except Exception as e:  # noqa: BLE001 -- one bad structure must not kill the whole run
        return pdb_id, chain_id, None, repr(e)


def extract(num_workers: int = 8, limit: int = None, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_protein_list()
    logger.info(f"Protein rows (45-class fold, all splits): {len(rows)}")
    unique_pairs = sorted(set((r["pdb_id"], r["chain_id"]) for r in rows))
    if limit:
        unique_pairs = unique_pairs[:limit]
        keep_pairs = set(unique_pairs)
        rows = [r for r in rows if (r["pdb_id"], r["chain_id"]) in keep_pairs]
    logger.info(f"Unique (pdb_id, chain_id) structures to parse: {len(unique_pairs)}")

    checkpoint_ids = list(ALL_CHECKPOINTS)  # already includes ae1-external + random-init
    logger.info(f"Encoders: {checkpoint_ids}")
    encoders = {c: load_frozen_encoder(c, device=device) for c in checkpoint_ids}

    # protein_key ("{pdb}_{chain}") -> {checkpoint_id: {"mean": [8], "max": [8]}}
    pooled = {}
    n_failed = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        for i, (pdb_id, chain_id, graph, err) in enumerate(pool.imap_unordered(_parse_one, unique_pairs, chunksize=1)):
            protein_key = f"{pdb_id}_{chain_id}"
            if graph is None:
                n_failed += 1
                if n_failed <= 20:
                    logger.warning(f"  parse failed for {protein_key}: {err}")
                continue

            batch = dense_padded_from_data_list([graph])
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
            batch = batch.to(device)
            mask = batch["mask"][0]

            per_encoder = {}
            for checkpoint_id, encoder in encoders.items():
                with torch.no_grad():
                    mean_latent = encoder(batch)["mean"][0][mask]  # [n_res, 8]
                per_encoder[checkpoint_id] = dict(
                    mean=mean_latent.mean(dim=0).cpu().numpy(),
                    max=mean_latent.max(dim=0).values.cpu().numpy(),
                )
            pooled[protein_key] = per_encoder

            if (i + 1) % 200 == 0 or (i + 1) == len(unique_pairs):
                logger.info(f"  ...{i + 1}/{len(unique_pairs)} structures encoded ({n_failed} failed so far)")

    logger.info(f"Done: {len(pooled)} structures encoded, {n_failed} failed to parse.")

    torch.save(pooled, LATENTS_DIR / "pooled_latents.pt")
    torch.save(rows, LATENTS_DIR / "protein_rows.pt")
    logger.info(f"Wrote {LATENTS_DIR / 'pooled_latents.pt'} and protein_rows.pt")
    return pooled, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="debug: only parse first N unique structures")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    extract(num_workers=args.num_workers, limit=args.limit, device=args.device)
