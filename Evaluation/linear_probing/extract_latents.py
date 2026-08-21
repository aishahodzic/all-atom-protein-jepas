"""B1: one pass over the protein pool computes labels once (encoder-agnostic)
and each requested encoder's frozen latents (mean, full-context, no masking
-- see checkpoints.load_frozen_encoder + tokenizer.py:930's "deterministic
latent, comparable across AE and JEPA checkpoints"), caching each to its own
file so later runs (or probe.py alone) don't re-run the encoder.

One protein at a time (batch size 1): every residue in a single-protein
batch is real, so there is no padding to exclude and the mask-handling from
A3 is satisfied.
"""

import pathlib
from typing import Dict, List, Optional

import torch
from loguru import logger
from tqdm import tqdm

from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list

from checkpoints import CHECKPOINTS, load_frozen_encoder
from data_pool import ProteinPool, build_protein_pool, eval_transform_compose
from labels import compute_labels

LATENTS_DIR = pathlib.Path(__file__).resolve().parent / "latents"
LABELS_CACHE_PATH = LATENTS_DIR / "labels.pt"


def _encoder_cache_path(checkpoint_id: str) -> pathlib.Path:
    return LATENTS_DIR / f"{checkpoint_id}.pt"


def extract(
    pool: Optional[ProteinPool] = None,
    checkpoint_ids: Optional[List[str]] = None,
    device: Optional[str] = None,
    force: bool = False,
):
    """Returns (pool, labels_by_protein, latents_by_checkpoint_by_protein)."""
    checkpoint_ids = list(CHECKPOINTS) if checkpoint_ids is None else checkpoint_ids
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)

    need_labels = force or not LABELS_CACHE_PATH.exists()
    encoders_to_run = [c for c in checkpoint_ids if force or not _encoder_cache_path(c).exists()]

    labels_by_protein: Dict[str, dict] = {} if need_labels else torch.load(LABELS_CACHE_PATH, weights_only=False)
    latents_by_checkpoint: Dict[str, Dict[str, torch.Tensor]] = {
        c: ({} if c in encoders_to_run else torch.load(_encoder_cache_path(c), weights_only=False))
        for c in checkpoint_ids
    }

    if not need_labels and not encoders_to_run:
        logger.info("Labels + all requested encoders already cached, nothing to extract.")
        pool = pool or build_protein_pool()
        return pool, labels_by_protein, latents_by_checkpoint

    pool = pool or build_protein_pool()
    logger.info(
        f"Extracting: labels={'yes' if need_labels else 'cached'}, "
        f"encoders={encoders_to_run or 'none (all cached)'}"
    )

    encoders = {c: load_frozen_encoder(c, device=device) for c in encoders_to_run}
    transform = eval_transform_compose()

    for i in tqdm(range(len(pool.dataset)), desc="Extracting"):
        protein_id = pool.protein_ids[i]
        graph = pool.dataset[i]  # raw: Angstrom, uncentered, atom37-openfold-ordered

        if need_labels:
            labels_by_protein[protein_id] = compute_labels(graph)

        if not encoders_to_run:
            continue

        graph = transform(graph)  # in-place -> nm, centered, chain_breaks_per_residue added
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
        batch = batch.to(device)

        for checkpoint_id, encoder in encoders.items():
            with torch.no_grad():
                mean = encoder(batch)["mean"]  # [1, n, 8]
            mask = batch["mask"][0]
            latents_by_checkpoint[checkpoint_id][protein_id] = mean[0][mask].cpu()

    if need_labels:
        torch.save(labels_by_protein, LABELS_CACHE_PATH)
        logger.info(f"Cached labels for {len(labels_by_protein)} proteins -> {LABELS_CACHE_PATH}")
        n_res = {"probe_train": 0, "probe_test": 0}
        n_prot = {"probe_train": 0, "probe_test": 0}
        for pid, lab in labels_by_protein.items():
            split = pool.probe_split[pid]
            n_res[split] += len(lab["aa"])
            n_prot[split] += 1
        logger.info(
            f"Final counts -- probe_train: {n_prot['probe_train']} proteins / {n_res['probe_train']} residues; "
            f"probe_test: {n_prot['probe_test']} proteins / {n_res['probe_test']} residues."
        )
    for checkpoint_id in encoders_to_run:
        out_path = _encoder_cache_path(checkpoint_id)
        torch.save(latents_by_checkpoint[checkpoint_id], out_path)
        logger.info(
            f"Cached latents for {checkpoint_id}: "
            f"{len(latents_by_checkpoint[checkpoint_id])} proteins -> {out_path}"
        )

    return pool, labels_by_protein, latents_by_checkpoint
