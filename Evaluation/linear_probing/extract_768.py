"""B1 (768-dim variant): same full-context pass as extract_latents.py, but
captures the pre-bottleneck 768-dim hidden state instead of the 8-dim mean --
via a forward pre-hook on encoder.latent_decoder_mean_n_log_scale, whose
input *is* the trunk's final-layer output (nothing else touches it in
between -- see EncoderTransformer.forward). One forward pass produces this
alongside the 8-dim mean; we just don't bother reading the 8-dim output here
since it's already cached separately.

Cached in latents_768/ (separate from the 8-dim latents/ cache). Reuses the
shared labels cache (latents/labels.pt, via extract_latents.LABELS_CACHE_PATH)
unmodified -- labels don't depend on which tap point you read the encoder at.
"""

import pathlib
from typing import Dict, List, Optional

import torch
from loguru import logger
from tqdm import tqdm

from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list

from checkpoints import load_frozen_encoder
from data_pool import ProteinPool, build_protein_pool, eval_transform_compose

LATENTS_768_DIR = pathlib.Path(__file__).resolve().parent / "latents_768"


def _encoder_cache_path(checkpoint_id: str) -> pathlib.Path:
    return LATENTS_768_DIR / f"{checkpoint_id}.pt"


def extract_768(
    checkpoint_ids: List[str],
    pool: Optional[ProteinPool] = None,
    device: Optional[str] = None,
    force: bool = False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    LATENTS_768_DIR.mkdir(parents=True, exist_ok=True)

    encoders_to_run = [c for c in checkpoint_ids if force or not _encoder_cache_path(c).exists()]
    latents_by_checkpoint: Dict[str, Dict[str, torch.Tensor]] = {
        c: ({} if c in encoders_to_run else torch.load(_encoder_cache_path(c), weights_only=False))
        for c in checkpoint_ids
    }

    if not encoders_to_run:
        logger.info("All requested 768-dim caches already present, nothing to extract.")
        pool = pool or build_protein_pool()
        return pool, latents_by_checkpoint

    pool = pool or build_protein_pool()
    logger.info(f"Extracting 768-dim hidden states for: {encoders_to_run}")

    encoders = {c: load_frozen_encoder(c, device=device) for c in encoders_to_run}

    captured = {}

    def make_hook(checkpoint_id):
        def hook(module, args):
            captured[checkpoint_id] = args[0]  # input to latent_decoder_mean_n_log_scale, [1, n, 768]
        return hook

    hook_handles = [
        encoder.latent_decoder_mean_n_log_scale.register_forward_pre_hook(make_hook(checkpoint_id))
        for checkpoint_id, encoder in encoders.items()
    ]

    transform = eval_transform_compose()

    for i in tqdm(range(len(pool.dataset)), desc="Extracting 768-dim"):
        protein_id = pool.protein_ids[i]
        graph = transform(pool.dataset[i])  # raw -> nm, centered, chain_breaks_per_residue added
        batch = dense_padded_from_data_list([graph])
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
        batch = batch.to(device)

        for checkpoint_id, encoder in encoders.items():
            captured.pop(checkpoint_id, None)
            with torch.no_grad():
                encoder(batch)  # hook captures the 768-dim hidden state as a side effect
            hidden = captured[checkpoint_id]  # [1, n, 768]
            mask = batch["mask"][0]
            latents_by_checkpoint[checkpoint_id][protein_id] = hidden[0][mask].cpu()

    for handle in hook_handles:
        handle.remove()

    for checkpoint_id in encoders_to_run:
        out_path = _encoder_cache_path(checkpoint_id)
        torch.save(latents_by_checkpoint[checkpoint_id], out_path)
        logger.info(
            f"Cached 768-dim latents for {checkpoint_id}: "
            f"{len(latents_by_checkpoint[checkpoint_id])} proteins -> {out_path}"
        )

    return pool, latents_by_checkpoint
