"""Cache missing main-100k residue latents, then render the full t-SNE panel.

Safe to resume: each encoder cache is written separately and existing caches
are skipped.  Encoders are also loaded one at a time to keep GPU memory bounded.
"""

import argparse
import pathlib
import sys

import torch
from loguru import logger

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path[:0] = [str(REPO_ROOT), str(LINEAR_PROBING_DIR), str(pathlib.Path(__file__).resolve().parent)]

from publication_100k import (  # noqa: E402
    ADDITIONAL_PUBLICATION_100K,
    PUBLICATION_100K_ORDER,
    PUBLICATION_SHORT_LABELS,
    register_publication_checkpoints,
)


def main(skip_tsne: bool = False):
    register_publication_checkpoints()

    from data_pool import build_protein_pool
    from extract_latents import LATENTS_DIR, extract

    missing = [
        checkpoint_id
        for checkpoint_id in ADDITIONAL_PUBLICATION_100K
        if not (LATENTS_DIR / f"{checkpoint_id}.pt").exists()
    ]
    logger.info(f"Missing publication-100k latent caches ({len(missing)}): {missing}")
    if missing:
        if not torch.cuda.is_available():
            raise RuntimeError("Missing latent caches require a CUDA node; refusing an accidental CPU extraction.")
        pool = build_protein_pool()
        for position, checkpoint_id in enumerate(missing, start=1):
            logger.info(f"[{position}/{len(missing)}] Extracting {checkpoint_id}")
            extract(pool=pool, checkpoint_ids=[checkpoint_id], device="cuda")
            torch.cuda.empty_cache()

    still_missing = [
        checkpoint_id
        for checkpoint_id in PUBLICATION_100K_ORDER
        if not (LATENTS_DIR / f"{checkpoint_id}.pt").exists()
    ]
    if still_missing:
        raise RuntimeError(f"Cannot render publication t-SNE; caches still missing: {still_missing}")

    if skip_tsne:
        logger.info("All publication caches are ready; --skip-tsne requested.")
        return

    # Import after registration so run_tsne sees the extended dictionary.
    import run_tsne

    run_tsne.SHORT_LABEL.update(PUBLICATION_SHORT_LABELS)
    run_tsne.run(
        checkpoint_ids=PUBLICATION_100K_ORDER,
        n_points=15_000,
        perplexity=30.0,
        seed=42,
        max_iter=1_000,
        force=False,
        output_tag="publication-100k",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()
    main(skip_tsne=args.skip_tsne)

