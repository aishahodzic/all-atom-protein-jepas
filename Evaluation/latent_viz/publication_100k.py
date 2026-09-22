"""Final-checkpoint registry for the main 100k latent-space comparison.

The original linear-probing registry predates several completed experiments.
This module adds those experiments without changing the default historical
linear-probe sweep.  All JEPA-family entries are loaded through the existing
online/student-encoder loader; their latent width is inferred from checkpoint
weights (8 or 64), so predictor/masking differences do not affect loading.
"""

from typing import Dict, Tuple


ADDITIONAL_PUBLICATION_100K: Dict[str, Tuple[str, str]] = {
    "random-init-64": ("random-init-64/step0.ckpt", "jepa"),
    "jepa-rand70-d64": ("bottle-inspect/64dim/jepa/100k.ckpt", "jepa"),
    "jepa-selfattn-rand50": (
        "bottle-inspect/self-attn/jepa/random_masking_50pct/100k.ckpt", "jepa"
    ),
    "jepa-selfattn-rand70": (
        "bottle-inspect/self-attn/jepa/random_masking_70pct/100k.ckpt", "jepa"
    ),
    "jepa-new-predictor": ("jepa/new_predictor/100k.ckpt", "jepa"),
    "le-jepa-rand70-d64": ("bottle-inspect/64dim/le-jepa/100k.ckpt", "jepa"),
    "le-jepa-new-predictor": ("le-jepa/new_predictor/100k.ckpt", "jepa"),
    "le-jepa-spatial3d-d64": (
        "le-jepa/spatial3d_block_dim64_new_predictor/100k.ckpt", "jepa"
    ),
    "le-jepa-chem-exp1": (
        "le-jepa/chem_masking/exp1_aa_partial50/100k.ckpt", "jepa"
    ),
    "le-jepa-chem-exp3": (
        "le-jepa/chem_masking/exp3_noaa_sc50/100k.ckpt", "jepa"
    ),
}


PUBLICATION_100K_ORDER = [
    "ca-cond",
    "fully-latent",
    "random-init",
    "random-init-64",
    "jepa-rand-mask-50",
    "jepa-rand-mask-70",
    "jepa-mutation",
    "jepa-rand70-d64",
    "jepa-selfattn-rand50",
    "jepa-selfattn-rand70",
    "jepa-new-predictor",
    "le-jepa-rand-mask-50",
    "le-jepa-rand-mask-70",
    "le-jepa-mutation",
    "le-jepa-rand70-d64",
    "le-jepa-new-predictor",
    "le-jepa-spatial3d-d64",
    "le-jepa-chem-exp1",
    "le-jepa-chem-exp3",
    "ae1-external",
]


PUBLICATION_SHORT_LABELS = {
    "random-init-64": "Random-init D64",
    "jepa-rand70-d64": "JEPA-70 D64",
    "jepa-selfattn-rand50": "JEPA-50 self-attn",
    "jepa-selfattn-rand70": "JEPA-70 self-attn",
    "jepa-new-predictor": "JEPA-50 new pred.",
    "le-jepa-rand70-d64": "LeJEPA-70 D64",
    "le-jepa-new-predictor": "LeJEPA-50 new pred.+EMA",
    "le-jepa-spatial3d-d64": "LeJEPA spatial D64",
    "le-jepa-chem-exp1": "LeJEPA chem exp1",
    "le-jepa-chem-exp3": "LeJEPA chem exp3",
}


def register_publication_checkpoints() -> None:
    """Extend the in-memory loader registry; does not alter default sweeps."""
    import checkpoints

    checkpoints.ALL_CHECKPOINTS.update(ADDITIONAL_PUBLICATION_100K)
