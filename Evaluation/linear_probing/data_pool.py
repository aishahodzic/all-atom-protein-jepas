"""Builds the protein pool used for linear probing: LaProteina's own held-out
val + test split (never seen by any of the 8 encoders during pretraining --
see proteinfoundation/partial_autoencoder/train.py / jepa/train.py, which
checkpoint purely on step count, not on val performance, so val is exactly
as clean as test here), combined and then re-split 80/20 BY PROTEIN into a
probe-train / probe-test set. This second split is ours alone and has
nothing to do with the encoders' own train/val/test split.

Params below mirror configs/dataset/pdb/pdb_train_ucond.yaml exactly (the
dataset config used by every one of the 8 checkpoints -- confirmed via
`defaults: - dataset: pdb/pdb_train_ucond` in both training_ae.yaml and
(on the le-jepa branch) training_jepa.yaml).
"""

import os
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, List

from loguru import logger

from proteinfoundation.datasets.pdb_data import (
    PDBDataSelector,
    PDBDataset,
    PDBDataSplitter,
    PDBLightningDataModule,
)
from proteinfoundation.datasets.transforms import (
    CenterStructureTransform,
    ChainBreakPerResidueTransform,
    CoordsToNanometers,
)

DATA_DIR = pathlib.Path(os.environ["DATA_PATH"]) / "pdb_train"

EXCLUDE_IDS = [
    "9b57", "9b5p", "9b5s", "9b5n", "9b5a", "9b5k", "9b5v", "9b59", "9b5w", "9b5g", "9b5i", "9b5x", "9b5d", "9b5e", "9b5t", "9b5f",
    "9b5o", "9b58", "9b5u", "9b5c", "9b5j", "9b5b", "9b5h", "9b5q", "9b5l", "9b56", "9b5m", "9b5l", "9b55", "9b5r", "9ij9", "9iix",
    "2ezq", "1kld", "1crr", "2ezs", "2ezr", "1vve", "7ll9_G",
    "7i9a_B", "7ll9_C", "7ll9_D", "7qiz", "7suk", "7ll9_H", "9uol",
]

PROBE_TEST_FRAC = 0.2
PROBE_SPLIT_SEED = 42


def eval_transform_compose():
    """Same three transforms as StructTokenBench's WrappedLaProteinaTokenizer
    ._get_eval_transform (tokenizer.py:753-761), minus the tokenizer-wrapper
    plumbing. GlobalRotationTransform (random training augmentation) is
    deliberately excluded -- no single "correct" rotation to apply at eval
    time.

    NOT passed to PDBDataset directly -- ProteinPool.dataset yields raw
    (Angstrom, uncentered) graphs. labels.py needs Angstrom-scale coordinates
    (biotite's SS/RSA algorithms are calibrated in Angstroms), so this
    transform is applied explicitly, in-place, by extract_latents.py AFTER
    labels have been computed from the same raw graph -- see its docstring.
    """
    import torch_geometric.transforms as T

    return T.Compose([CoordsToNanometers(), CenterStructureTransform(), ChainBreakPerResidueTransform()])


def _build_datamodule() -> PDBLightningDataModule:
    dataselector = PDBDataSelector(
        data_dir=str(DATA_DIR),
        fraction=1,
        molecule_type="protein",
        experiment_types=["diffraction", "EM"],
        min_length=50,
        max_length=256,
        oligomeric_min=None,
        oligomeric_max=None,
        best_resolution=0.0,
        worst_resolution=5.0,
        has_ligands=[],
        remove_ligands=[],
        remove_non_standard_residues=True,
        remove_pdb_unavailable=True,
        exclude_ids=EXCLUDE_IDS,
    )
    datasplitter = PDBDataSplitter(
        data_dir=str(DATA_DIR),
        train_val_test=[0.98, 0.019, 0.001],
        split_type="sequence_similarity",
        split_sequence_similarity=0.5,
        overwrite_sequence_clusters=False,
    )
    return PDBLightningDataModule(
        data_dir=str(DATA_DIR),
        dataselector=dataselector,
        datasplitter=datasplitter,
        in_memory=False,
        format="cif",
        skip_unprocessed=True,
        batch_padding=True,
        sampling_mode="random",
        transforms=None,  # ProteinPool.dataset yields raw graphs -- see eval_transform_compose()
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )


@dataclass
class ProteinPool:
    dataset: PDBDataset            # union of val + test, indexable -> transformed graph
    protein_ids: List[str]         # "{pdb}_{chain}", parallel to dataset
    probe_split: Dict[str, str]    # protein_id -> "probe_train" | "probe_test"
    origin: Dict[str, str]         # protein_id -> "val" | "test" (encoder's own split, not probe_split)


def build_protein_pool(seed: int = PROBE_SPLIT_SEED) -> ProteinPool:
    dm = _build_datamodule()
    dm.setup(stage="fit")  # populates dm.dfs_splits (train/val/test all at once) + dm.val_ds
    val_ds = dm.val_ds
    test_ds = dm._get_dataset("test")  # reuses dm.dfs_splits computed above, no re-clustering

    pdb_codes = val_ds.pdb_codes + test_ds.pdb_codes
    chains = val_ds.chains + test_ds.chains
    file_names = val_ds.file_names + test_ds.file_names
    origin_list = ["val"] * len(val_ds.file_names) + ["test"] * len(test_ds.file_names)

    n_dupes = len(file_names) - len(set(file_names))
    if n_dupes:
        logger.warning(f"{n_dupes} proteins appear in both val and test splits -- deduplicating.")
        seen = set()
        keep = [i for i, fn in enumerate(file_names) if not (fn in seen or seen.add(fn))]
        pdb_codes = [pdb_codes[i] for i in keep]
        chains = [chains[i] for i in keep]
        file_names = [file_names[i] for i in keep]
        origin_list = [origin_list[i] for i in keep]

    origin = dict(zip(file_names, origin_list))

    dataset = PDBDataset(
        pdb_codes=pdb_codes,
        chains=chains,
        data_dir=str(DATA_DIR),
        transform=None,  # raw, Angstrom, uncentered -- see eval_transform_compose()
        format="cif",
        in_memory=False,
        file_names=file_names,
        num_workers=0,
    )

    rng = random.Random(seed)
    indices = list(range(len(file_names)))
    rng.shuffle(indices)
    n_test = int(round(len(indices) * PROBE_TEST_FRAC))
    probe_test_idx = set(indices[:n_test])
    probe_split = {
        file_names[i]: ("probe_test" if i in probe_test_idx else "probe_train")
        for i in range(len(file_names))
    }

    n_train = len(file_names) - len(probe_test_idx)
    logger.info(
        f"Protein pool: {len(val_ds)} val + {len(test_ds)} test "
        f"({n_dupes} overlap removed) = {len(file_names)} proteins total. "
        f"Probe split: {n_train} probe_train / {len(probe_test_idx)} probe_test proteins "
        f"(seed={seed})."
    )

    return ProteinPool(dataset=dataset, protein_ids=file_names, probe_split=probe_split, origin=origin)
