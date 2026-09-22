"""Test whether input rotations induce a shared, lawful transform in latent space.

Raw cosine agreement cannot distinguish harmful frame dependence from a clean
equivariant representation ``z(x R) = z(x) A_R``.  This diagnostic fits the
latent action on whole training proteins and evaluates it on disjoint proteins.
It reports orthogonal-Procrustes, affine, centered-Gram/CKA, and group-composition
tests.  No residue from a test protein is used to estimate the latent map.
"""

from __future__ import annotations

import copy
import argparse
import json
import pathlib
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric import transforms as T

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proteinfoundation.datasets.pdb_data import PDBDataset
from proteinfoundation.datasets.transforms import (
    CenterStructureTransform,
    ChainBreakPerResidueTransform,
    CoordsToNanometers,
)
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list


DATA = pathlib.Path("/mnt/hopper-raid/aishod/data/pdb_train")
CFG = REPO / "StructTokenBench/100k-checkpoints/shared/exp_config_jepa_nn130m.json"

TRAIN_IDS = [
    "2y0q_B", "6n6r_D", "2bq7_A", "5wsn_F", "6gsi_J", "6bmt_B",
    "2ybr_F", "9u6l_P", "7azy_j", "4nj6_F", "7tk8_8", "4dra_E",
]
TEST_IDS = [
    "6wnq_Q", "3znj_S", "1mn3_A", "4o9b_A", "6bm7_C", "2ns0_A",
    "8zdc_b", "4eo5_C", "3qvg_C", "6nbi_G", "3wwf_C",
    "6b8h_i", "6tz8_F", "8w9j_H",
]

CHECKPOINTS = {
    "jepa_rand70_d64_100k_online": REPO / "StructTokenBench/100k-checkpoints/bottle-inspect/64dim/jepa/100k.ckpt",
    "new_le_100k_online": REPO / "StructTokenBench/100k-checkpoints/le-jepa/new_predictor/100k.ckpt",
    "old_le_rand70_100k_online": REPO / "StructTokenBench/100k-checkpoints/le-jepa/random_masking_70pct/100k.ckpt",
}

RZ = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
RX = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
ROTATIONS = {"identity": torch.eye(3), "rz90": RZ, "rx90": RX, "rz90_rx90": RZ @ RX}


def load_encoder(path: pathlib.Path) -> EncoderTransformer:
    cfg = json.loads(CFG.read_text())["nn_ae"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    weights = {
        key.removeprefix("online_encoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("online_encoder.")
    }
    output_width = weights["latent_decoder_mean_n_log_scale.1.weight"].shape[0]
    latent_dim = output_width // 2
    cfg["latent_z_dim"] = latent_dim
    cfg["encoder"]["latent_z_dim"] = latent_dim
    cfg["decoder"]["latent_z_dim"] = latent_dim
    encoder = EncoderTransformer(**cfg)
    encoder.load_state_dict(weights)
    encoder.eval().requires_grad_(False)
    del checkpoint, weights
    return encoder


def make_graphs(ids: list[str]):
    transform = T.Compose(
        [CoordsToNanometers(), CenterStructureTransform(), ChainBreakPerResidueTransform()]
    )
    dataset = PDBDataset(
        pdb_codes=ids,
        chains=None,
        data_dir=str(DATA),
        transform=transform,
        format="cif",
        in_memory=False,
        file_names=ids,
        num_workers=0,
    )
    return [dataset[i] for i in range(len(dataset))]


@torch.inference_mode()
def encode_all(encoder: EncoderTransformer, ids: list[str], batch_size: int = 4):
    outputs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for start in range(0, len(ids), batch_size):
        chunk_ids = ids[start : start + batch_size]
        graphs = make_graphs(chunk_ids)
        batch = dense_padded_from_data_list(graphs)
        batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
        batch["chain_breaks_per_residue"] = torch.stack(
            [
                F.pad(
                    graph.chain_breaks_per_residue,
                    (0, batch["coords"].shape[1] - graph.chain_breaks_per_residue.shape[0]),
                )
                for graph in graphs
            ]
        )
        base_nm = batch["coords_nm"].clone()
        base_angstrom = batch["coords"].clone()
        for rotation_name, rotation in ROTATIONS.items():
            batch["coords_nm"] = base_nm @ rotation
            batch["coords"] = base_angstrom @ rotation
            latent = encoder(batch)["mean"]
            for i, protein_id in enumerate(chunk_ids):
                outputs[rotation_name][protein_id] = latent[i][batch["mask"][i]].float().cpu()
        print(f"encoded {start + len(chunk_ids):2d}/{len(ids)} proteins", flush=True)
    return outputs


def orthogonal_map(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Minimize ||source @ map - target||_F.
    u, _, vh = torch.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vh


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def rel_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(prediction - target) / torch.linalg.norm(target)).item()


def cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    gram_a, gram_b = a @ a.T, b @ b.T
    return ((gram_a * gram_b).sum() / (torch.linalg.norm(gram_a) * torch.linalg.norm(gram_b))).item()


def pooled(values, ids, rotation):
    return torch.cat([values[rotation][protein_id] for protein_id in ids], dim=0).double()


def evaluate_rotation(values, rotation: str):
    x_train = pooled(values, TRAIN_IDS, "identity")
    y_train = pooled(values, TRAIN_IDS, rotation)
    x_test = pooled(values, TEST_IDS, "identity")
    y_test = pooled(values, TEST_IDS, rotation)

    action = orthogonal_map(x_train, y_train)  # original -> rotated
    train_orth = x_train @ action
    test_orth = x_test @ action

    x_mean, y_mean = x_train.mean(0, keepdim=True), y_train.mean(0, keepdim=True)
    centered_action = orthogonal_map(x_train - x_mean, y_train - y_mean)
    test_centered = (x_test - x_mean) @ centered_action + y_mean

    x_aug_train = torch.cat([x_train, torch.ones(len(x_train), 1, dtype=x_train.dtype)], dim=1)
    affine = torch.linalg.lstsq(x_aug_train, y_train).solution
    x_aug_test = torch.cat([x_test, torch.ones(len(x_test), 1, dtype=x_test.dtype)], dim=1)
    test_affine = x_aug_test @ affine
    test_sst = ((y_test - y_test.mean(0, keepdim=True)) ** 2).sum()
    affine_r2 = 1.0 - ((test_affine - y_test) ** 2).sum() / test_sst

    per_protein_cka = [
        cka(values["identity"][protein_id].double(), values[rotation][protein_id].double())
        for protein_id in TEST_IDS
    ]
    return {
        "n_train_proteins": len(TRAIN_IDS),
        "n_test_proteins": len(TEST_IDS),
        "n_train_residues": len(x_train),
        "n_test_residues": len(x_test),
        "raw_test_cosine": cosine(x_test, y_test),
        "raw_test_relative_error": rel_error(x_test, y_test),
        "orthogonal_train_cosine": cosine(train_orth, y_train),
        "orthogonal_train_relative_error": rel_error(train_orth, y_train),
        "orthogonal_test_cosine": cosine(test_orth, y_test),
        "orthogonal_test_relative_error": rel_error(test_orth, y_test),
        "centered_orthogonal_test_cosine": cosine(test_centered, y_test),
        "centered_orthogonal_test_relative_error": rel_error(test_centered, y_test),
        "affine_test_cosine": cosine(test_affine, y_test),
        "affine_test_relative_error": rel_error(test_affine, y_test),
        "affine_test_r2": affine_r2.item(),
        "pooled_centered_cka_test": cka(x_test, y_test),
        "mean_per_protein_centered_cka_test": sum(per_protein_cka) / len(per_protein_cka),
        "action": action,
    }


def main(checkpoint_names: list[str] | None = None):
    torch.set_num_threads(16)
    all_ids = TRAIN_IDS + TEST_IDS
    names = list(CHECKPOINTS) if checkpoint_names is None else checkpoint_names
    unknown = sorted(set(names) - set(CHECKPOINTS))
    if unknown:
        raise ValueError(f"Unknown checkpoint(s) {unknown}; choices={list(CHECKPOINTS)}")
    for checkpoint_name in names:
        checkpoint_path = CHECKPOINTS[checkpoint_name]
        print(f"\n=== {checkpoint_name} ===", flush=True)
        encoder = load_encoder(checkpoint_path)
        values = encode_all(encoder, all_ids)
        results = {name: evaluate_rotation(values, name) for name in ROTATIONS if name != "identity"}

        a_z = results["rz90"]["action"]
        a_x = results["rx90"]["action"]
        a_zx = results["rz90_rx90"]["action"]
        group_error = torch.linalg.norm(a_zx - a_z @ a_x) / torch.linalg.norm(a_zx)
        reverse_group_error = torch.linalg.norm(a_zx - a_x @ a_z) / torch.linalg.norm(a_zx)

        printable = copy.deepcopy(results)
        for value in printable.values():
            del value["action"]
        printable["group_composition_relative_error_z_then_x"] = group_error.item()
        printable["group_composition_relative_error_reverse_order"] = reverse_group_error.item()
        print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
        del encoder, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", choices=list(CHECKPOINTS))
    args = parser.parse_args()
    main(args.checkpoints)
