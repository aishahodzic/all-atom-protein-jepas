"""Evaluate side-chain coordinates when residue identity is supplied as an oracle.

The encoder and La Proteina decoder are unchanged.  The decoder still predicts
all Atom37 coordinates and sequence logits, but coordinate scoring uses the
true residue identity/mask.  This isolates sequence-classification failures
from side-chain coordinate and rotamer failures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from openfold.np import residue_constants


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path.insert(0, str(LINEAR_PROBING_DIR))

from Evaluation.structure_rec.geometry_projection import (  # noqa: E402
    idealize_atom37_geometry,
)
from Evaluation.structure_rec.train_decoder import (  # noqa: E402
    DECODER_STYLE_OF,
    load_donor_cfg_ae,
)
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import (  # noqa: E402
    dense_padded_from_data_list,
)
from proteinfoundation.utils.coors_utils import nm_to_ang  # noqa: E402
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb  # noqa: E402

import checkpoints as checkpoint_registry  # noqa: E402
from data_pool import build_protein_pool, eval_transform_compose  # noqa: E402


DEFAULT_PROTEINS = (
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
BACKBONE_ATOMS = (0, 1, 2, 4)  # N, CA, C, O in Atom37 order.


def _squared_error_angstrom(predicted_nm, native_nm):
    return ((predicted_nm - native_nm) * 10.0).square().sum(dim=-1)


def _accumulate(accumulator, squared_error, atom_mask, residue_mask):
    selected = atom_mask.bool() & residue_mask[..., None].bool()
    accumulator["squared_error"] += float(squared_error[selected].sum())
    accumulator["atoms"] += int(selected.sum())


def _finish(accumulator):
    if accumulator["atoms"] == 0:
        return {"rmsd_angstrom": None, "atoms": 0}
    return {
        "rmsd_angstrom": (
            accumulator["squared_error"] / accumulator["atoms"]
        )
        ** 0.5,
        "atoms": accumulator["atoms"],
    }


def evaluate(args):
    device = torch.device(args.device)
    pool = build_protein_pool(seed=args.pool_seed)
    id_to_index = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}

    protein_ids = args.protein_ids
    if args.protein_ids_file is not None:
        manifest = json.loads(args.protein_ids_file.read_text())
        if isinstance(manifest, list):
            protein_ids = manifest
        else:
            protein_ids = manifest[args.protein_ids_key]

    checkpoint_type = checkpoint_registry.ALL_CHECKPOINTS[args.encoder][1]
    checkpoint_registry.ALL_CHECKPOINTS[args.encoder] = (
        str(args.encoder_checkpoint.resolve()),
        checkpoint_type,
    )
    encoder = checkpoint_registry.load_frozen_encoder(
        args.encoder,
        device=str(device),
        encoder_source=args.encoder_source,
    )
    decoder_style = args.decoder_style or DECODER_STYLE_OF[args.encoder]
    model = AutoEncoder(
        cfg_ae=load_donor_cfg_ae(decoder_style),
        store_dir=str(pathlib.Path.home() / "tmp" / "oracle-aa-sidechains"),
    )
    model.encoder = encoder
    decoder_state = torch.load(
        args.decoder_checkpoint.resolve(), map_location=device, weights_only=True
    )
    model.decoder.load_state_dict(decoder_state, strict=True)
    model = model.to(device).eval()

    totals = {
        name: {"squared_error": 0.0, "atoms": 0}
        for name in (
            "oracle_sidechain_raw",
            "oracle_sidechain_idealized",
            "correct_aa_sidechain_raw",
            "incorrect_aa_sidechain_raw",
            "backbone_raw",
        )
    }
    proteins = []
    total_residues = 0
    correct_residues = 0
    transform = eval_transform_compose()

    with torch.inference_mode():
        for protein_id in protein_ids:
            graph = transform(pool.dataset[id_to_index[protein_id]])
            batch = dense_padded_from_data_list([graph])
            batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
            batch["chain_breaks_per_residue"] = (
                graph.chain_breaks_per_residue.unsqueeze(0)
            )
            batch = batch.to(device)
            residue_mask = batch["mask"].bool()
            native = batch["coords_nm"]
            true_atom_mask = batch["coord_mask"].bool()
            ca = native[..., 1, :] * residue_mask[..., None]
            torch.manual_seed(args.latent_seed)
            encoder_output = model.encoder(batch)
            if args.latent_tap == "mean":
                latent = encoder_output["mean"]
            elif args.latent_tap == "normalized-mean":
                latent = model.encoder.ln_z(encoder_output["mean"]) * residue_mask[..., None]
            elif args.latent_tap == "sample":
                latent = encoder_output["z_latent"]
            else:  # guarded by argparse
                raise ValueError(f"unknown latent tap {args.latent_tap!r}")
            decoded = model.decoder(
                {
                    "z_latent": latent,
                    "ca_coors_nm": ca,
                    "residue_mask": residue_mask,
                    "mask": residue_mask,
                }
            )
            predicted = decoded["coors_nm"]
            correct_aa = decoded["aatype_max"] == batch["residue_type"]

            export_paths = None
            if args.export_root is not None:
                protein_root = args.export_root / protein_id
                protein_root.mkdir(parents=True, exist_ok=True)
                valid = residue_mask[0]
                native_coords = nm_to_ang(native[0, valid]).detach().cpu().numpy()
                predicted_coords = (
                    nm_to_ang(predicted[0, valid]).detach().cpu().numpy()
                )
                native_aatype = (
                    batch["residue_type"][0, valid].detach().cpu().numpy()
                )
                predicted_aatype = (
                    decoded["aatype_max"][0, valid].detach().cpu().numpy()
                )
                native_mask = (
                    true_atom_mask[0, valid].detach().cpu().numpy().astype(bool)
                )
                predicted_mask = (
                    decoded["atom_mask"][0, valid]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                chain_index = batch["chains"][0, valid].detach().cpu().numpy()
                native_path = protein_root / f"{protein_id}_native.pdb"
                true_path = (
                    protein_root
                    / f"{protein_id}_reconstructed_true_sequence.pdb"
                )
                predicted_path = (
                    protein_root
                    / f"{protein_id}_reconstructed_predicted_sequence.pdb"
                )
                write_prot_to_pdb(
                    prot_pos=native_coords * native_mask[..., None],
                    file_path=str(native_path),
                    aatype=native_aatype,
                    chain_index=chain_index,
                    overwrite=True,
                    no_indexing=True,
                )
                write_prot_to_pdb(
                    prot_pos=predicted_coords * native_mask[..., None],
                    file_path=str(true_path),
                    aatype=native_aatype,
                    chain_index=chain_index,
                    overwrite=True,
                    no_indexing=True,
                )
                write_prot_to_pdb(
                    prot_pos=predicted_coords * predicted_mask[..., None],
                    file_path=str(predicted_path),
                    aatype=predicted_aatype,
                    chain_index=chain_index,
                    overwrite=True,
                    no_indexing=True,
                )
                export_paths = {
                    "native_pdb": str(native_path.resolve()),
                    "reconstructed_true_sequence_pdb": str(true_path.resolve()),
                    "reconstructed_predicted_sequence_pdb": str(
                        predicted_path.resolve()
                    ),
                }

            sidechain_mask = true_atom_mask.clone()
            sidechain_mask[..., list(BACKBONE_ATOMS)] = False
            backbone_mask = true_atom_mask & ~sidechain_mask
            raw_error = _squared_error_angstrom(predicted, native)
            idealized, idealized_mask = idealize_atom37_geometry(
                predicted,
                batch["residue_type"],
                true_atom_mask,
            )
            idealized_error = _squared_error_angstrom(idealized, native)

            local = {
                name: {"squared_error": 0.0, "atoms": 0}
                for name in totals
            }
            _accumulate(
                local["oracle_sidechain_raw"],
                raw_error,
                sidechain_mask,
                residue_mask,
            )
            _accumulate(
                local["oracle_sidechain_idealized"],
                idealized_error,
                sidechain_mask & idealized_mask.bool(),
                residue_mask,
            )
            _accumulate(
                local["correct_aa_sidechain_raw"],
                raw_error,
                sidechain_mask,
                residue_mask & correct_aa,
            )
            _accumulate(
                local["incorrect_aa_sidechain_raw"],
                raw_error,
                sidechain_mask,
                residue_mask & ~correct_aa,
            )
            _accumulate(
                local["backbone_raw"], raw_error, backbone_mask, residue_mask
            )
            for name in totals:
                totals[name]["squared_error"] += local[name]["squared_error"]
                totals[name]["atoms"] += local[name]["atoms"]

            n_residues = int(residue_mask.sum())
            n_correct = int((correct_aa & residue_mask).sum())
            native_indices = batch["residue_type"][residue_mask].detach().cpu().tolist()
            predicted_indices = decoded["aatype_max"][residue_mask].detach().cpu().tolist()
            total_residues += n_residues
            correct_residues += n_correct
            proteins.append(
                {
                    "protein_id": protein_id,
                    "residues": n_residues,
                    "sequence_recovery": n_correct / n_residues,
                    "native_sequence": "".join(
                        residue_constants.restypes_with_x[index]
                        for index in native_indices
                    ),
                    "predicted_sequence": "".join(
                        residue_constants.restypes_with_x[index]
                        for index in predicted_indices
                    ),
                    **(export_paths or {}),
                    **{name: _finish(value) for name, value in local.items()},
                }
            )

    result = {
        "protocol": (
            "Frozen encoder and unchanged La Proteina decoder; true amino-acid "
            "identity/mask used only for coordinate scoring and idealization."
        ),
        "encoder": args.encoder,
        "encoder_source": args.encoder_source,
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "decoder_checkpoint": str(args.decoder_checkpoint.resolve()),
        "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
        "latent_tap": args.latent_tap,
        "latent_seed": args.latent_seed,
        "proteins": proteins,
        "aggregate": {
            "proteins": len(proteins),
            "residues": total_residues,
            "sequence_recovery": correct_residues / total_residues,
            **{name: _finish(value) for name, value in totals.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="jepa-rand-mask-70")
    parser.add_argument("--encoder-source", default="target", choices=("online", "target"))
    parser.add_argument("--encoder-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--decoder-style", choices=("ca-cond", "fully-latent"))
    parser.add_argument("--protein-ids", nargs="+", default=list(DEFAULT_PROTEINS))
    parser.add_argument(
        "--protein-ids-file",
        type=pathlib.Path,
        help="JSON list or cohort manifest containing the requested protein IDs.",
    )
    parser.add_argument(
        "--protein-ids-key",
        default="export_ids",
        help="Manifest key used when --protein-ids-file contains an object.",
    )
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument(
        "--latent-tap",
        choices=("mean", "normalized-mean", "sample"),
        default="mean",
    )
    parser.add_argument("--latent-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--export-root",
        type=pathlib.Path,
        help=(
            "Optionally export native, predicted-sequence and true-sequence "
            "PDB files for each evaluated protein."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
