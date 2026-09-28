"""Export one frozen-encoder decoder reconstruction for PyMOL.

Fresh decoders trained by ``train_decoder.py`` are C-alpha anchored.  The
original fully-latent La Proteina branch is different: its paired decoder
predicts absolute Atom37 coordinates without receiving C-alpha coordinates.
This script supports both coordinate modes.  It is an evaluation visualizer,
not sequence-to-structure generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path.insert(0, str(LINEAR_PROBING_DIR))
CA_SLOT = 1


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={requested!r} requested, but CUDA is unavailable")
    return requested


def _select_protein(pool: Any, protein_id: str | None, split: str) -> tuple[int, str]:
    if protein_id is not None:
        try:
            return pool.protein_ids.index(protein_id), protein_id
        except ValueError as error:
            close = [pid for pid in pool.protein_ids if protein_id.lower() in pid.lower()]
            suffix = f"; partial matches: {close[:10]}" if close else ""
            raise ValueError(f"protein {protein_id!r} is not in the held-out pool{suffix}") from error

    for index, candidate in enumerate(pool.protein_ids):
        if split == "any" or pool.origin[candidate] == split:
            return index, candidate
    raise RuntimeError(f"held-out pool contains no protein from split={split!r}")


def _pdb_path(output_dir: pathlib.Path, protein_id: str, suffix: str) -> pathlib.Path:
    safe_id = protein_id.replace("/", "_").replace(":", "_")
    return output_dir / f"{safe_id}_{suffix}.pdb"


def _write_pymol_script(
    path: pathlib.Path,
    native_path: pathlib.Path,
    reconstructed_path: pathlib.Path,
    reconstructed_predicted_sequence_path: pathlib.Path,
) -> None:
    # Keep paths relative so the complete output directory can be copied to a
    # workstation and opened there. Run PyMOL with this directory as its CWD.
    commands = [
        "reinitialize",
        f'load "./{native_path.name}", native',
        f'load "./{reconstructed_path.name}", reconstruction_true_sequence',
        f'load "./{reconstructed_predicted_sequence_path.name}", reconstruction_predicted_sequence',
        "dss native",
        "dss reconstruction_true_sequence",
        "dss reconstruction_predicted_sequence",
        "bg_color white",
        "set cartoon_side_chain_helper, 1",
        "set cartoon_fancy_helices, 1",
        "set cartoon_fancy_sheets, 1",
        "set cartoon_highlight_color, grey90",
        "set cartoon_discrete_colors, 0",
        "set ray_trace_mode, 1",
        "set antialias, 2",
        "set ray_shadows, 1",
        "set ray_opaque_background, 0",
        "set ambient, 0.1",
        "set ray_trace_gain, 0.05",
        "set ray_blend_colors, 1",
        "set ray_trace_color, black",
        "set stick_radius, 0.15",
        "set_color helix_color, [0.81, 0.59, 0.79]",
        "set_color beta_color, [0.88, 0.70, 0.51]",
        "set_color loop_color, [0.84, 0.85, 0.86]",
        "set_color c_color, [0.45, 0.45, 0.45]",
        "set_color o_color, [1.0, 0.7, 0.7]",
        "set_color n_color, [0.7, 0.8, 1.0]",
        "set_color s_color, [1.0, 0.95, 0.7]",
        "set_color cl_color, [0.75, 0.90, 0.75]",
        "set_color f_color, [0.7, 1.0, 0.9]",
        "set_color hbond_color, [0.86, 0.08, 0.24]",
        "set_color target_cartoon_color, [0.61, 0.59, 0.81]",
        "hide everything",
        "show cartoon, native or reconstruction_true_sequence",
        "show sticks, reconstruction_true_sequence and not backbone",
        "color c_color, reconstruction_true_sequence and elem C and not name CA",
        "color o_color, reconstruction_true_sequence and elem O",
        "color n_color, reconstruction_true_sequence and elem N",
        "color s_color, reconstruction_true_sequence and elem S",
        "color f_color, reconstruction_true_sequence and elem F",
        "color cl_color, reconstruction_true_sequence and elem Cl",
        "set cartoon_color, target_cartoon_color, native",
        "set cartoon_color, helix_color, reconstruction_true_sequence and ss H",
        "set cartoon_color, beta_color, reconstruction_true_sequence and ss S",
        "set cartoon_color, loop_color, reconstruction_true_sequence and not (ss H+S)",
        "color tv_orange, reconstruction_predicted_sequence",
        "disable reconstruction_predicted_sequence",
        "align reconstruction_true_sequence and name CA, native and name CA",
        "set cartoon_transparency, 0.55, native",
        "set cartoon_transparency, 0.00, reconstruction_true_sequence",
        "create target_surface, native",
        "show surface, target_surface",
        "color grey90, target_surface",
        "set transparency, 0.75, target_surface",
        "disable target_surface",
        "distance hbonds, reconstruction_true_sequence and donors, reconstruction_true_sequence and acceptors, 3.2, 2",
        "color hbond_color, hbonds",
        "set dash_width, 7.0, hbonds",
        "set dash_gap, 0.4, hbonds",
        "hide labels, hbonds",
        "disable hbonds",
        "orient native or reconstruction_true_sequence",
        "zoom native or reconstruction_true_sequence",
    ]
    path.write_text("\n".join(commands) + "\n")


def export_reconstruction(args: argparse.Namespace) -> dict[str, Any]:
    if "DATA_PATH" not in os.environ:
        raise RuntimeError(
            "DATA_PATH is unset. From the repository root, run `set -a; "
            "source .env; set +a` before exporting."
        )

    import torch

    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    from proteinfoundation.utils.coors_utils import nm_to_ang
    from proteinfoundation.utils.dense_padding_data_loader import (
        dense_padded_from_data_list,
    )
    from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

    import checkpoints as checkpoint_registry
    from data_pool import build_protein_pool, eval_transform_compose
    from train_decoder import DECODER_STYLE_OF, compute_rmsd, load_donor_cfg_ae

    decoder_checkpoint = args.decoder_checkpoint.resolve()
    if not decoder_checkpoint.is_file():
        raise FileNotFoundError(decoder_checkpoint)

    device = _resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = build_protein_pool(seed=args.pool_seed)
    protein_index, protein_id = _select_protein(pool, args.protein_id, args.split)
    graph = eval_transform_compose()(pool.dataset[protein_index])
    batch = dense_padded_from_data_list([graph])
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
    batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
    batch = batch.to(device)

    encoder_checkpoint = None
    encoder_checkpoint_sha256 = None
    if args.encoder_checkpoint is not None:
        encoder_checkpoint = args.encoder_checkpoint.resolve()
        if not encoder_checkpoint.is_file():
            raise FileNotFoundError(encoder_checkpoint)
        encoder_checkpoint_sha256 = _sha256(encoder_checkpoint)
        if (
            args.expected_encoder_sha256 is not None
            and encoder_checkpoint_sha256 != args.expected_encoder_sha256
        ):
            raise RuntimeError(
                "encoder checkpoint SHA256 mismatch: "
                f"got {encoder_checkpoint_sha256}, "
                f"expected {args.expected_encoder_sha256}"
            )
        # Preserve the registry's checkpoint type when overriding its path.
        # Explicit paths are useful for both JEPA checkpoints and paired
        # online/EMA autoencoder checkpoints; treating every override as JEPA
        # would incorrectly search an autoencoder checkpoint for
        # ``online_encoder.*`` keys.
        checkpoint_type = checkpoint_registry.ALL_CHECKPOINTS[args.encoder][1]
        checkpoint_registry.ALL_CHECKPOINTS[args.encoder] = (
            str(encoder_checkpoint),
            checkpoint_type,
        )

    encoder = checkpoint_registry.load_frozen_encoder(
        args.encoder, device=device, encoder_source=args.encoder_source
    )
    decoder_style = args.decoder_style or DECODER_STYLE_OF[args.encoder]
    donor_cfg_ae = load_donor_cfg_ae(decoder_style)
    scratch_root = pathlib.Path.home() / "tmp" / "decoder-reconstruction-export"
    scratch_root.mkdir(parents=True, exist_ok=True)
    model = AutoEncoder(cfg_ae=donor_cfg_ae, store_dir=str(scratch_root))
    model.encoder = encoder
    decoder_state = torch.load(
        decoder_checkpoint, map_location=device, weights_only=True
    )
    model.decoder.load_state_dict(decoder_state, strict=True)
    model = model.to(device).eval()

    with torch.inference_mode():
        mask = batch["mask"].bool()
        ca_coors_nm = batch["coords_nm"][..., CA_SLOT, :] * mask[..., None]
        torch.manual_seed(args.latent_seed)
        encoder_output = model.encoder(batch)
        if args.latent_tap == "mean":
            latent = encoder_output["mean"]
        elif args.latent_tap == "normalized-mean":
            # The jointly trained AE decoder consumes encoder z_latent, which
            # is layer-normalized after sampling.  Normalizing the mean gives
            # a deterministic counterpart in that same decoder input space.
            latent = model.encoder.ln_z(encoder_output["mean"]) * mask[..., None]
        elif args.latent_tap == "sample":
            latent = encoder_output["z_latent"]
        else:
            raise ValueError(f"unknown latent tap {args.latent_tap!r}")
        decoder_input = {
            "z_latent": latent,
            "residue_mask": mask,
            "mask": mask,
        }
        if args.coordinate_mode == "ca-anchored":
            decoder_input["ca_coors_nm"] = ca_coors_nm
        decoded = model.decoder(decoder_input)
        rmsd_all = float(compute_rmsd(decoded, batch, exclude_ca=False)[0])
        rmsd_no_ca = float(compute_rmsd(decoded, batch, exclude_ca=True)[0])
        sequence_recovery = float(
            (decoded["aatype_max"][mask] == batch["residue_type"][mask])
            .float()
            .mean()
        )

    valid = mask[0]
    native_coords = nm_to_ang(batch["coords_nm"][0, valid]).detach().cpu().numpy()
    reconstructed_coords = (
        nm_to_ang(decoded["coors_nm"][0, valid]).detach().cpu().numpy()
    )
    native_aatype = batch["residue_type"][0, valid].detach().cpu().numpy()
    predicted_aatype = decoded["aatype_max"][0, valid].detach().cpu().numpy()
    native_atom_mask = batch["coord_mask"][0, valid].bool().detach().cpu().numpy()
    predicted_atom_mask = decoded["atom_mask"][0, valid].bool().detach().cpu().numpy()
    chain_index = batch["chains"][0, valid].detach().cpu().numpy()

    native_path = _pdb_path(output_dir, protein_id, "native")
    reconstructed_path = _pdb_path(
        output_dir, protein_id, "reconstructed_true_sequence"
    )
    reconstructed_predicted_sequence_path = _pdb_path(
        output_dir, protein_id, "reconstructed_predicted_sequence"
    )
    write_prot_to_pdb(
        prot_pos=native_coords * native_atom_mask[..., None],
        file_path=str(native_path),
        aatype=native_aatype,
        chain_index=chain_index,
        overwrite=True,
        no_indexing=True,
    )
    write_prot_to_pdb(
        prot_pos=reconstructed_coords * native_atom_mask[..., None],
        file_path=str(reconstructed_path),
        aatype=native_aatype,
        chain_index=chain_index,
        overwrite=True,
        no_indexing=True,
    )
    write_prot_to_pdb(
        prot_pos=reconstructed_coords * predicted_atom_mask[..., None],
        file_path=str(reconstructed_predicted_sequence_path),
        aatype=predicted_aatype,
        chain_index=chain_index,
        overwrite=True,
        no_indexing=True,
    )

    pymol_path = output_dir / f"{protein_id.replace('/', '_').replace(':', '_')}_view.pml"
    _write_pymol_script(
        pymol_path,
        native_path,
        reconstructed_path,
        reconstructed_predicted_sequence_path,
    )

    result = {
        "protein_id": protein_id,
        "source_split": pool.origin[protein_id],
        "num_residues": int(valid.sum()),
        "encoder": args.encoder,
        "encoder_source": args.encoder_source,
        "encoder_checkpoint": (
            str(encoder_checkpoint) if encoder_checkpoint is not None else "registry"
        ),
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "decoder_style": decoder_style,
        "decoder_checkpoint": str(decoder_checkpoint),
        "decoder_checkpoint_sha256": _sha256(decoder_checkpoint),
        "latent_tap": args.latent_tap,
        "latent_seed": args.latent_seed,
        "device": device,
        "c_alpha_anchored": args.coordinate_mode == "ca-anchored",
        "rmsd_all_atom_angstrom": rmsd_all,
        "rmsd_no_ca_angstrom": rmsd_no_ca,
        "sequence_recovery": sequence_recovery,
        "native_pdb": str(native_path),
        "reconstructed_true_sequence_pdb": str(reconstructed_path),
        "reconstructed_predicted_sequence_pdb": str(
            reconstructed_predicted_sequence_path
        ),
        "pymol_script": str(pymol_path),
    }
    metadata_path = output_dir / f"{protein_id.replace('/', '_').replace(':', '_')}_metrics.json"
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    result["metadata"] = str(metadata_path)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    default_checkpoint = (
        REPO_ROOT
        / "Evaluation"
        / "structure_rec"
        / "checkpoints"
        / "jepa-rand-mask-50-target-calib_step20000.pt"
    )
    parser = argparse.ArgumentParser(
        description="Export one C-alpha-anchored decoder reconstruction for PyMOL."
    )
    parser.add_argument("--decoder-checkpoint", type=pathlib.Path, default=default_checkpoint)
    parser.add_argument("--encoder", default="jepa-rand-mask-50")
    parser.add_argument("--encoder-source", choices=("online", "target"), default="target")
    parser.add_argument(
        "--encoder-checkpoint",
        type=pathlib.Path,
        default=None,
        help="Optional explicit JEPA checkpoint, overriding the encoder registry path.",
    )
    parser.add_argument(
        "--expected-encoder-sha256",
        default=None,
        help="If supplied, fail unless the explicit encoder checkpoint has this SHA256.",
    )
    parser.add_argument("--decoder-style", choices=("ca-cond", "fully-latent"), default=None)
    parser.add_argument(
        "--coordinate-mode",
        choices=("ca-anchored", "fully-latent"),
        default="ca-anchored",
        help=(
            "Whether the decoder receives native C-alpha coordinates. Fresh "
            "probe decoders use ca-anchored; the original fully-latent La "
            "Proteina decoder uses fully-latent."
        ),
    )
    parser.add_argument(
        "--latent-tap",
        choices=("mean", "normalized-mean", "sample"),
        default="mean",
        help=(
            "Decoder input. Fresh frozen-encoder probes use mean; a decoder "
            "jointly trained with the AE expects sampled z_latent or the "
            "deterministic normalized-mean counterpart."
        ),
    )
    parser.add_argument("--latent-seed", type=int, default=42)
    parser.add_argument("--protein-id", default=None)
    parser.add_argument(
        "--split",
        choices=("test", "val", "any"),
        default="test",
        help="Split used only when --protein-id is omitted (default: first test protein).",
    )
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "Evaluation" / "structure_rec" / "exports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_reconstruction(parse_args())
