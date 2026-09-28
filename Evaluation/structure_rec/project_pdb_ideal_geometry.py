"""Project an exported protein PDB onto OpenFold ideal residue geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from openfold.np import residue_constants

from geometry_projection import idealize_atom37_geometry


def project_pdb(
    input_path: Path,
    output_path: Path,
    *,
    preserve_backbone: bool = False,
    only_malformed_threshold: float | None = None,
) -> dict:
    lines = input_path.read_text().splitlines()
    residues = []
    residue_by_key = {}
    atom_records = []
    for line_index, line in enumerate(lines):
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:26], line[26])
        if key not in residue_by_key:
            resname = line[17:20].strip()
            one_letter = residue_constants.restype_3to1.get(resname, "X")
            aatype = residue_constants.restype_order.get(one_letter, 20)
            residue_by_key[key] = len(residues)
            residues.append({"aatype": aatype, "atoms": {}})
        residue_index = residue_by_key[key]
        atom_name = line[12:16].strip()
        if atom_name not in residue_constants.atom_order:
            continue
        atom_index = residue_constants.atom_order[atom_name]
        coordinate_ang = torch.tensor(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
        residues[residue_index]["atoms"][atom_index] = coordinate_ang
        atom_records.append((line_index, residue_index, atom_index))

    length = len(residues)
    coords_nm = torch.zeros((1, length, 37, 3), dtype=torch.float32)
    mask = torch.zeros((1, length, 37), dtype=torch.bool)
    aatype = torch.tensor([[residue["aatype"] for residue in residues]])
    for residue_index, residue in enumerate(residues):
        for atom_index, coordinate_ang in residue["atoms"].items():
            coords_nm[0, residue_index, atom_index] = coordinate_ang / 10.0
            mask[0, residue_index, atom_index] = True

    projected_nm, _ = idealize_atom37_geometry(coords_nm, aatype, mask)
    if preserve_backbone:
        # Retain the decoder's N, CA, C, and O coordinates verbatim.  CA was
        # already invariant; preserving the other backbone atoms makes this a
        # targeted side-chain repair rather than a full residue projection.
        backbone_indices = torch.tensor([0, 1, 2, 4])
        projected_nm[..., backbone_indices, :] = coords_nm[..., backbone_indices, :]
    selected = set(range(length))
    maximum_bond_deviation = [0.0] * length
    if only_malformed_threshold is not None:
        bonds, _, _ = residue_constants.load_stereo_chemical_props()
        selected = set()
        for residue_index, residue in enumerate(residues):
            resname = residue_constants.restype_1to3.get(
                residue_constants.restypes_with_x[residue["aatype"]], "UNK"
            )
            for bond in bonds.get(resname, []):
                left = residue_constants.atom_order[bond.atom1_name]
                right = residue_constants.atom_order[bond.atom2_name]
                if not (mask[0, residue_index, left] and mask[0, residue_index, right]):
                    continue
                distance = torch.linalg.vector_norm(
                    coords_nm[0, residue_index, left]
                    - coords_nm[0, residue_index, right]
                ).item() * 10.0
                deviation = abs(distance - bond.length)
                maximum_bond_deviation[residue_index] = max(
                    maximum_bond_deviation[residue_index], deviation
                )
            if maximum_bond_deviation[residue_index] > only_malformed_threshold:
                selected.add(residue_index)
    projected_ang = projected_nm[0] * 10.0
    for line_index, residue_index, atom_index in atom_records:
        source = projected_ang if residue_index in selected else coords_nm[0] * 10.0
        x, y, z = source[residue_index, atom_index].tolist()
        line = lines[line_index]
        lines[line_index] = (
            line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    displacement = torch.linalg.vector_norm(
        projected_nm - coords_nm, dim=-1
    )[mask]
    selected_mask = torch.zeros((1, length, 37), dtype=torch.bool)
    for residue_index in selected:
        selected_mask[0, residue_index] = mask[0, residue_index]
    selected_displacement = torch.linalg.vector_norm(
        projected_nm - coords_nm, dim=-1
    )[selected_mask]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "preserve_backbone": preserve_backbone,
        "only_malformed_threshold_angstrom": only_malformed_threshold,
        "residue_count": length,
        "projected_residue_count": len(selected),
        "projected_residue_indices_1based": [index + 1 for index in sorted(selected)],
        "mean_selected_atom_displacement_angstrom": (
            float(selected_displacement.mean() * 10.0)
            if selected_displacement.numel()
            else 0.0
        ),
        "max_selected_atom_displacement_angstrom": (
            float(selected_displacement.max() * 10.0)
            if selected_displacement.numel()
            else 0.0
        ),
        "mean_full_projection_displacement_angstrom": float(
            displacement.mean() * 10.0
        ),
    }


def write_pymol(path: Path, native: Path, raw: Path, projected: Path) -> None:
    commands = [
        "reinitialize",
        f'load "./{native.name}", native',
        f'load "./{raw.name}", reconstruction_raw',
        f'load "./{projected.name}", reconstruction_projected',
        "dss native",
        "dss reconstruction_raw",
        "dss reconstruction_projected",
        "bg_color white",
        "hide everything",
        "show cartoon, native or reconstruction_projected",
        "show sticks, reconstruction_projected and not backbone",
        "color gray70, native",
        "color tv_orange, reconstruction_projected",
        "color red, reconstruction_projected and elem O",
        "color marine, reconstruction_projected and elem N",
        "color yellow, reconstruction_projected and elem S",
        "disable reconstruction_raw",
        "align reconstruction_projected and name CA, native and name CA",
        "set cartoon_transparency, 0.55, native",
        "set cartoon_transparency, 0.00, reconstruction_projected",
        "set stick_radius, 0.15",
        "set cartoon_fancy_helices, 1",
        "set cartoon_fancy_sheets, 1",
        "set ray_trace_mode, 1",
        "set antialias, 2",
        "set ray_opaque_background, 0",
        "orient native or reconstruction_projected",
        "zoom native or reconstruction_projected",
    ]
    path.write_text("\n".join(commands) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--pml", type=Path)
    parser.add_argument("--preserve-backbone", action="store_true")
    parser.add_argument("--only-malformed-threshold", type=float)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    result = project_pdb(
        args.input,
        args.output,
        preserve_backbone=args.preserve_backbone,
        only_malformed_threshold=args.only_malformed_threshold,
    )
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.pml is not None:
        if args.native is None:
            raise ValueError("--native is required with --pml")
        write_pymol(args.pml, args.native, args.input, args.output)


if __name__ == "__main__":
    main()
