"""Compare two postprocessed predicted-sequence PDB reconstruction sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BACKBONE = {"N", "CA", "C", "O"}


def read_atoms(path: Path):
    atoms = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key = (line[21], int(line[22:26]), line[26], line[12:16].strip())
        atoms[key] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    return atoms


def rmsd(native, predicted, keys):
    if not keys:
        return None
    return float(
        np.sqrt(np.mean([np.sum((native[key] - predicted[key]) ** 2) for key in keys]))
    )


def summarize(native, predicted, keys):
    backbone = {key for key in keys if key[3] in BACKBONE}
    sidechain = keys - backbone
    return {
        "all_common_atoms": len(keys),
        "all_common_rmsd_angstrom": rmsd(native, predicted, keys),
        "backbone_atoms": len(backbone),
        "backbone_rmsd_angstrom": rmsd(native, predicted, backbone),
        "sidechain_atoms": len(sidechain),
        "sidechain_rmsd_angstrom": rmsd(native, predicted, sidechain),
    }


def pooled_rmsd(squared_errors):
    return float(np.sqrt(np.mean(squared_errors))) if squared_errors else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonrotation-root", type=Path, required=True)
    parser.add_argument("--rotation-root", type=Path, required=True)
    parser.add_argument("--protein-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proteins = []
    pooled = {
        label: {group: [] for group in ("all", "backbone", "sidechain")}
        for label in ("nonrotation", "rotation")
    }
    pooled_all_atom = {
        label: {group: [] for group in ("all", "backbone", "sidechain")}
        for label in ("nonrotation", "rotation")
    }
    all_atom_proteins = []
    for protein_id in args.protein_ids:
        base = args.nonrotation_root / protein_id
        native_path = base / f"{protein_id}_native.pdb"
        nonrotation_path = base / f"{protein_id}_reconstructed_idealized.pdb"
        rotation_path = (
            args.rotation_root / protein_id / f"{protein_id}_reconstructed_idealized.pdb"
        )
        native = read_atoms(native_path)
        nonrotation = read_atoms(nonrotation_path)
        rotation = read_atoms(rotation_path)
        shared = set(native) & set(nonrotation) & set(rotation)
        shared_sidechain = {key for key in shared if key[3] not in BACKBONE}
        has_all_atom_reference = bool(shared_sidechain)
        if has_all_atom_reference:
            all_atom_proteins.append(protein_id)
        result = {"protein_id": protein_id, "shared_atoms": len(shared)}
        for label, prediction in (
            ("nonrotation", nonrotation),
            ("rotation", rotation),
        ):
            result[label] = summarize(native, prediction, shared)
            for key in shared:
                group = "backbone" if key[3] in BACKBONE else "sidechain"
                error = float(np.sum((native[key] - prediction[key]) ** 2))
                pooled[label]["all"].append(error)
                pooled[label][group].append(error)
                if has_all_atom_reference:
                    pooled_all_atom[label]["all"].append(error)
                    pooled_all_atom[label][group].append(error)
        proteins.append(result)

    aggregate = {}
    aggregate_all_atom = {}
    for label in pooled:
        aggregate[label] = {
            f"{group}_rmsd_angstrom": pooled_rmsd(errors)
            for group, errors in pooled[label].items()
        }
        aggregate[label].update(
            {f"{group}_atoms": len(errors) for group, errors in pooled[label].items()}
        )
        aggregate_all_atom[label] = {
            f"{group}_rmsd_angstrom": pooled_rmsd(errors)
            for group, errors in pooled_all_atom[label].items()
        }
        aggregate_all_atom[label].update(
            {
                f"{group}_atoms": len(errors)
                for group, errors in pooled_all_atom[label].items()
            }
        )
    output = {
        "protocol": (
            "Direct C-alpha-anchored RMSD after identical geometry postprocessing, "
            "using only Atom37 names present in native and both predicted-sequence PDBs."
        ),
        "aggregate": aggregate,
        "aggregate_all_atom_proteins": aggregate_all_atom,
        "all_atom_proteins": all_atom_proteins,
        "proteins": proteins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
