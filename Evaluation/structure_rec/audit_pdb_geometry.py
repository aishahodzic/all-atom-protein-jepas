"""Audit residue covalent/virtual and peptide-bond geometry in PDB exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openfold.np import residue_constants


def read_pdb(path: Path):
    residues = []
    by_key = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip()
        resseq = int(line[22:26])
        insertion = line[26].strip()
        key = (chain, resseq, insertion)
        if key not in by_key:
            entry = {"key": key, "resname": resname, "atoms": {}}
            by_key[key] = entry
            residues.append(entry)
        by_key[key]["atoms"][atom] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    return residues


def summarize(values):
    deviations = np.asarray([abs(actual - ideal) for actual, ideal in values])
    return {
        "count": int(len(deviations)),
        "mean_absolute_deviation_angstrom": float(deviations.mean()),
        "median_absolute_deviation_angstrom": float(np.median(deviations)),
        "max_absolute_deviation_angstrom": float(deviations.max()),
        "fraction_over_0p2_angstrom": float((deviations > 0.2).mean()),
        "fraction_over_0p5_angstrom": float((deviations > 0.5).mean()),
    }


def audit(path: Path, ca_gap_threshold: float = 4.5):
    bonds, virtual_bonds, _ = residue_constants.load_stereo_chemical_props()
    residues = read_pdb(path)
    covalent = []
    virtual = []
    peptide = []
    residue_details = []
    for residue in residues:
        atoms = residue["atoms"]
        residue_bonds = []
        for bond in bonds.get(residue["resname"], []):
            if bond.atom1_name in atoms and bond.atom2_name in atoms:
                distance = float(
                    np.linalg.norm(atoms[bond.atom1_name] - atoms[bond.atom2_name])
                )
                covalent.append((distance, bond.length))
                residue_bonds.append(
                    {
                        "atoms": f"{bond.atom1_name}-{bond.atom2_name}",
                        "distance": distance,
                        "ideal": float(bond.length),
                        "absolute_deviation": abs(distance - bond.length),
                    }
                )
        for bond in virtual_bonds.get(residue["resname"], []):
            if bond.atom1_name in atoms and bond.atom2_name in atoms:
                distance = float(
                    np.linalg.norm(atoms[bond.atom1_name] - atoms[bond.atom2_name])
                )
                virtual.append((distance, bond.length))
        residue_details.append(
            {
                "chain": residue["key"][0],
                "residue_number": residue["key"][1],
                "resname": residue["resname"],
                "covalent_bonds": residue_bonds,
            }
        )
    for left, right in zip(residues[:-1], residues[1:]):
        if left["key"][0] != right["key"][0]:
            continue
        if right["key"][1] - left["key"][1] != 1:
            continue
        if "CA" not in left["atoms"] or "CA" not in right["atoms"]:
            continue
        ca_distance = float(
            np.linalg.norm(left["atoms"]["CA"] - right["atoms"]["CA"])
        )
        if ca_distance > ca_gap_threshold:
            continue
        if "C" not in left["atoms"] or "N" not in right["atoms"]:
            continue
        distance = float(np.linalg.norm(left["atoms"]["C"] - right["atoms"]["N"]))
        next_is_proline = right["resname"] == "PRO"
        ideal = float(residue_constants.between_res_bond_length_c_n[next_is_proline])
        peptide.append((distance, ideal))

    return {
        "pdb": str(path),
        "residues": len(residues),
        "ca_gap_threshold_angstrom": ca_gap_threshold,
        "covalent": summarize(covalent),
        "virtual": summarize(virtual),
        "peptide_c_n": summarize(peptide),
        "residue_details": residue_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ca-gap-threshold", type=float, default=4.5)
    args = parser.parse_args()
    result = [
        audit(path.resolve(), ca_gap_threshold=args.ca_gap_threshold)
        for path in args.pdb
    ]
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
