"""Clash-aware canonical-rotamer repair for idealized decoder PDBs.

This is a post-processing experiment.  It never uses native coordinates.
Starting from an ideal-geometry PDB, it finds severe contacts between
non-neighbouring residues and repacks only the implicated side chains onto a
small canonical chi-angle grid.  Candidates are selected by steric-clash
energy plus a weak displacement penalty from the decoder's conformation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from openfold.np import residue_constants

from Evaluation.structure_rec.restrained_geometry_repair import _parse


BACKBONE = {"N", "CA", "C", "O", "OXT"}
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
PEPTIDE_O_N_MIN_DISTANCE = 2.20


def _dihedral(p0, p1, p2, p3) -> float:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def _rotation(points, origin, axis, angle):
    axis = axis / np.linalg.norm(axis)
    shifted = points - origin
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        shifted * cosine
        + np.cross(axis, shifted) * sine
        + np.outer(shifted @ axis, axis) * (1.0 - cosine)
        + origin
    )


def _bond_graph(resname: str, present: set[str]) -> dict[str, set[str]]:
    bonds, _, _ = residue_constants.load_stereo_chemical_props()
    graph = {name: set() for name in present}
    for bond in bonds.get(resname, []):
        if bond.atom1_name in present and bond.atom2_name in present:
            graph[bond.atom1_name].add(bond.atom2_name)
            graph[bond.atom2_name].add(bond.atom1_name)
    return graph


def _set_chi(atom_coords: dict[str, np.ndarray], resname: str, chi_index: int, target: float):
    names = residue_constants.chi_angles_atoms[resname][chi_index]
    if any(name not in atom_coords for name in names):
        return atom_coords
    a, b, c, d = names
    graph = _bond_graph(resname, set(atom_coords))
    downstream = set()
    stack = [c]
    while stack:
        node = stack.pop()
        if node in downstream or node == b:
            continue
        downstream.add(node)
        stack.extend(graph[node] - {b})

    current = _dihedral(atom_coords[a], atom_coords[b], atom_coords[c], atom_coords[d])
    delta = math.atan2(math.sin(target - current), math.cos(target - current))
    moving = sorted(downstream - {c})
    if not moving:
        return atom_coords
    result = {name: value.copy() for name, value in atom_coords.items()}
    points = np.stack([result[name] for name in moving])
    rotated = _rotation(points, result[c], result[c] - result[b], delta)
    for name, point in zip(moving, rotated):
        result[name] = point
    check = _dihedral(result[a], result[b], result[c], result[d])
    error = abs(math.atan2(math.sin(check - target), math.cos(check - target)))
    if error > 1e-5:
        raise RuntimeError(
            f"failed to set {resname} chi{chi_index + 1}: error={math.degrees(error):.4g} deg"
        )
    return result


def _chi_bins(resname: str, chi_index: int) -> tuple[float, ...]:
    # Planar terminal groups and aromatic rings use approximately orthogonal
    # chi values; flexible sp3 chains use the three staggered states.
    if chi_index == 1 and resname in {"ASP", "ASN", "PHE", "TYR", "HIS", "TRP"}:
        degrees = (-90.0, 90.0)
    elif chi_index == 2 and resname in {"GLU", "GLN"}:
        degrees = (-90.0, 90.0)
    elif chi_index == 3 and resname == "ARG":
        degrees = (-90.0, 90.0, 180.0)
    else:
        degrees = (-60.0, 60.0, 180.0)
    return tuple(math.radians(value) for value in degrees)


def _residue_atoms(residue, coords, residue_index):
    return {
        name: coords[residue_index, atom_index].copy()
        for name, atom_index in residue["atoms"].items()
    }


def _candidate_atoms(residue, coords, residue_index):
    resname = residue["resname"]
    if resname == "PRO" or resname not in residue_constants.chi_angles_atoms:
        return []
    n_chi = len(residue_constants.chi_angles_atoms[resname])
    if n_chi == 0:
        return []
    original = _residue_atoms(residue, coords, residue_index)
    candidates = [original]
    for targets in itertools.product(*[_chi_bins(resname, index) for index in range(n_chi)]):
        candidate = original
        for index, target in enumerate(targets):
            candidate = _set_chi(candidate, resname, index, target)
        candidates.append(candidate)
    return candidates


def _sidechain_names(atom_coords):
    return [name for name in atom_coords if name not in BACKBONE]


def _clash_energy(candidate, residue_index, residues, coords, overlap):
    names = _sidechain_names(candidate)
    if not names:
        return 0.0
    energy = 0.0
    for name in names:
        point = candidate[name]
        radius = VDW.get(name[0], 1.70)
        for other_index, other in enumerate(residues):
            if other_index == residue_index:
                continue
            for other_name, atom_index in other["atoms"].items():
                other_radius = VDW.get(other_name[0], 1.70)
                threshold = radius + other_radius - overlap
                distance = float(np.linalg.norm(point - coords[other_index, atom_index]))
                if distance < threshold:
                    energy += (threshold - distance) ** 2
    return energy


def _displacement(candidate, original):
    names = _sidechain_names(candidate)
    if not names:
        return 0.0
    return float(np.mean([np.sum((candidate[name] - original[name]) ** 2) for name in names]))


def _severe_pairs(residues, coords, threshold):
    pairs = []
    for left_index, left in enumerate(residues):
        for right_index in range(left_index + 1, len(residues)):
            right = residues[right_index]
            adjacent = (
                right_index == left_index + 1
                and left["key"][0] == right["key"][0]
                and right["key"][1] - left["key"][1] == 1
            )
            ca_left = left["atoms"].get("CA")
            ca_right = right["atoms"].get("CA")
            continuous = (
                adjacent
                and ca_left is not None
                and ca_right is not None
                and float(np.linalg.norm(coords[left_index, ca_left] - coords[right_index, ca_right]))
                <= 4.5
            )
            minimum = float("inf")
            minimum_margin = float("inf")
            closest = None
            for left_name, left_atom in left["atoms"].items():
                for right_name, right_atom in right["atoms"].items():
                    # The only covalent link between consecutive residues is
                    # C(i)-N(i+1).  Excluding whole adjacent residues hides
                    # genuine side-chain/backbone collisions.
                    if continuous and left_name == "C" and right_name == "N":
                        continue
                    distance = float(np.linalg.norm(coords[left_index, left_atom] - coords[right_index, right_atom]))
                    pair_threshold = (
                        PEPTIDE_O_N_MIN_DISTANCE
                        if continuous and left_name == "O" and right_name == "N"
                        else threshold
                    )
                    margin = distance - pair_threshold
                    if margin < minimum_margin:
                        minimum = distance
                        minimum_margin = margin
                        closest = (left_name, right_name)
            if minimum_margin < 0:
                pairs.append((left_index, right_index, minimum, closest))
    return pairs


def repair(input_path: Path, output_path: Path, *, overlap: float, trigger: float, displacement_weight: float, passes: int):
    lines, residues, records, coords_torch, _ = _parse(input_path)
    coords = coords_torch.numpy().copy()
    original = coords.copy()
    before = _severe_pairs(residues, coords, trigger)
    active_set = set()
    for left_index, right_index, _, atom_names in before:
        left_name, right_name = atom_names
        if left_name not in BACKBONE:
            active_set.add(left_index)
        if right_name not in BACKBONE:
            active_set.add(right_index)
    active = sorted(active_set)
    choices = []
    for pass_index in range(passes):
        changed = False
        for residue_index in active:
            residue = residues[residue_index]
            candidates = _candidate_atoms(residue, coords, residue_index)
            if not candidates:
                continue
            original_atoms = _residue_atoms(residue, original, residue_index)
            scores = [
                _clash_energy(candidate, residue_index, residues, coords, overlap)
                + displacement_weight * _displacement(candidate, original_atoms)
                for candidate in candidates
            ]
            best = int(np.argmin(scores))
            if best != 0 and scores[best] + 1e-9 < scores[0]:
                for name, point in candidates[best].items():
                    coords[residue_index, residue["atoms"][name]] = point
                changed = True
                choices.append(
                    {
                        "pass": pass_index + 1,
                        "residue_number": residue["key"][1],
                        "resname": residue["resname"],
                        "candidate": best,
                        "score_before": scores[0],
                        "score_after": scores[best],
                    }
                )
        if not changed:
            break

    for line_index, residue_index, atom_index, _ in records:
        x, y, z = coords[residue_index, atom_index]
        line = lines[line_index]
        lines[line_index] = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    after = _severe_pairs(residues, coords, trigger)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "overlap": overlap,
        "trigger": trigger,
        "displacement_weight": displacement_weight,
        "passes": passes,
        "severe_pairs_before": len(before),
        "severe_pairs_after": len(after),
        "before": [
            {
                "left_residue": residues[x[0]]["key"][1],
                "right_residue": residues[x[1]]["key"][1],
                "distance": x[2],
                "atoms": x[3],
            }
            for x in before
        ],
        "after": [
            {
                "left_residue": residues[x[0]]["key"][1],
                "right_residue": residues[x[1]]["key"][1],
                "distance": x[2],
                "atoms": x[3],
            }
            for x in after
        ],
        "choices": choices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--overlap", type=float, default=1.2)
    parser.add_argument("--trigger", type=float, default=1.8)
    parser.add_argument("--displacement-weight", type=float, default=0.02)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    result = repair(
        args.input,
        args.output,
        overlap=args.overlap,
        trigger=args.trigger,
        displacement_weight=args.displacement_weight,
        passes=args.passes,
    )
    if args.metadata:
        args.metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
