"""Minimal-displacement repair of decoder geometry violations in a PDB.

This is deliberately post-processing, not a decoder architecture change.  It
keeps every C-alpha fixed and solves for coordinates close to the decoder
output while enforcing residue geometry, peptide continuity, and non-bonded
sterics.  C-alpha gaps are treated as real chain breaks, never stitched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from openfold.np import residue_constants


def _parse(path: Path):
    lines = path.read_text().splitlines()
    residues = []
    by_key = {}
    records = []
    for line_index, line in enumerate(lines):
        if not line.startswith("ATOM"):
            continue
        key = (line[21], int(line[22:26]), line[26])
        if key not in by_key:
            resname = line[17:20].strip()
            by_key[key] = len(residues)
            residues.append({"key": key, "resname": resname, "atoms": {}})
        residue_index = by_key[key]
        atom_name = line[12:16].strip()
        if atom_name not in residue_constants.atom_order:
            continue
        atom_index = residue_constants.atom_order[atom_name]
        coordinate = torch.tensor(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=torch.float64,
        )
        residues[residue_index]["atoms"][atom_name] = atom_index
        records.append((line_index, residue_index, atom_index, coordinate))

    coords = torch.zeros((len(residues), 37, 3), dtype=torch.float64)
    mask = torch.zeros((len(residues), 37), dtype=torch.bool)
    for _, residue_index, atom_index, coordinate in records:
        coords[residue_index, atom_index] = coordinate
        mask[residue_index, atom_index] = True
    return lines, residues, records, coords, mask


def _pairs(residues, mask, include_virtual: bool):
    bonds, virtual_bonds, _ = residue_constants.load_stereo_chemical_props()
    source = virtual_bonds if include_virtual else bonds
    pairs = []
    for residue_index, residue in enumerate(residues):
        for bond in source.get(residue["resname"], []):
            atom1 = residue_constants.atom_order[bond.atom1_name]
            atom2 = residue_constants.atom_order[bond.atom2_name]
            if mask[residue_index, atom1] and mask[residue_index, atom2]:
                pairs.append((residue_index, atom1, residue_index, atom2, bond.length))
    return pairs


BACKBONE_MOVABLE = {"N", "C", "O", "OXT"}
VDW_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
PEPTIDE_O_N_MIN_DISTANCE = 2.20


def _continuous_edges(residues, coords, mask, ca_gap_threshold: float):
    ca_atom = residue_constants.atom_order["CA"]
    edges = []
    for left_index, (left, right) in enumerate(zip(residues[:-1], residues[1:])):
        right_index = left_index + 1
        same_chain = left["key"][0] == right["key"][0]
        consecutive = right["key"][1] - left["key"][1] == 1
        ca_present = mask[left_index, ca_atom] and mask[right_index, ca_atom]
        ca_distance = (
            torch.linalg.vector_norm(
                coords[left_index, ca_atom] - coords[right_index, ca_atom]
            ).item()
            if ca_present
            else math.inf
        )
        edges.append(
            bool(
                same_chain
                and consecutive
                and ca_present
                and ca_distance <= ca_gap_threshold
            )
        )
    return edges


def _peptide_pairs(residues, mask, continuous_edges):
    pairs = []
    for left_index, (left, right) in enumerate(zip(residues[:-1], residues[1:])):
        right_index = left_index + 1
        c_atom = residue_constants.atom_order["C"]
        n_atom = residue_constants.atom_order["N"]
        if not (
            continuous_edges[left_index]
            and mask[left_index, c_atom]
            and mask[right_index, n_atom]
        ):
            continue
        next_is_proline = right["resname"] == "PRO"
        ideal = residue_constants.between_res_bond_length_c_n[next_is_proline]
        pairs.append((left_index, c_atom, right_index, n_atom, ideal))
    return pairs


def _peptide_angle_loss(coords, peptide_edges, tolerance):
    """Hinge loss for CA-C-N and C-N-CA peptide angles."""
    if not peptide_edges:
        return coords.new_zeros(())
    ca_atom = residue_constants.atom_order["CA"]
    c_atom = residue_constants.atom_order["C"]
    n_atom = residue_constants.atom_order["N"]
    left = torch.tensor(peptide_edges, dtype=torch.long, device=coords.device)
    right = left + 1

    def unit(vector):
        return vector / torch.linalg.vector_norm(
            vector, dim=-1, keepdim=True
        ).clamp(min=1e-8)

    ca_i = coords[left, ca_atom]
    c_i = coords[left, c_atom]
    n_j = coords[right, n_atom]
    ca_j = coords[right, ca_atom]
    ca_c_n_cos = (unit(ca_i - c_i) * unit(n_j - c_i)).sum(dim=-1)
    c_n_ca_cos = (unit(c_i - n_j) * unit(ca_j - n_j)).sum(dim=-1)
    target_ca_c_n = coords.new_tensor(
        residue_constants.between_res_cos_angles_ca_c_n[0]
    )
    target_c_n_ca = coords.new_tensor(
        residue_constants.between_res_cos_angles_c_n_ca[0]
    )
    errors = torch.cat(
        [
            torch.abs(ca_c_n_cos - target_ca_c_n),
            torch.abs(c_n_ca_cos - target_c_n_ca),
        ]
    )
    return torch.relu(errors - tolerance).square().mean()


def _chirality_volumes(coords, residue_indices):
    """Signed N-CA-C-CB volumes; positive values are L-amino-acid geometry."""
    if not residue_indices:
        return coords.new_empty((0,))
    indices = torch.tensor(residue_indices, dtype=torch.long, device=coords.device)
    n_atom = residue_constants.atom_order["N"]
    ca_atom = residue_constants.atom_order["CA"]
    c_atom = residue_constants.atom_order["C"]
    cb_atom = residue_constants.atom_order["CB"]
    n_ca = coords[indices, n_atom] - coords[indices, ca_atom]
    c_ca = coords[indices, c_atom] - coords[indices, ca_atom]
    cb_ca = coords[indices, cb_atom] - coords[indices, ca_atom]
    return (n_ca * torch.linalg.cross(c_ca, cb_ca, dim=-1)).sum(dim=-1)


def _chirality_loss(coords, residue_indices, target_magnitudes, minimum_ratio):
    if not residue_indices:
        return coords.new_zeros(())
    volumes = _chirality_volumes(coords, residue_indices)
    ratio = volumes / target_magnitudes.to(coords.device)
    return torch.relu(minimum_ratio - ratio).square().mean()


def _peptide_omega_cosines(coords, peptide_edges):
    """Cosines of CA(i)-C(i)-N(i+1)-CA(i+1) torsion angles."""
    if not peptide_edges:
        return coords.new_empty((0,))
    ca_atom = residue_constants.atom_order["CA"]
    c_atom = residue_constants.atom_order["C"]
    n_atom = residue_constants.atom_order["N"]
    left = torch.tensor(peptide_edges, dtype=torch.long, device=coords.device)
    right = left + 1
    b0 = coords[left, ca_atom] - coords[left, c_atom]
    b1 = coords[right, n_atom] - coords[left, c_atom]
    b2 = coords[right, ca_atom] - coords[right, n_atom]
    axis = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp(min=1e-8)
    v = b0 - (b0 * axis).sum(dim=-1, keepdim=True) * axis
    w = b2 - (b2 * axis).sum(dim=-1, keepdim=True) * axis
    return (v * w).sum(dim=-1) / (
        torch.linalg.vector_norm(v, dim=-1)
        * torch.linalg.vector_norm(w, dim=-1)
    ).clamp(min=1e-8)


def _trans_peptide_loss(coords, peptide_edges, minimum_angle_degrees):
    if not peptide_edges:
        return coords.new_zeros(())
    cosines = _peptide_omega_cosines(coords, peptide_edges)
    maximum_cosine = math.cos(math.radians(minimum_angle_degrees))
    # Trans is 180 degrees (cos=-1).  Penalize only torsions below the
    # permitted absolute angle, leaving already-trans peptides untouched.
    return torch.relu(cosines - maximum_cosine).square().mean()


def _element(atom_name: str) -> str:
    # Atom37 names in this project contain no leading digit/hydrogen records.
    return atom_name[0]


def _nonbonded_pairs(
    residues,
    raw,
    mask,
    continuous_edges,
    overlap,
    candidate_cutoff,
):
    """Preselect nearby non-bonded atom pairs from non-neighbouring residues."""
    atom_entries = []
    for residue_index, residue in enumerate(residues):
        for atom_name, atom_index in residue["atoms"].items():
            if mask[residue_index, atom_index]:
                atom_entries.append((residue_index, atom_index, atom_name))

    pairs = []
    for left_entry, (left_residue, left_atom, left_name) in enumerate(atom_entries):
        for right_residue, right_atom, right_name in atom_entries[left_entry + 1 :]:
            if left_residue == right_residue:
                continue
            peptide_bond = (
                right_residue == left_residue + 1
                and continuous_edges[left_residue]
                and left_name == "C"
                and right_name == "N"
            )
            if peptide_bond:
                continue
            distance = torch.linalg.vector_norm(
                raw[left_residue, left_atom] - raw[right_residue, right_atom]
            ).item()
            if distance > candidate_cutoff:
                continue
            threshold = (
                VDW_RADIUS.get(_element(left_name), 1.70)
                + VDW_RADIUS.get(_element(right_name), 1.70)
                - overlap
            )
            if (
                right_residue == left_residue + 1
                and continuous_edges[left_residue]
                and left_name == "O"
                and right_name == "N"
            ):
                # O(i)-N(i+1) is not covalent.  If it is too short, viewers
                # infer a false second peptide bond and draw a triangle.  The
                # held-out native structures have a minimum of 2.183 A.
                threshold = max(threshold, PEPTIDE_O_N_MIN_DISTANCE)
            pairs.append(
                (left_residue, left_atom, right_residue, right_atom, threshold)
            )
    return pairs


def _pair_loss(coords, pairs, tolerance):
    if not pairs:
        return coords.new_zeros(())
    r1 = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    a1 = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    r2 = torch.tensor([p[2] for p in pairs], dtype=torch.long)
    a2 = torch.tensor([p[3] for p in pairs], dtype=torch.long)
    ideal = coords.new_tensor([p[4] for p in pairs])
    distance = torch.linalg.vector_norm(coords[r1, a1] - coords[r2, a2], dim=-1)
    excess = torch.relu(torch.abs(distance - ideal) - tolerance)
    return excess.square().mean()


def _clash_loss(coords, pairs):
    if not pairs:
        return coords.new_zeros(())
    r1 = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=coords.device)
    a1 = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=coords.device)
    r2 = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=coords.device)
    a2 = torch.tensor([p[3] for p in pairs], dtype=torch.long, device=coords.device)
    minimum = coords.new_tensor([p[4] for p in pairs])
    distance = torch.linalg.vector_norm(coords[r1, a1] - coords[r2, a2], dim=-1)
    return torch.relu(minimum - distance).square().mean()


def _write_with_chain_breaks(lines, residues, continuous_edges, output_path):
    """Write coordinates with TER records at inferred C-alpha chain breaks."""
    break_after = {
        residues[index]["key"]
        for index, continuous in enumerate(continuous_edges)
        if not continuous
    }
    output = []
    inserted = 0
    serial = 0
    last_atom = None
    atom_lines = [index for index, line in enumerate(lines) if line.startswith("ATOM")]
    last_atom_for_key = {}
    for line_index in atom_lines:
        line = lines[line_index]
        key = (line[21], int(line[22:26]), line[26])
        last_atom_for_key[key] = line_index
    for line_index, line in enumerate(lines):
        # Remove old TER records; they are regenerated deterministically below.
        if line.startswith("TER"):
            continue
        if line.startswith("ATOM"):
            serial += 1
            line = line[:6] + f"{serial:5d}" + line[11:]
            last_atom = line
        output.append(line)
        if line_index in last_atom_for_key.values():
            key = (line[21], int(line[22:26]), line[26])
            if key in break_after:
                serial += 1
                output.append(
                    f"TER   {serial:5d}      {line[17:20]} {line[21]}{line[22:27]}"
                )
                inserted += 1
    # Retain an explicit terminal marker before ENDMDL/END as well.
    if last_atom is not None:
        serial += 1
        terminal = (
            f"TER   {serial:5d}      {last_atom[17:20]} "
            f"{last_atom[21]}{last_atom[22:27]}"
        )
        insert_at = next(
            (i for i, line in enumerate(output) if line.startswith(("ENDMDL", "END"))),
            len(output),
        )
        output.insert(insert_at, terminal)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output) + "\n")
    return inserted


def repair(
    input_path: Path,
    output_path: Path,
    *,
    strength: float,
    steps: int,
    learning_rate: float,
    bond_tolerance: float,
    virtual_tolerance: float,
    peptide_tolerance: float,
    peptide_angle_tolerance: float = 0.03,
    clash_strength: float = 1.0,
    clash_overlap: float = 1.2,
    clash_candidate_cutoff: float = 5.0,
    ca_gap_threshold: float = 4.5,
    backbone_steps: int | None = None,
    chirality_minimum_ratio: float = 0.75,
    trans_peptide_minimum_angle: float = 150.0,
    covalent_weight: float = 10.0,
    virtual_weight: float = 1.0,
    peptide_weight: float = 10.0,
) -> dict:
    lines, residues, records, raw, mask = _parse(input_path)
    covalent = _pairs(residues, mask, include_virtual=False)
    virtual = _pairs(residues, mask, include_virtual=True)
    continuous_edges = _continuous_edges(
        residues, raw, mask, ca_gap_threshold=ca_gap_threshold
    )
    peptide_edge_indices = [
        index for index, continuous in enumerate(continuous_edges) if continuous
    ]
    trans_peptide_edges = [
        index
        for index in peptide_edge_indices
        if residues[index + 1]["resname"] != "PRO"
    ]
    peptide = _peptide_pairs(residues, mask, continuous_edges)
    nonbonded = _nonbonded_pairs(
        residues,
        raw,
        mask,
        continuous_edges,
        clash_overlap,
        clash_candidate_cutoff,
    )
    chirality_atoms = tuple(
        residue_constants.atom_order[name] for name in ("N", "CA", "C", "CB")
    )
    chirality_residues = [
        index
        for index, residue in enumerate(residues)
        if residue["resname"] != "GLY"
        and all(mask[index, atom] for atom in chirality_atoms)
    ]
    raw_chirality = _chirality_volumes(raw, chirality_residues)
    chirality_targets = raw_chirality.abs().clamp(min=2.0)

    delta = torch.zeros_like(raw, requires_grad=True)
    all_movable = mask[..., None].to(raw.dtype)
    all_movable[:, residue_constants.atom_order["CA"], :] = 0.0
    backbone_movable = torch.zeros_like(all_movable)
    for atom_name in BACKBONE_MOVABLE:
        atom_index = residue_constants.atom_order.get(atom_name)
        if atom_index is not None:
            backbone_movable[:, atom_index, :] = all_movable[:, atom_index, :]
    if backbone_steps is None:
        backbone_steps = steps // 2
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    history = []
    for step in range(steps):
        movable = backbone_movable if step < backbone_steps else all_movable
        coords = raw + delta * movable
        movable_atoms = movable[..., 0].sum().clamp(min=1.0)
        displacement = ((coords - raw).square() * movable).sum() / movable_atoms
        covalent_loss = _pair_loss(coords, covalent, bond_tolerance)
        virtual_loss = _pair_loss(coords, virtual, virtual_tolerance)
        peptide_loss = _pair_loss(coords, peptide, peptide_tolerance)
        peptide_angle_loss = _peptide_angle_loss(
            coords, peptide_edge_indices, peptide_angle_tolerance
        )
        chirality_loss = _chirality_loss(
            coords,
            chirality_residues,
            chirality_targets,
            chirality_minimum_ratio,
        )
        trans_peptide_loss = _trans_peptide_loss(
            coords,
            trans_peptide_edges,
            trans_peptide_minimum_angle,
        )
        clash_loss = _clash_loss(coords, nonbonded) if step >= backbone_steps else coords.new_zeros(())
        geometry = (
            covalent_weight * covalent_loss
            + virtual_weight * virtual_loss
            + peptide_weight * peptide_loss
            + peptide_angle_loss
            + chirality_loss
            + trans_peptide_loss
        )
        loss = displacement + strength * geometry + clash_strength * clash_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in {0, max(backbone_steps - 1, 0), backbone_steps, steps - 1}:
            history.append(
                {
                    "step": step + 1,
                    "stage": "peptide" if step < backbone_steps else "relaxation",
                    "loss": float(loss.detach()),
                    "displacement": float(displacement.detach()),
                    "covalent": float(covalent_loss.detach()),
                    "virtual": float(virtual_loss.detach()),
                    "peptide": float(peptide_loss.detach()),
                    "peptide_angles": float(peptide_angle_loss.detach()),
                    "chirality": float(chirality_loss.detach()),
                    "trans_peptide": float(trans_peptide_loss.detach()),
                    "clash": float(clash_loss.detach()),
                }
            )

    repaired = raw + delta.detach() * all_movable
    repaired_chirality = _chirality_volumes(repaired, chirality_residues)
    repaired_omega_cosines = _peptide_omega_cosines(repaired, trans_peptide_edges)
    for line_index, residue_index, atom_index, _ in records:
        x, y, z = repaired[residue_index, atom_index].tolist()
        line = lines[line_index]
        lines[line_index] = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
    inserted_breaks = _write_with_chain_breaks(
        lines, residues, continuous_edges, output_path
    )
    ca_atom = residue_constants.atom_order["CA"]
    ca_shift = torch.linalg.vector_norm(
        repaired[:, ca_atom] - raw[:, ca_atom], dim=-1
    ).max()
    moved = torch.linalg.vector_norm(repaired - raw, dim=-1)[mask]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "strength": strength,
        "steps": steps,
        "learning_rate": learning_rate,
        "bond_tolerance_angstrom": bond_tolerance,
        "virtual_tolerance_angstrom": virtual_tolerance,
        "peptide_tolerance_angstrom": peptide_tolerance,
        "peptide_angle_cosine_tolerance": peptide_angle_tolerance,
        "clash_strength": clash_strength,
        "clash_overlap_angstrom": clash_overlap,
        "ca_gap_threshold_angstrom": ca_gap_threshold,
        "backbone_steps": backbone_steps,
        "chirality_minimum_ratio": chirality_minimum_ratio,
        "trans_peptide_minimum_angle_degrees": trans_peptide_minimum_angle,
        "covalent_weight": covalent_weight,
        "virtual_weight": virtual_weight,
        "peptide_weight": peptide_weight,
        "chirality_centers": len(chirality_residues),
        "inverted_chirality_before": int((raw_chirality <= 0).sum()),
        "inverted_chirality_after": int((repaired_chirality <= 0).sum()),
        "nonproline_cis_like_before": int(
            (_peptide_omega_cosines(raw, trans_peptide_edges) > 0).sum()
        ),
        "nonproline_cis_like_after": int((repaired_omega_cosines > 0).sum()),
        "continuous_peptide_edges": len(peptide),
        "inserted_internal_ter_records": inserted_breaks,
        "max_ca_shift_angstrom": float(ca_shift),
        "mean_moved_atom_distance_angstrom": float(moved.mean()),
        "max_moved_atom_distance_angstrom": float(moved.max()),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--bond-tolerance", type=float, default=0.15)
    parser.add_argument("--virtual-tolerance", type=float, default=0.20)
    parser.add_argument("--peptide-tolerance", type=float, default=0.15)
    parser.add_argument("--peptide-angle-tolerance", type=float, default=0.03)
    parser.add_argument("--clash-strength", type=float, default=1.0)
    parser.add_argument("--clash-overlap", type=float, default=1.2)
    parser.add_argument("--clash-candidate-cutoff", type=float, default=5.0)
    parser.add_argument("--ca-gap-threshold", type=float, default=4.5)
    parser.add_argument("--backbone-steps", type=int)
    parser.add_argument("--chirality-minimum-ratio", type=float, default=0.75)
    parser.add_argument(
        "--trans-peptide-minimum-angle", type=float, default=150.0
    )
    parser.add_argument("--covalent-weight", type=float, default=10.0)
    parser.add_argument("--virtual-weight", type=float, default=1.0)
    parser.add_argument("--peptide-weight", type=float, default=10.0)
    args = parser.parse_args()
    result = repair(
        args.input,
        args.output,
        strength=args.strength,
        steps=args.steps,
        learning_rate=args.learning_rate,
        bond_tolerance=args.bond_tolerance,
        virtual_tolerance=args.virtual_tolerance,
        peptide_tolerance=args.peptide_tolerance,
        peptide_angle_tolerance=args.peptide_angle_tolerance,
        clash_strength=args.clash_strength,
        clash_overlap=args.clash_overlap,
        clash_candidate_cutoff=args.clash_candidate_cutoff,
        ca_gap_threshold=args.ca_gap_threshold,
        backbone_steps=args.backbone_steps,
        chirality_minimum_ratio=args.chirality_minimum_ratio,
        trans_peptide_minimum_angle=args.trans_peptide_minimum_angle,
        covalent_weight=args.covalent_weight,
        virtual_weight=args.virtual_weight,
        peptide_weight=args.peptide_weight,
    )
    if args.metadata:
        args.metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
