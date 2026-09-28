from __future__ import annotations

import torch

from openfold.np import residue_constants

from Evaluation.structure_rec.restrained_geometry_repair import (
    _chirality_loss,
    _chirality_volumes,
    _continuous_edges,
    _nonbonded_pairs,
    _peptide_pairs,
    _peptide_omega_cosines,
    _trans_peptide_loss,
)
from Evaluation.structure_rec.rotamer_clash_repair import _severe_pairs


def _two_residue_geometry(ca_distance: float):
    residues = [
        {"key": ("A", 1, " "), "resname": "ALA", "atoms": {}},
        {"key": ("A", 2, " "), "resname": "ALA", "atoms": {}},
    ]
    coords = torch.zeros((2, 37, 3), dtype=torch.float64)
    mask = torch.zeros((2, 37), dtype=torch.bool)
    ca = residue_constants.atom_order["CA"]
    c = residue_constants.atom_order["C"]
    n = residue_constants.atom_order["N"]
    coords[1, ca, 0] = ca_distance
    mask[:, ca] = True
    mask[0, c] = True
    mask[1, n] = True
    return residues, coords, mask


def test_ca_gap_is_not_treated_as_a_peptide_edge():
    residues, coords, mask = _two_residue_geometry(ca_distance=8.0)
    edges = _continuous_edges(residues, coords, mask, ca_gap_threshold=4.5)
    assert edges == [False]
    assert _peptide_pairs(residues, mask, edges) == []


def test_contiguous_ca_trace_is_stitched():
    residues, coords, mask = _two_residue_geometry(ca_distance=3.8)
    edges = _continuous_edges(residues, coords, mask, ca_gap_threshold=4.5)
    assert edges == [True]
    assert len(_peptide_pairs(residues, mask, edges)) == 1


def test_adjacent_sidechain_clash_is_not_hidden_by_peptide_bond():
    residues, coords, mask = _two_residue_geometry(ca_distance=3.8)
    c = residue_constants.atom_order["C"]
    n = residue_constants.atom_order["N"]
    cb = residue_constants.atom_order["CB"]
    residues[0]["atoms"] = {"CA": 1, "C": c, "CB": cb}
    residues[1]["atoms"] = {"CA": 1, "N": n, "CB": cb}
    coords[0, c] = torch.tensor([1.50, 0.0, 0.0])
    coords[1, n] = torch.tensor([2.83, 0.0, 0.0])  # Valid peptide C-N.
    coords[0, cb] = torch.tensor([0.0, 1.0, 0.0])
    coords[1, cb] = torch.tensor([0.0, 1.5, 0.0])  # Impossible overlap.
    mask[:, cb] = True

    severe = _severe_pairs(residues, coords.numpy(), threshold=1.8)
    assert len(severe) == 1
    assert severe[0][3] == ("CB", "CB")

    edges = _continuous_edges(residues, coords, mask, ca_gap_threshold=4.5)
    nonbonded = _nonbonded_pairs(
        residues,
        coords,
        mask,
        edges,
        overlap=1.2,
        candidate_cutoff=5.0,
    )
    assert not any(pair[1] == c and pair[3] == n for pair in nonbonded)
    assert any(pair[1] == cb and pair[3] == cb for pair in nonbonded)


def test_short_peptide_oxygen_to_next_nitrogen_is_flagged():
    residues, coords, mask = _two_residue_geometry(ca_distance=3.8)
    c = residue_constants.atom_order["C"]
    n = residue_constants.atom_order["N"]
    o = residue_constants.atom_order["O"]
    residues[0]["atoms"] = {"CA": 1, "C": c, "O": o}
    residues[1]["atoms"] = {"CA": 1, "N": n}
    coords[0, c] = torch.tensor([1.50, 0.0, 0.0])
    coords[1, n] = torch.tensor([2.83, 0.0, 0.0])  # Valid peptide C-N.
    coords[0, o] = torch.tensor([0.96, 0.0, 0.0])  # O-N = 1.87 A.
    mask[0, o] = True

    severe = _severe_pairs(residues, coords.numpy(), threshold=1.8)
    assert len(severe) == 1
    assert severe[0][3] == ("O", "N")

    edges = _continuous_edges(residues, coords, mask, ca_gap_threshold=4.5)
    nonbonded = _nonbonded_pairs(
        residues,
        coords,
        mask,
        edges,
        overlap=1.2,
        candidate_cutoff=5.0,
    )
    on_pair = next(pair for pair in nonbonded if pair[1] == o and pair[3] == n)
    assert on_pair[4] == 2.20


def test_chirality_loss_rejects_inverted_ca_center():
    coords = torch.zeros((1, 37, 3), dtype=torch.float64)
    n = residue_constants.atom_order["N"]
    ca = residue_constants.atom_order["CA"]
    c = residue_constants.atom_order["C"]
    cb = residue_constants.atom_order["CB"]
    coords[0, n] = torch.tensor([1.0, 0.0, 0.0])
    coords[0, ca] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, c] = torch.tensor([0.0, 1.0, 0.0])
    coords[0, cb] = torch.tensor([0.0, 0.0, 1.0])
    target = _chirality_volumes(coords, [0]).abs()
    assert _chirality_loss(coords, [0], target, minimum_ratio=0.75) == 0

    coords[0, cb, 2] = -1.0
    assert _chirality_loss(coords, [0], target, minimum_ratio=0.75) > 0


def test_trans_peptide_loss_rejects_cis_geometry():
    coords = torch.zeros((2, 37, 3), dtype=torch.float64)
    ca = residue_constants.atom_order["CA"]
    c = residue_constants.atom_order["C"]
    n = residue_constants.atom_order["N"]
    coords[0, ca] = torch.tensor([0.0, 1.0, 0.0])
    coords[0, c] = torch.tensor([0.0, 0.0, 0.0])
    coords[1, n] = torch.tensor([1.0, 0.0, 0.0])
    coords[1, ca] = torch.tensor([1.0, -1.0, 0.0])
    assert torch.allclose(
        _peptide_omega_cosines(coords, [0]),
        torch.tensor([-1.0], dtype=torch.float64),
    )
    assert _trans_peptide_loss(coords, [0], minimum_angle_degrees=150.0) == 0

    coords[1, ca] = torch.tensor([1.0, 1.0, 0.0])
    assert torch.allclose(
        _peptide_omega_cosines(coords, [0]),
        torch.tensor([1.0], dtype=torch.float64),
    )
    assert _trans_peptide_loss(coords, [0], minimum_angle_degrees=150.0) > 0
