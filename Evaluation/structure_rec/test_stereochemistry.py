from __future__ import annotations

import torch

from openfold.np import residue_constants

from Evaluation.structure_rec.stereochemistry import (
    build_geometry_residue_index,
    stereochemical_loss,
)


ARG = residue_constants.restype_order["R"]
ALA = residue_constants.restype_order["A"]


def _atom37(residue_name: str, positions_angstrom: dict[str, tuple[float, float, float]]):
    coords = torch.zeros((1, 1, 37, 3), dtype=torch.float32)
    mask = torch.zeros((1, 1, 37), dtype=torch.bool)
    for atom_name, position in positions_angstrom.items():
        atom_index = residue_constants.atom_order[atom_name]
        coords[0, 0, atom_index] = torch.tensor(position) / 10.0
        mask[0, 0, atom_index] = True
    aatype = torch.tensor(
        [[residue_constants.restype_order[residue_constants.restype_3to1[residue_name]]]]
    )
    return coords, mask, aatype


ARG_NATIVE = {
    "N": (-1.529, -1.231, -5.325),
    "CA": (-2.402, -2.278, -5.849),
    "C": (-3.170, -1.743, -7.052),
    "CB": (-3.377, -2.708, -4.751),
    "O": (-3.245, -2.392, -8.096),
    "CG": (-4.394, -3.777, -5.138),
    "CD": (-3.731, -5.069, -5.593),
    "NE": (-4.230, -6.232, -4.863),
    "NH1": (-3.035, -5.756, -2.956),
    "NH2": (-4.393, -7.609, -3.024),
    "CZ": (-3.886, -6.533, -3.615),
}

ARG_KNOT = {
    "N": (-1.476, -1.116, -5.221),
    "CA": (-2.402, -2.278, -5.849),
    "C": (-3.249, -1.601, -6.999),
    "CB": (-3.224, -2.981, -4.758),
    "O": (-3.457, -2.175, -8.059),
    "CG": (-3.214, -3.925, -4.309),
    "CD": (-3.986, -4.966, -3.912),
    "NE": (-3.667, -5.870, -4.325),
    "NH1": (-3.645, -6.281, -3.908),
    "NH2": (-3.758, -6.989, -4.269),
    "CZ": (-3.427, -6.081, -3.410),
}


def _batch(coords, atom_mask, true_aatype):
    return {
        "coords_nm": coords.detach().clone(),
        "coord_mask": atom_mask,
        "residue_type": true_aatype,
        "chains": torch.zeros_like(true_aatype),
        "residue_pdb_idx": torch.zeros_like(true_aatype),
        "chain_breaks_per_residue": torch.zeros_like(true_aatype, dtype=torch.bool),
    }


def _loss(coords, atom_mask, true_aatype, predicted_aatype=None, predicted_weight=0.5):
    predicted_aatype = true_aatype if predicted_aatype is None else predicted_aatype
    output = {
        "coors_nm": coords,
        "aatype_max": predicted_aatype,
        "residue_mask": torch.ones_like(true_aatype, dtype=torch.bool),
    }
    return stereochemical_loss(
        output,
        _batch(coords, atom_mask, true_aatype),
        predicted_sequence_weight=predicted_weight,
    )


def test_arginine_knot_is_penalized_and_has_finite_gradients():
    native, atom_mask, aatype = _atom37("ARG", ARG_NATIVE)
    knot, _, _ = _atom37("ARG", ARG_KNOT)

    native_loss = _loss(native, atom_mask, aatype)["total"]
    knot.requires_grad_()
    knot_loss = _loss(knot, atom_mask, aatype)["total"]

    assert native_loss.item() < 0.01
    assert knot_loss.item() > 0.5
    assert knot_loss.item() > 100 * native_loss.item()

    knot_loss.backward()
    assert knot.grad is not None
    assert torch.isfinite(knot.grad).all()
    assert knot.grad.norm().item() > 0


def test_predicted_residue_topology_regularizes_new_sidechain_atoms():
    knot, atom_mask, arg = _atom37("ARG", ARG_KNOT)
    true_ala = torch.tensor([[ALA]])
    ala_atom_mask = torch.zeros_like(atom_mask)
    for atom_name in residue_constants.residue_atoms["ALA"]:
        ala_atom_mask[..., residue_constants.atom_order[atom_name]] = True

    true_only = _loss(
        knot, ala_atom_mask, true_ala, predicted_aatype=arg, predicted_weight=0.0
    )["total"]
    predicted_only = _loss(
        knot, ala_atom_mask, true_ala, predicted_aatype=arg, predicted_weight=1.0
    )["total"]

    assert predicted_only.item() > true_only.item() + 0.5


def test_geometry_indices_break_at_chain_gaps_missing_residues_and_padding():
    residue_mask = torch.tensor([[True, True, True, True, False]])
    batch = {
        "chains": torch.tensor([[0, 0, 0, 1, 0]]),
        "residue_pdb_idx": torch.tensor([[10, 11, 15, 1, 0]]),
        "chain_breaks_per_residue": torch.tensor(
            [[False, False, False, False, False]]
        ),
    }
    residue_index = build_geometry_residue_index(batch, residue_mask)
    assert residue_index.tolist() == [[0, 1, 3, 5, 7]]


def test_padding_does_not_change_single_residue_loss():
    coords, atom_mask, aatype = _atom37("ARG", ARG_KNOT)
    base = _loss(coords, atom_mask, aatype)["total"]

    padded_coords = torch.cat((coords, torch.zeros_like(coords)), dim=1)
    padded_atom_mask = torch.cat((atom_mask, torch.zeros_like(atom_mask)), dim=1)
    padded_aatype = torch.cat((aatype, torch.tensor([[ALA]])), dim=1)
    output = {
        "coors_nm": padded_coords,
        "aatype_max": padded_aatype,
        "residue_mask": torch.tensor([[True, False]]),
    }
    batch = {
        "coords_nm": padded_coords.clone(),
        "coord_mask": padded_atom_mask,
        "residue_type": padded_aatype,
        "chains": torch.tensor([[0, 0]]),
        "residue_pdb_idx": torch.tensor([[0, 0]]),
        "chain_breaks_per_residue": torch.tensor([[False, False]]),
    }
    padded = stereochemical_loss(output, batch)["total"]
    torch.testing.assert_close(base, padded)
