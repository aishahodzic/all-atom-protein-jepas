"""Differentiable stereochemical regularization for Atom37 decoder outputs.

This module deliberately does not alter the decoder architecture.  It maps the
decoder's Atom37 coordinates to the residue-specific Atom14 representation and
uses OpenFold's structural-violation loss.  The loss constrains covalent and
virtual bond geometry, peptide connections, and steric clashes.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from openfold.utils.loss import find_structural_violations

from proteinfoundation.utils.coors_utils import nm_to_ang


def _build_pair_tables(
    include_virtual: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residue_bonds, residue_virtual_bonds, _ = (
        residue_constants.load_stereo_chemical_props()
    )
    all_pairs = []
    for one_letter in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[one_letter]
        pairs = list(residue_bonds[residue_name])
        if include_virtual:
            pairs += list(residue_virtual_bonds[residue_name])
        all_pairs.append(pairs)
    all_pairs.append([])  # Unknown residue.

    max_pairs = max(len(pairs) for pairs in all_pairs)
    atom1 = torch.zeros((21, max_pairs), dtype=torch.long)
    atom2 = torch.zeros((21, max_pairs), dtype=torch.long)
    ideal = torch.zeros((21, max_pairs), dtype=torch.float32)
    exists = torch.zeros((21, max_pairs), dtype=torch.bool)
    for residue_type, pairs in enumerate(all_pairs):
        for pair_index, pair in enumerate(pairs):
            atom1[residue_type, pair_index] = residue_constants.atom_order[
                pair.atom1_name
            ]
            atom2[residue_type, pair_index] = residue_constants.atom_order[
                pair.atom2_name
            ]
            ideal[residue_type, pair_index] = pair.length
            exists[residue_type, pair_index] = True
    return atom1, atom2, ideal, exists


_COVALENT_PAIR_TABLES = _build_pair_tables(include_virtual=False)
_ALL_PAIR_TABLES = _build_pair_tables(include_virtual=True)


def _literature_pair_loss(
    coords_ang: torch.Tensor,
    aatype: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    virtual_only: bool,
    atom37_mask: torch.Tensor | None = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """Smooth-L1 error against covalent or angle-derived virtual bonds."""

    tables = _ALL_PAIR_TABLES if virtual_only else _COVALENT_PAIR_TABLES
    atom1, atom2, ideal, exists = [table.to(coords_ang.device) for table in tables]
    atom1 = atom1[aatype]
    atom2 = atom2[aatype]
    ideal = ideal[aatype].to(coords_ang.dtype)
    exists = exists[aatype] & residue_mask[..., None]

    if virtual_only:
        # The combined table is built with covalent pairs first for every type.
        per_type_covalent_count = _COVALENT_PAIR_TABLES[3].sum(dim=-1).to(
            coords_ang.device
        )[aatype]
        pair_number = torch.arange(atom1.shape[-1], device=coords_ang.device)
        virtual_mask = pair_number.view(*((1,) * (atom1.ndim - 1)), -1)
        virtual_mask = virtual_mask >= per_type_covalent_count[..., None]
        exists = exists & virtual_mask

    if atom37_mask is not None:
        atom1_exists = torch.gather(atom37_mask.bool(), dim=-1, index=atom1)
        atom2_exists = torch.gather(atom37_mask.bool(), dim=-1, index=atom2)
        exists = exists & atom1_exists & atom2_exists

    gather_shape = (*atom1.shape, 3)
    pos1 = torch.gather(
        coords_ang, dim=-2, index=atom1[..., None].expand(gather_shape)
    )
    pos2 = torch.gather(
        coords_ang, dim=-2, index=atom2[..., None].expand(gather_shape)
    )
    distance = torch.linalg.vector_norm(pos1 - pos2, dim=-1)
    error = F.smooth_l1_loss(distance, ideal, beta=beta, reduction="none")
    weights = exists.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp(min=1.0)


def _peptide_geometry_loss(
    coords_ang: torch.Tensor,
    aatype: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    atom37_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_mask = (
        residue_mask[:, :-1]
        & residue_mask[:, 1:]
        & ((residue_index[:, 1:] - residue_index[:, :-1]) == 1)
    )
    bond_edge_mask = edge_mask
    angle_edge_mask = edge_mask
    if atom37_mask is not None:
        bond_edge_mask = (
            bond_edge_mask
            & atom37_mask[:, :-1, 2].bool()
            & atom37_mask[:, 1:, 0].bool()
        )
        angle_edge_mask = (
            bond_edge_mask
            & atom37_mask[:, :-1, 1].bool()
            & atom37_mask[:, 1:, 1].bool()
        )
    bond_weights = bond_edge_mask.to(coords_ang.dtype)
    angle_weights = angle_edge_mask.to(coords_ang.dtype)

    ca_i = coords_ang[:, :-1, 1]
    c_i = coords_ang[:, :-1, 2]
    n_j = coords_ang[:, 1:, 0]
    ca_j = coords_ang[:, 1:, 1]

    c_n = torch.linalg.vector_norm(c_i - n_j, dim=-1)
    next_is_proline = aatype[:, 1:] == residue_constants.resname_to_idx["PRO"]
    ideal_c_n = torch.where(
        next_is_proline,
        c_n.new_tensor(residue_constants.between_res_bond_length_c_n[1]),
        c_n.new_tensor(residue_constants.between_res_bond_length_c_n[0]),
    )
    bond_error = F.smooth_l1_loss(c_n, ideal_c_n, beta=0.05, reduction="none")
    bond_loss = (bond_error * bond_weights).sum() / bond_weights.sum().clamp(min=1.0)

    def unit(vector: torch.Tensor) -> torch.Tensor:
        return vector / torch.linalg.vector_norm(
            vector, dim=-1, keepdim=True
        ).clamp(min=1e-6)

    ca_c_n_cos = (unit(ca_i - c_i) * unit(n_j - c_i)).sum(dim=-1)
    c_n_ca_cos = (unit(c_i - n_j) * unit(ca_j - n_j)).sum(dim=-1)
    ca_c_n_target = ca_c_n_cos.new_tensor(
        residue_constants.between_res_cos_angles_ca_c_n[0]
    ).expand_as(ca_c_n_cos)
    c_n_ca_target = c_n_ca_cos.new_tensor(
        residue_constants.between_res_cos_angles_c_n_ca[0]
    ).expand_as(c_n_ca_cos)
    angle_error = F.smooth_l1_loss(
        ca_c_n_cos, ca_c_n_target, beta=0.05, reduction="none"
    ) + F.smooth_l1_loss(
        c_n_ca_cos, c_n_ca_target, beta=0.05, reduction="none"
    )
    angle_loss = (
        (angle_error * angle_weights).sum() / angle_weights.sum().clamp(min=1.0)
    )
    return bond_loss, angle_loss


def build_geometry_residue_index(
    batch: Dict[str, torch.Tensor], residue_mask: torch.Tensor
) -> torch.Tensor:
    """Return monotonically increasing indices with gaps at chain breaks.

    OpenFold treats adjacent tensor rows as peptide neighbours only when their
    residue indices differ by one.  Dense protein batches can contain multiple
    chains, missing PDB indices, and padding, so those edges receive a gap of
    two here.  This prevents a peptide-bond loss across a real chain break.
    """

    residue_mask = residue_mask.bool()
    batch_size, length = residue_mask.shape
    if length == 0:
        return torch.empty_like(residue_mask, dtype=torch.long)

    continuous = residue_mask[:, :-1] & residue_mask[:, 1:]

    chains = batch.get("chains")
    if chains is not None:
        continuous = continuous & (chains[:, :-1] == chains[:, 1:])

    pdb_index = batch.get("residue_pdb_idx")
    if pdb_index is not None:
        continuous = continuous & ((pdb_index[:, 1:] - pdb_index[:, :-1]) == 1)

    chain_breaks = batch.get("chain_breaks_per_residue")
    if chain_breaks is not None:
        # The transform marks residue i when the i -> i+1 C-alpha edge breaks.
        continuous = continuous & ~chain_breaks[:, :-1].bool()

    increments = torch.where(
        continuous,
        torch.ones_like(continuous, dtype=torch.long),
        torch.full_like(continuous, 2, dtype=torch.long),
    )
    residue_index = torch.zeros(
        (batch_size, length), dtype=torch.long, device=residue_mask.device
    )
    if length > 1:
        residue_index[:, 1:] = torch.cumsum(increments, dim=-1)
    return residue_index


def _atom14_geometry_batch(
    coords_nm: torch.Tensor,
    aatype: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    atom37_mask: torch.Tensor | None = None,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Map Atom37 coordinates to the Atom14 layout for ``aatype``."""

    geometry_batch: Dict[str, torch.Tensor] = {"aatype": aatype.long()}
    make_atom14_masks(geometry_batch)

    gather_index = geometry_batch["residx_atom14_to_atom37"].unsqueeze(-1)
    gather_index = gather_index.expand(*gather_index.shape[:-1], 3)
    atom14_positions = torch.gather(
        nm_to_ang(coords_nm.float()), dim=-2, index=gather_index
    )

    atom14_exists = geometry_batch["atom14_atom_exists"]
    atom14_exists = atom14_exists * residue_mask[..., None].to(atom14_exists.dtype)
    if atom37_mask is not None:
        gathered_atom_mask = torch.gather(
            atom37_mask.to(atom14_exists.dtype),
            dim=-1,
            index=geometry_batch["residx_atom14_to_atom37"],
        )
        atom14_exists = atom14_exists * gathered_atom_mask
    geometry_batch["atom14_atom_exists"] = atom14_exists
    geometry_batch["residue_index"] = residue_index
    return geometry_batch, atom14_positions


def _one_sequence_violation_loss(
    coords_nm: torch.Tensor,
    aatype: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    tolerance_factor: float,
    clash_overlap_tolerance: float,
    atom37_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    coords_ang = nm_to_ang(coords_nm.float())
    covalent = _literature_pair_loss(
        coords_ang,
        aatype,
        residue_mask,
        virtual_only=False,
        atom37_mask=atom37_mask,
    )
    virtual = _literature_pair_loss(
        coords_ang,
        aatype,
        residue_mask,
        virtual_only=True,
        atom37_mask=atom37_mask,
    )
    peptide_bond, peptide_angles = _peptide_geometry_loss(
        coords_ang, aatype, residue_mask, residue_index, atom37_mask
    )

    geometry_batch, atom14_positions = _atom14_geometry_batch(
        coords_nm, aatype, residue_mask, residue_index, atom37_mask
    )
    violations = find_structural_violations(
        geometry_batch,
        atom14_positions,
        violation_tolerance_factor=tolerance_factor,
        clash_overlap_tolerance=clash_overlap_tolerance,
    )
    between = violations["between_residues"]
    within = violations["within_residues"]
    num_atoms = geometry_batch["atom14_atom_exists"].sum().clamp(min=1.0)
    between_clash = between["clashes_mean_loss"]
    within_violation = within["per_atom_loss_sum"].sum() / num_atoms
    clash = between_clash + within_violation
    total = covalent + virtual + peptide_bond + peptide_angles + 0.1 * clash
    components = {
        "covalent": covalent,
        "virtual": virtual,
        "peptide_bond": peptide_bond,
        "peptide_angles": peptide_angles,
        "between_clash": between_clash,
        "within_residue": within_violation,
    }
    return total, components


def stereochemical_loss(
    output_dec: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    predicted_sequence_weight: float = 0.5,
    tolerance_factor: float = 12.0,
    clash_overlap_tolerance: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """Compute true- and predicted-sequence structural-violation losses.

    Coordinate reconstruction is supervised with the true Atom37 mask.  If a
    different amino acid is predicted, however, its additional atom slots may
    otherwise be unconstrained and appear as knots in a PDB export.  Mixing the
    two topology losses keeps both the reconstruction and the actually exported
    predicted residue chemically regular.  ``aatype_max`` is intentionally
    detached: sequence identity continues to be learned by cross entropy, while
    this term regularizes coordinates for the selected topology.
    """

    if not 0.0 <= predicted_sequence_weight <= 1.0:
        raise ValueError("predicted_sequence_weight must be between zero and one")

    coords_nm = output_dec["coors_nm"]
    residue_mask = output_dec["residue_mask"].bool()
    residue_index = build_geometry_residue_index(batch, residue_mask)

    true_total, true_components = _one_sequence_violation_loss(
        coords_nm,
        batch["residue_type"],
        residue_mask,
        residue_index,
        tolerance_factor,
        clash_overlap_tolerance,
        batch.get("coord_mask"),
    )
    predicted_total, predicted_components = _one_sequence_violation_loss(
        coords_nm,
        output_dec["aatype_max"].detach(),
        residue_mask,
        residue_index,
        tolerance_factor,
        clash_overlap_tolerance,
        None,
    )

    predicted_weight = coords_nm.new_tensor(predicted_sequence_weight)
    true_weight = 1.0 - predicted_weight
    total = true_weight * true_total + predicted_weight * predicted_total

    result = {
        "total": total,
        "true_total": true_total,
        "predicted_total": predicted_total,
    }
    for name in true_components:
        result[name] = (
            true_weight * true_components[name]
            + predicted_weight * predicted_components[name]
        )
    return result
