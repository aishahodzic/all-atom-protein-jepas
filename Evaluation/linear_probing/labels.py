"""Per-residue labels for linear probing, computed from a RAW (Angstrom,
uncentered, atom37-openfold-ordered) PDBDataset graph -- i.e. the object
data_pool.ProteinPool.dataset[i] returns, BEFORE eval_transform_compose()
is applied for the encoder's input.

Labels:
  - aa:        residue identity, graph.residue_type directly (int 0..19,
               already the checkpoint-agnostic mapping pdb_data.py uses --
               not re-derived from residue names, so no risk of a mismatched
               convention).
  - bfactor_z: graph.bfactor (graphein's own per-residue value: mean B-factor
               over whichever atoms of that residue are resolved -- see
               io.py:280-285's `groupby("residue_id")["b_factor"].mean()`;
               there is no atom-level bfactor tensor in this pipeline, so
               "CA-only" isn't an available option), z-scored within this
               protein.
  - ss:        3-state secondary structure via biotite's annotate_sse
               (P-SEA algorithm, CA-trace only, returns 'a'/'b'/'c' directly).
  - rsa:       relative solvent accessibility: biotite Shrake-Rupley SASA
               (ProtOr VdW radii) summed per residue, divided by the Tien et
               al. 2013 empirical max-ASA per residue type, clipped to
               [0, 1]. Values below supplied directly (2026-08-08), replacing
               an earlier from-memory attempt that was wrong.

Every label array is length n_residues, plus a same-length boolean validity
mask per property (aa/bfactor are always valid; ss/rsa can be invalid for a
residue with no resolved CA -- annotate_sse returns '' for those).
"""

from typing import Dict

import numpy as np

from openfold.np.residue_constants import atom_types

CA_SLOT = atom_types.index("CA")

# Tien et al. 2013 (PLOS ONE), Table 1, "Empirical" column. Angstrom^2.
MAX_ASA_EMPIRICAL: Dict[str, float] = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0, "CYS": 148.0,
    "GLN": 214.0, "GLU": 214.0, "GLY": 97.0, "HIS": 216.0, "ILE": 195.0,
    "LEU": 191.0, "LYS": 230.0, "MET": 203.0, "PHE": 228.0, "PRO": 154.0,
    "SER": 143.0, "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0,
}

SS_STATE_TO_INT = {"a": 0, "b": 1, "c": 2}  # helix / strand / coil


def _build_atom_array(coords: np.ndarray, coord_mask: np.ndarray, residues, residue_pdb_idx: np.ndarray):
    """coords: [n,37,3] Angstrom. coord_mask: [n,37] bool. residues: list[str]
    len n (3-letter codes). residue_pdb_idx: [n] int. Only atoms with
    coord_mask=True are included (both "chemically absent for this residue
    type" and "unresolved in the structure" show up as False -- see
    pdb_data.py:697-698)."""
    import biotite.structure as struc

    n_atoms = int(coord_mask.sum())
    atom_array = struc.AtomArray(n_atoms)
    pos = 0
    res_id_out = []
    for i in range(coords.shape[0]):
        for slot in np.nonzero(coord_mask[i])[0]:
            atom_array.coord[pos] = coords[i, slot]
            name = atom_types[slot]
            atom_array.atom_name[pos] = name
            atom_array.element[pos] = name[0]  # atom37 is C/N/O/S only for standard residues
            atom_array.res_name[pos] = residues[i]
            atom_array.res_id[pos] = int(residue_pdb_idx[i])
            atom_array.chain_id[pos] = "A"
            pos += 1
        res_id_out.append(int(residue_pdb_idx[i]))
    return atom_array


def compute_labels(graph) -> Dict[str, np.ndarray]:
    coords = graph.coords.numpy()
    coord_mask = graph.coord_mask.numpy()
    residues = graph.residues
    residue_pdb_idx = graph.residue_pdb_idx.numpy()
    n_res = coords.shape[0]

    # --- AA identity ---
    aa = graph.residue_type.numpy().astype(np.int64)
    aa_valid = np.ones(n_res, dtype=bool)

    # --- B-factor, z-scored within this protein ---
    bfactor = graph.bfactor.numpy().astype(np.float64)
    std = bfactor.std()
    bfactor_z = (bfactor - bfactor.mean()) / std if std > 1e-6 else np.zeros(n_res)
    bfactor_valid = np.ones(n_res, dtype=bool) if std > 1e-6 else np.zeros(n_res, dtype=bool)

    # --- SS + RSA via biotite ---
    import biotite.structure as struc

    ss = np.full(n_res, -1, dtype=np.int64)
    ss_valid = np.zeros(n_res, dtype=bool)
    rsa = np.zeros(n_res, dtype=np.float64)
    rsa_valid = np.zeros(n_res, dtype=bool)

    has_ca = coord_mask[:, CA_SLOT]
    if has_ca.sum() >= 2:  # annotate_sse needs at least a couple of CAs to say anything
        atom_array = _build_atom_array(coords, coord_mask, residues, residue_pdb_idx)
        idx_by_res_id = {int(residue_pdb_idx[i]): i for i in range(n_res)}

        sse = struc.annotate_sse(atom_array)
        res_ids, res_names = struc.get_residues(atom_array)
        for res_id, state in zip(res_ids, sse):
            if state not in SS_STATE_TO_INT:
                continue
            i = idx_by_res_id.get(int(res_id))
            if i is not None:
                ss[i] = SS_STATE_TO_INT[state]
                ss_valid[i] = True

        sasa_per_atom = struc.sasa(atom_array, vdw_radii="ProtOr")
        sasa_per_atom = np.nan_to_num(sasa_per_atom, nan=0.0)
        for res_id, res_name in zip(res_ids, res_names):
            i = idx_by_res_id.get(int(res_id))
            if i is None or res_name not in MAX_ASA_EMPIRICAL:
                continue
            atom_sel = atom_array.res_id == res_id
            residue_sasa = sasa_per_atom[atom_sel].sum()
            rsa[i] = np.clip(residue_sasa / MAX_ASA_EMPIRICAL[res_name], 0.0, 1.0)
            rsa_valid[i] = True

    return {
        "aa": aa, "aa_valid": aa_valid,
        "bfactor_z": bfactor_z, "bfactor_valid": bfactor_valid,
        "ss": ss, "ss_valid": ss_valid,
        "rsa": rsa, "rsa_valid": rsa_valid,
    }
