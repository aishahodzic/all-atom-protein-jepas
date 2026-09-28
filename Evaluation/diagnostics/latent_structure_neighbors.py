"""Test whether nearby protein embeddings correspond to similar 3D structures.

For each cached representation, the experiment finds the nearest protein-level
embedding for a fixed set of held-out query proteins.  Chains from the same PDB
entry are excluded.  Each query is also paired with a random protein of similar
length, giving a control for the strong effect of chain length.  Structural
similarity is measured by a Biotite structural alignment followed by TM-score
normalised by the longer chain.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
from typing import Dict, Iterable, Tuple

import biotite.structure as struc
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal, get_sequence_identity
import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LINEAR_PROBING_DIR = REPO_ROOT / "Evaluation" / "linear_probing"
sys.path[:0] = [str(REPO_ROOT), str(LINEAR_PROBING_DIR)]

from data_pool import build_protein_pool  # noqa: E402
from openfold.np import residue_constants  # noqa: E402


LATENT_DIR = LINEAR_PROBING_DIR / "latents"
OUTPUT_DIR = REPO_ROOT / "Evaluation" / "diagnostics" / "results" / "latent_structure_neighbors"
MODEL_FILES = {
    "JEPA 64D": LATENT_DIR / "jepa-r70-d64-step100000.pt",
    "JEPA 64D + SO(3)": LATENT_DIR / "jepa-r70-d64-rot-step100000.pt",
    "Uncropped LeJEPA control": LATENT_DIR / "canonical-lejepa-crop-control-uncropped-raw-step10k.pt",
    "Cropped LeJEPA control": LATENT_DIR / "canonical-lejepa-crop-control-fixed96-raw-step10k.pt",
}


def _pdb_id(protein_id: str) -> str:
    return protein_id.split("_", 1)[0].lower()


def _load_latents(path: pathlib.Path) -> Dict[str, torch.Tensor]:
    values = torch.load(path, map_location="cpu", weights_only=False)
    return {protein_id: tensor.float() for protein_id, tensor in values.items()}


def _pool_latents(latents: Dict[str, torch.Tensor], protein_ids: Iterable[str]) -> np.ndarray:
    return np.stack([latents[protein_id].mean(dim=0).numpy() for protein_id in protein_ids])


def _nearest_different_entry(
    query_index: int,
    vectors: np.ndarray,
    protein_ids: list[str],
) -> Tuple[int, float]:
    difference = vectors - vectors[query_index]
    distances = np.sqrt(np.sum(difference * difference, axis=1))
    order = np.argsort(distances)
    query_pdb = _pdb_id(protein_ids[query_index])
    for candidate_index in order:
        if candidate_index == query_index:
            continue
        if _pdb_id(protein_ids[candidate_index]) == query_pdb:
            continue
        return int(candidate_index), float(distances[candidate_index])
    raise RuntimeError(f"No eligible neighbour for {protein_ids[query_index]}")


def _length_matched_control(
    query_index: int,
    lengths: np.ndarray,
    protein_ids: list[str],
    rng: random.Random,
) -> int:
    query_length = int(lengths[query_index])
    tolerance = max(5, round(0.10 * query_length))
    query_pdb = _pdb_id(protein_ids[query_index])
    candidates = [
        index
        for index, length in enumerate(lengths)
        if index != query_index
        and _pdb_id(protein_ids[index]) != query_pdb
        and abs(int(length) - query_length) <= tolerance
    ]
    if not candidates:
        raise RuntimeError(f"No length-matched control for {protein_ids[query_index]}")
    return rng.choice(candidates)


def _graph_to_backbone(graph) -> struc.AtomArray:
    atom_names = ("N", "CA", "C")
    atom_indices = np.array([residue_constants.atom_order[name] for name in atom_names])
    coords = graph.coords.detach().cpu().numpy()
    coord_mask = graph.coord_mask.detach().cpu().numpy().astype(bool)
    residue_valid = coord_mask[:, atom_indices].all(axis=1)
    if residue_valid.sum() < 10:
        raise ValueError(f"Too few complete backbone residues for {graph.id}")

    residue_names = np.asarray(graph.residues)[residue_valid]
    selected = coords[residue_valid][:, atom_indices, :]
    residue_count = selected.shape[0]
    atoms = struc.AtomArray(residue_count * len(atom_names))
    atoms.coord = selected.reshape(-1, 3)
    atoms.chain_id = np.full(len(atoms), "A")
    atoms.res_id = np.repeat(np.arange(1, residue_count + 1), len(atom_names))
    atoms.res_name = np.repeat(residue_names, len(atom_names))
    atoms.atom_name = np.tile(np.asarray(atom_names), residue_count)
    atoms.element = np.tile(np.asarray(("N", "C", "C")), residue_count)
    atoms.hetero = np.zeros(len(atoms), dtype=bool)
    return atoms


def _tm_score(reference: struc.AtomArray, subject: struc.AtomArray) -> float:
    fitted, _, reference_indices, subject_indices = struc.superimpose_structural_homologs(
        reference,
        subject,
        max_iterations=5,
        reference_length="longer",
    )
    return float(
        struc.tm_score(
            reference,
            fitted,
            reference_indices,
            subject_indices,
            reference_length="longer",
        )
    )


_PROTEIN_MATRIX = SubstitutionMatrix.std_protein_matrix()


def _sequence_identity(reference: str, subject: str) -> float:
    alignment = align_optimal(
        ProteinSequence(reference),
        ProteinSequence(subject),
        _PROTEIN_MATRIX,
        gap_penalty=(-10, -1),
        terminal_penalty=False,
        max_number=1,
    )[0]
    return float(get_sequence_identity(alignment, mode="shortest"))


def _bootstrap_mean_ci(values: np.ndarray, seed: int, iterations: int = 20_000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 1_000):
        stop = min(start + 1_000, iterations)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        sampled_means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return float(low), float(high)


def run(n_queries: int, seed: int) -> None:
    pool = build_protein_pool(seed=42)
    all_latents = {name: _load_latents(path) for name, path in MODEL_FILES.items()}
    common_ids = [
        protein_id
        for protein_id in pool.protein_ids
        if all(protein_id in model_latents for model_latents in all_latents.values())
    ]
    index_in_pool = {protein_id: index for index, protein_id in enumerate(pool.protein_ids)}
    lengths = np.array([len(all_latents["Uncropped LeJEPA control"][protein_id]) for protein_id in common_ids])

    held_out_indices = [
        index for index, protein_id in enumerate(common_ids) if pool.probe_split[protein_id] == "probe_test"
    ]
    rng = random.Random(seed)
    rng.shuffle(held_out_indices)
    query_indices = held_out_indices[: min(n_queries, len(held_out_indices))]
    control_indices = {
        query_index: _length_matched_control(query_index, lengths, common_ids, rng)
        for query_index in query_indices
    }

    graph_cache = {}
    structure_cache: Dict[str, struc.AtomArray] = {}

    def graph(protein_id: str):
        if protein_id not in graph_cache:
            graph_cache[protein_id] = pool.dataset[index_in_pool[protein_id]]
        return graph_cache[protein_id]

    def structure(protein_id: str) -> struc.AtomArray:
        if protein_id not in structure_cache:
            structure_cache[protein_id] = _graph_to_backbone(graph(protein_id))
        return structure_cache[protein_id]

    def sequence(protein_id: str) -> str:
        return "".join(residue_constants.restype_3to1[name] for name in graph(protein_id).residues)

    rows = []
    for model_name, model_latents in all_latents.items():
        vectors = _pool_latents(model_latents, common_ids)
        for position, query_index in enumerate(query_indices, start=1):
            query_id = common_ids[query_index]
            neighbour_index, latent_distance = _nearest_different_entry(
                query_index, vectors, common_ids
            )
            control_index = control_indices[query_index]
            neighbour_id = common_ids[neighbour_index]
            control_id = common_ids[control_index]
            try:
                neighbour_tm = _tm_score(structure(query_id), structure(neighbour_id))
                control_tm = _tm_score(structure(query_id), structure(control_id))
                neighbour_sequence_identity = _sequence_identity(
                    sequence(query_id), sequence(neighbour_id)
                )
                control_sequence_identity = _sequence_identity(
                    sequence(query_id), sequence(control_id)
                )
            except Exception as error:  # one unusual structure should not stop the audit
                rows.append(
                    {
                        "model": model_name,
                        "query_id": query_id,
                        "neighbour_id": neighbour_id,
                        "control_id": control_id,
                        "error": repr(error),
                    }
                )
                continue
            rows.append(
                {
                    "model": model_name,
                    "query_id": query_id,
                    "neighbour_id": neighbour_id,
                    "control_id": control_id,
                    "query_length": int(lengths[query_index]),
                    "neighbour_length": int(lengths[neighbour_index]),
                    "control_length": int(lengths[control_index]),
                    "latent_distance": latent_distance,
                    "neighbour_tm": neighbour_tm,
                    "control_tm": control_tm,
                    "tm_difference": neighbour_tm - control_tm,
                    "neighbour_sequence_identity": neighbour_sequence_identity,
                    "control_sequence_identity": control_sequence_identity,
                    "error": "",
                }
            )
            if position % 25 == 0:
                print(f"{model_name}: {position}/{len(query_indices)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"nearest_structure_n{len(query_indices)}_seed{seed}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "design": {
            "n_requested_queries": n_queries,
            "n_selected_queries": len(query_indices),
            "seed": seed,
            "query_split": "probe_test",
            "same_pdb_entries_excluded": True,
            "control": "random different-PDB chain within max(5 residues, 10% query length)",
            "latent_distance": "Euclidean distance between mean-pooled residue embeddings",
            "structural_metric": "Biotite structural alignment; TM-score normalised by longer chain",
        },
        "models": {},
    }
    for model_position, model_name in enumerate(all_latents):
        valid = [row for row in rows if row["model"] == model_name and not row.get("error")]
        differences = np.asarray([row["tm_difference"] for row in valid], dtype=float)
        neighbour_scores = np.asarray([row["neighbour_tm"] for row in valid], dtype=float)
        control_scores = np.asarray([row["control_tm"] for row in valid], dtype=float)
        neighbour_identities = np.asarray(
            [row["neighbour_sequence_identity"] for row in valid], dtype=float
        )
        control_identities = np.asarray(
            [row["control_sequence_identity"] for row in valid], dtype=float
        )
        low, high = _bootstrap_mean_ci(differences, seed + model_position)
        remote = [row for row in valid if row["neighbour_sequence_identity"] < 0.30]
        remote_differences = np.asarray([row["tm_difference"] for row in remote], dtype=float)
        if remote:
            remote_low, remote_high = _bootstrap_mean_ci(
                remote_differences, seed + len(all_latents) + model_position
            )
        else:
            remote_low, remote_high = None, None
        summary["models"][model_name] = {
            "n_valid": len(valid),
            "n_failed": len(query_indices) - len(valid),
            "mean_neighbour_tm": float(neighbour_scores.mean()),
            "mean_control_tm": float(control_scores.mean()),
            "mean_paired_difference": float(differences.mean()),
            "paired_difference_95ci": [low, high],
            "fraction_neighbour_higher": float((differences > 0).mean()),
            "mean_neighbour_sequence_identity": float(neighbour_identities.mean()),
            "mean_control_sequence_identity": float(control_identities.mean()),
            "n_neighbours_below_30pct_identity": len(remote),
            "below_30pct_mean_neighbour_tm": (
                float(np.mean([row["neighbour_tm"] for row in remote])) if remote else None
            ),
            "below_30pct_mean_paired_difference": (
                float(remote_differences.mean()) if remote else None
            ),
            "below_30pct_paired_difference_95ci": [remote_low, remote_high],
            "below_30pct_fraction_neighbour_higher": (
                float((remote_differences > 0).mean()) if remote else None
            ),
        }

    json_path = OUTPUT_DIR / f"nearest_structure_n{len(query_indices)}_seed{seed}_summary.json"
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Rows: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run(arguments.n_queries, arguments.seed)
