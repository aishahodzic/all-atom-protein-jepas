"""Decompose residue-LeJEPA latents into protein-mean and residual components.

This is a read-only analysis of the already extracted EMA latent caches.  It
uses the exact frozen 15,000-residue sample and cross-protein cosine-neighbour
protocol from ``analyze_canonical_residue_ab_10k.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


REPO = pathlib.Path(__file__).resolve().parents[2]
LINEAR = REPO / "Evaluation/linear_probing"
KNN = REPO / "Evaluation/knn_purity"
LATENT_VIZ = REPO / "Evaluation/latent_viz"
for path in (REPO, LINEAR, KNN, LATENT_VIZ):
    sys.path.insert(0, str(path))

from analyze_current_four_health import (  # noqa: E402
    aligned_property,
    external_neighbours,
    neighbour_spearman,
    purity,
    spectrum_metrics,
)
from knn import common_protein_ids, load_encoder_latents, load_labels  # noqa: E402
from viz import CombinedSelection, SEED, sample_indices  # noqa: E402


CHECKPOINTS = {
    "iterative": "canonical-lejepa-residue-iterative-ema-step10000",
    "visible_only": "canonical-lejepa-residue-visible-only-ema-step10000",
}
EXPECTED_PROTEINS = 5_224
EXPECTED_RESIDUE_POOL = 748_402
N_POINTS = 15_000
K = 10
EXPECTED_SAMPLE_SHA256 = (
    "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"
)
RESULTS = REPO / "Evaluation/diagnostics/results"
CSV_PATH = RESULTS / "canonical_lejepa_residue_ab_ema_10k_decomposition.csv"
JSON_PATH = RESULTS / "canonical_lejepa_residue_ab_ema_10k_decomposition.json"


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def physical_sample_sha256(
    selection: CombinedSelection, sampled: np.ndarray
) -> str:
    physical = np.concatenate(
        [
            np.asarray([f"{protein_id}:{int(index)}" for index in indices])
            for protein_id, indices in selection.keys
        ]
    )[sampled]
    return hashlib.sha256("\n".join(physical.tolist()).encode()).hexdigest()


def repeated_protein_means(
    selection: CombinedSelection, cache: dict[str, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    repeated = []
    unique = []
    for protein_id, indices in selection.keys:
        mean = cache[protein_id].float().mean(dim=0).numpy()
        unique.append(mean)
        repeated.append(np.repeat(mean[None, :], len(indices), axis=0))
    return np.concatenate(repeated, axis=0), np.stack(unique, axis=0)


def component_metrics(
    values: np.ndarray,
    protein_index: np.ndarray,
    amino_acids: np.ndarray,
    secondary_structure: np.ndarray,
    rsa: np.ndarray,
    rsa_valid: np.ndarray,
    log_length: np.ndarray,
    raw_total_variance: float,
) -> dict[str, float | int]:
    neighbours = external_neighbours(values, protein_index)
    all_valid = np.ones(len(values), dtype=bool)
    rsa_rho, rsa_n = neighbour_spearman(rsa, rsa_valid, neighbours)
    length_rho, length_n = neighbour_spearman(
        log_length, all_valid, neighbours
    )
    spectrum = spectrum_metrics(values)
    variance = float(np.var(values.astype(np.float64), axis=0).sum())
    return {
        "effective_rank": spectrum["effective_rank"],
        "embedding_std": float(torch.from_numpy(values).float().std().item()),
        "total_coordinate_variance": variance,
        "variance_fraction_of_raw": variance / raw_total_variance,
        "aa_purity_at_10_cross_protein": purity(amino_acids, neighbours),
        "ss_purity_at_10_cross_protein": purity(
            secondary_structure, neighbours
        ),
        "rsa_knn10_spearman": rsa_rho,
        "rsa_n": rsa_n,
        "log_chain_length_knn10_spearman": length_rho,
        "log_chain_length_n": length_n,
    }


def protein_balanced_mean_metrics(
    means: np.ndarray, lengths: np.ndarray
) -> dict[str, float | int | None]:
    unit = normalized(means)
    _, neighbours = cKDTree(unit).query(unit, k=K + 1, workers=-1)
    neighbours = neighbours[:, 1:]
    log_length = np.log(lengths)
    prediction = log_length[neighbours].mean(axis=1)
    spectrum = spectrum_metrics(means)
    variance = float(np.var(means.astype(np.float64), axis=0).sum())
    return {
        "effective_rank": spectrum["effective_rank"],
        "embedding_std": float(torch.from_numpy(means).float().std().item()),
        "total_coordinate_variance": variance,
        "variance_fraction_of_raw": None,
        "aa_purity_at_10_cross_protein": None,
        "ss_purity_at_10_cross_protein": None,
        "rsa_knn10_spearman": None,
        "rsa_n": 0,
        "log_chain_length_knn10_spearman": float(
            spearmanr(log_length, prediction).statistic
        ),
        "log_chain_length_n": len(means),
    }


def main() -> None:
    labels = load_labels()
    caches = {
        mode: load_encoder_latents(checkpoint_id)
        for mode, checkpoint_id in CHECKPOINTS.items()
    }
    protein_ids = common_protein_ids(caches)
    selection = CombinedSelection(protein_ids, labels)
    if len(protein_ids) != EXPECTED_PROTEINS or selection.n != EXPECTED_RESIDUE_POOL:
        raise RuntimeError(
            f"cohort changed: proteins={len(protein_ids)}, pool={selection.n}"
        )
    sampled = sample_indices(selection.n, N_POINTS, SEED)
    sample_sha256 = physical_sample_sha256(selection, sampled)
    if sample_sha256 != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(f"frozen physical sample changed: {sample_sha256}")

    amino_acids = selection.aa[sampled]
    secondary_structure = selection.ss[sampled]
    protein_index = selection.protein_idx[sampled]
    length_by_selection_protein = np.asarray(
        [len(labels[protein_id]["aa"]) for protein_id, _ in selection.keys],
        dtype=np.float64,
    )
    if not np.array_equal(
        np.unique(selection.protein_idx), np.arange(len(selection.keys))
    ):
        raise RuntimeError("selection protein indices are not contiguous")
    log_length = np.log(length_by_selection_protein[protein_index])
    rsa, rsa_valid = aligned_property(selection, labels, "rsa", "rsa_valid")
    rsa, rsa_valid = rsa[sampled], rsa_valid[sampled]

    rows: list[dict[str, object]] = []
    decomposition_checks: dict[str, dict[str, float]] = {}
    for mode, checkpoint_id in CHECKPOINTS.items():
        cache = caches[mode]
        raw_all = selection.build_X(cache).astype(np.float32)
        protein_mean_all, protein_mean_unique = repeated_protein_means(
            selection, cache
        )
        raw = raw_all[sampled]
        protein_mean = protein_mean_all[sampled]
        residual = raw - protein_mean
        raw_variance = float(np.var(raw.astype(np.float64), axis=0).sum())

        component_values = {
            "raw_z": raw,
            "protein_mean_broadcast": protein_mean,
            "within_protein_residual": residual,
        }
        for component, values in component_values.items():
            metrics = component_metrics(
                values,
                protein_index,
                amino_acids,
                secondary_structure,
                rsa,
                rsa_valid,
                log_length,
                raw_variance,
            )
            rows.append(
                {
                    "checkpoint": checkpoint_id,
                    "masked_kv_mode": mode,
                    "component": component,
                    "observations": len(values),
                    **metrics,
                }
            )

        rows.append(
            {
                "checkpoint": checkpoint_id,
                "masked_kv_mode": mode,
                "component": "protein_mean_one_per_protein",
                "observations": len(protein_mean_unique),
                **protein_balanced_mean_metrics(
                    protein_mean_unique, length_by_selection_protein
                ),
            }
        )

        mu_fraction = float(
            np.var(protein_mean.astype(np.float64), axis=0).sum()
            / raw_variance
        )
        residual_fraction = float(
            np.var(residual.astype(np.float64), axis=0).sum() / raw_variance
        )
        decomposition_checks[mode] = {
            "protein_mean_variance_fraction": mu_fraction,
            "residual_variance_fraction": residual_fraction,
            "cross_covariance_fraction": 1.0 - mu_fraction - residual_fraction,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "description": (
            "Protein-mean decomposition of the matched canonical residue-"
            "LeJEPA A/B EMA 10k latent caches"
        ),
        "checkpoint_ids": CHECKPOINTS,
        "sample": {
            "proteins": len(protein_ids),
            "residue_pool": selection.n,
            "sampled_residues": len(sampled),
            "seed": SEED,
            "physical_sample_sha256": sample_sha256,
        },
        "definitions": {
            "protein_mean": (
                "mean of all cached real-residue embeddings in each protein"
            ),
            "residual": "raw residue embedding minus its protein mean",
            "neighbours": (
                "cosine 10-NN in the component space, excluding residues from "
                "the query protein"
            ),
            "variance_fraction": (
                "sum of component coordinate variances divided by the sum for "
                "raw z, evaluated on the frozen residue sample"
            ),
        },
        "decomposition_checks": decomposition_checks,
        "csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {JSON_PATH}")
    for row in rows:
        print(row)
    print(decomposition_checks)


if __name__ == "__main__":
    main()
