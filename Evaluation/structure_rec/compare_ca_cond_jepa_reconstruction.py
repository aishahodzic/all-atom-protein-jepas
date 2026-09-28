"""Compare C-alpha-conditioned baseline and JEPA reconstruction fairly.

The report separates three questions that otherwise become conflated when the
decoder predicts both residue identity and Atom37 coordinates:

1. ``predicted_sequence_common_heavy`` scores the exact same heavy-atom
   intersection for both models.
2. ``all_residue_backbone`` scores N, CA, C and O for every residue, avoiding
   side-chain atom-name ambiguity when the predicted amino acid is wrong.
3. ``true_sequence_heavy`` supplies the native amino-acid identity only when
   selecting/writing atoms, isolating coordinate recovery from sequence
   classification.

Both models receive native C-alpha coordinates, so C-alpha TM-score measures
anchor preservation rather than independent fold reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from biotite.structure import lddt, rmsd, superimpose, tm_score
from biotite.structure.io import load_structure


PROTEINS = (
    "1kfm_A",
    "3d3r_A",
    "5a13_G",
    "5fli_A",
    "5l0o_A",
    "5xr7_A",
    "6o14_A",
    "7cnw_A",
    "8cbv_A",
)
BACKBONE = {"N", "CA", "C", "O"}


def _key(array, index: int) -> tuple[str, int, str, str]:
    insertion = (
        str(array.ins_code[index])
        if "ins_code" in array.get_annotation_categories()
        else ""
    )
    return (
        str(array.chain_id[index]),
        int(array.res_id[index]),
        insertion,
        str(array.atom_name[index]),
    )


def _mapping(array) -> dict[tuple[str, int, str, str], int]:
    result = {}
    for index in range(array.array_length()):
        if str(array.element[index]).upper() == "H":
            continue
        atom_key = _key(array, index)
        if atom_key in result:
            raise ValueError(f"duplicate atom key {atom_key}")
        result[atom_key] = index
    return result


def _match(arrays, key_filter=None):
    mappings = [_mapping(array) for array in arrays]
    keys = set(mappings[0])
    for mapping in mappings[1:]:
        keys &= set(mapping)
    if key_filter is not None:
        keys = {key for key in keys if key_filter(key)}
    ordered = sorted(keys)
    if not ordered:
        raise ValueError("no matched atoms")
    matched = [
        array[np.asarray([mapping[key] for key in ordered])]
        for array, mapping in zip(arrays, mappings)
    ]
    return ordered, matched


def _metrics(reference, subject, keys) -> dict[str, float | int | None]:
    ca_mask = np.asarray([key[3] == "CA" for key in keys])
    value = float(lddt(reference, subject.coord, aggregation="all"))
    result = {
        "matched_heavy_atoms": int(reference.array_length()),
        "rmsd_angstrom": float(rmsd(reference, subject)),
        "lddt": value if np.isfinite(value) else None,
        "tm_score_ca": None,
    }
    if np.any(ca_mask):
        ca_reference = reference[ca_mask]
        ca_subject = subject[ca_mask]
        ca_fitted, _ = superimpose(ca_reference, ca_subject)
        indices = np.arange(ca_reference.array_length())
        result["tm_score_ca"] = float(
            tm_score(
                ca_reference,
                ca_fitted,
                indices,
                indices,
                reference_length=ca_reference.array_length(),
            )
        )
    return result


def _mean(values):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate(records, comparison, model):
    selected = [record["comparisons"][comparison][model] for record in records]
    return {
        "protein_count": len(selected),
        "total_matched_heavy_atoms": int(
            sum(item["matched_heavy_atoms"] for item in selected)
        ),
        "mean_rmsd_angstrom": _mean([item["rmsd_angstrom"] for item in selected]),
        "mean_lddt": _mean([item["lddt"] for item in selected]),
        "mean_tm_score_ca": _mean([item["tm_score_ca"] for item in selected]),
    }


def _summary_rates(summary_path: Path, proteins, label: str):
    summary = json.loads(summary_path.read_text())
    selected = [
        item for item in summary["proteins"] if item["protein_id"] in proteins
    ]
    if {item["protein_id"] for item in selected} != set(proteins):
        raise ValueError(f"{label} sequence summary does not contain the frozen cohort")
    rates = [float(item["sequence_recovery"]) for item in selected]
    total = sum(int(item["residues"]) for item in selected)
    correct = sum(
        round(float(item["sequence_recovery"]) * int(item["residues"]))
        for item in selected
    )
    return rates, correct, total


def _sequence_summary(
    ca_root: Path,
    ca_summary_path: Path | None,
    jepa_summary_path: Path,
    proteins,
):
    if ca_summary_path is not None:
        baseline_rates, baseline_correct, baseline_total = _summary_rates(
            ca_summary_path, proteins, "CA-conditioned"
        )
    else:
        baseline_rates = []
        baseline_correct = 0
        baseline_total = 0
        for protein in proteins:
            metadata = json.loads(
                (ca_root / protein / f"{protein}_metrics.json").read_text()
            )
            rate = float(metadata["sequence_recovery"])
            residues = int(metadata["num_residues"])
            baseline_rates.append(rate)
            baseline_correct += round(rate * residues)
            baseline_total += residues
    jepa_summary = json.loads(jepa_summary_path.read_text())
    selected = [
        item for item in jepa_summary["proteins"] if item["protein_id"] in proteins
    ]
    if {item["protein_id"] for item in selected} != set(proteins):
        raise ValueError("JEPA sequence summary does not contain the frozen protein set")
    jepa_rates = [float(item["sequence_recovery"]) for item in selected]
    jepa_total = sum(int(item["residues"]) for item in selected)
    jepa_correct = sum(
        round(float(item["sequence_recovery"]) * int(item["residues"]))
        for item in selected
    )
    return {
        "ca_conditioned_baseline": {
            "macro_mean": _mean(baseline_rates),
            "micro_mean": baseline_correct / baseline_total,
            "correct_residues": baseline_correct,
            "total_residues": baseline_total,
        },
        "jepa": {
            "macro_mean": _mean(jepa_rates),
            "micro_mean": jepa_correct / jepa_total,
            "correct_residues": jepa_correct,
            "total_residues": jepa_total,
        },
    }


def compare(args):
    proteins = PROTEINS
    if args.protein_ids_file is not None:
        manifest = json.loads(args.protein_ids_file.read_text())
        if isinstance(manifest, list):
            proteins = tuple(manifest)
        else:
            proteins = tuple(manifest[args.protein_ids_key])
    records = []
    for protein in proteins:
        ca_dir = args.ca_root / protein
        jepa_dir = args.jepa_root / protein
        native = load_structure(ca_dir / f"{protein}_native.pdb", model=1)
        ca_predicted = load_structure(
            ca_dir / f"{protein}_reconstructed_predicted_sequence.pdb", model=1
        )
        ca_true = load_structure(
            ca_dir / f"{protein}_reconstructed_true_sequence.pdb", model=1
        )
        jepa_predicted = load_structure(
            jepa_dir / f"{protein}_reconstructed_predicted_sequence.pdb", model=1
        )
        jepa_true = load_structure(
            jepa_dir / f"{protein}_reconstructed_true_sequence.pdb", model=1
        )
        predicted_arrays_to_match = [native, ca_predicted, jepa_predicted]
        true_arrays_to_match = [native, ca_true, jepa_true]
        if args.jepa_postprocessed_root is not None:
            post_dir = args.jepa_postprocessed_root / protein
            predicted_arrays_to_match.append(
                load_structure(
                    post_dir
                    / f"{protein}_reconstructed_predicted_sequence_postprocessed.pdb",
                    model=1,
                )
            )
            true_arrays_to_match.append(
                load_structure(
                    post_dir
                    / f"{protein}_reconstructed_true_sequence_postprocessed.pdb",
                    model=1,
                )
            )

        comparisons = {}
        predicted_keys, predicted_arrays = _match(predicted_arrays_to_match)
        comparisons["predicted_sequence_common_heavy"] = {
            "ca_conditioned_baseline": _metrics(
                predicted_arrays[0], predicted_arrays[1], predicted_keys
            ),
            "jepa": _metrics(
                predicted_arrays[0], predicted_arrays[2], predicted_keys
            ),
        }
        if args.jepa_postprocessed_root is not None:
            comparisons["predicted_sequence_common_heavy"]["jepa_postprocessed"] = (
                _metrics(predicted_arrays[0], predicted_arrays[3], predicted_keys)
            )

        backbone_keys, backbone_arrays = _match(
            predicted_arrays_to_match,
            key_filter=lambda key: key[3] in BACKBONE,
        )
        expected_residues = len({key[:3] for key in backbone_keys})
        if len(backbone_keys) != 4 * expected_residues:
            raise ValueError(
                f"{protein}: all-residue backbone is incomplete "
                f"({len(backbone_keys)} atoms for {expected_residues} residues)"
            )
        comparisons["all_residue_backbone"] = {
            "ca_conditioned_baseline": _metrics(
                backbone_arrays[0], backbone_arrays[1], backbone_keys
            ),
            "jepa": _metrics(backbone_arrays[0], backbone_arrays[2], backbone_keys),
        }
        if args.jepa_postprocessed_root is not None:
            comparisons["all_residue_backbone"]["jepa_postprocessed"] = _metrics(
                backbone_arrays[0], backbone_arrays[3], backbone_keys
            )

        true_keys, true_arrays = _match(true_arrays_to_match)
        comparisons["true_sequence_heavy"] = {
            "ca_conditioned_baseline": _metrics(
                true_arrays[0], true_arrays[1], true_keys
            ),
            "jepa": _metrics(true_arrays[0], true_arrays[2], true_keys),
        }
        if args.jepa_postprocessed_root is not None:
            comparisons["true_sequence_heavy"]["jepa_postprocessed"] = _metrics(
                true_arrays[0], true_arrays[3], true_keys
            )

        true_non_ca_keys, true_non_ca_arrays = _match(
            true_arrays_to_match,
            key_filter=lambda key: key[3] != "CA",
        )
        comparisons["true_sequence_non_ca"] = {
            "ca_conditioned_baseline": _metrics(
                true_non_ca_arrays[0], true_non_ca_arrays[1], true_non_ca_keys
            ),
            "jepa": _metrics(
                true_non_ca_arrays[0], true_non_ca_arrays[2], true_non_ca_keys
            ),
        }
        if args.jepa_postprocessed_root is not None:
            comparisons["true_sequence_non_ca"]["jepa_postprocessed"] = _metrics(
                true_non_ca_arrays[0], true_non_ca_arrays[3], true_non_ca_keys
            )
        records.append({"protein_id": protein, "comparisons": comparisons})

    comparison_names = tuple(records[0]["comparisons"])
    models = ["ca_conditioned_baseline", "jepa"]
    if args.jepa_postprocessed_root is not None:
        models.append("jepa_postprocessed")
    sequence_recovery = _sequence_summary(
        args.ca_root,
        args.ca_sequence_summary,
        args.jepa_sequence_summary,
        proteins,
    )
    if args.jepa_postprocessed_root is not None:
        sequence_recovery["jepa_postprocessed"] = dict(sequence_recovery["jepa"])
    result = {
        "protocol": {
            "predicted_sequence_common_heavy": (
                "Exact native/baseline/JEPA heavy-atom intersection after each "
                "decoder chooses its amino-acid sequence."
            ),
            "all_residue_backbone": (
                "N, CA, C and O for every residue; this remains defined when "
                "the predicted amino-acid identity is wrong."
            ),
            "true_sequence_heavy": (
                "All native heavy atoms with true amino-acid identities supplied "
                "only for atom selection; coordinates remain decoder predictions."
            ),
            "true_sequence_non_ca": (
                "All native heavy atoms except C-alpha, with true amino-acid "
                "identities supplied only for atom selection; coordinates remain "
                "decoder predictions."
            ),
            "warning": (
                "Both decoders receive native C-alpha coordinates, so C-alpha "
                "TM-score measures anchor preservation, not fold generation."
            ),
        },
        "sequence_recovery": sequence_recovery,
        "aggregates": {
            comparison: {
                model: _aggregate(records, comparison, model) for model in models
            }
            for comparison in comparison_names
        },
        "proteins": records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "comparison",
                "model",
                "protein_count",
                "matched_heavy_atoms",
                "mean_rmsd_angstrom",
                "mean_lddt",
                "mean_tm_score_ca",
                "sequence_recovery_macro",
                "sequence_recovery_micro",
            ),
        )
        writer.writeheader()
        for comparison in comparison_names:
            for model in models:
                aggregate = result["aggregates"][comparison][model]
                sequence = result["sequence_recovery"][model]
                writer.writerow(
                    {
                        "comparison": comparison,
                        "model": model,
                        "protein_count": aggregate["protein_count"],
                        "matched_heavy_atoms": aggregate[
                            "total_matched_heavy_atoms"
                        ],
                        "mean_rmsd_angstrom": aggregate["mean_rmsd_angstrom"],
                        "mean_lddt": aggregate["mean_lddt"],
                        "mean_tm_score_ca": aggregate["mean_tm_score_ca"],
                        "sequence_recovery_macro": sequence["macro_mean"],
                        "sequence_recovery_micro": sequence["micro_mean"],
                    }
                )
    print(json.dumps(result["sequence_recovery"], indent=2))
    print(json.dumps(result["aggregates"], indent=2))


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    audits = repo / "non_code" / "pymol_proteins" / "audits"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ca-root",
        type=Path,
        default=(
            repo
            / "Evaluation"
            / "structure_rec"
            / "exports"
            / "ca-cond-original-same10"
        ),
    )
    parser.add_argument("--jepa-root", type=Path, required=True)
    parser.add_argument(
        "--ca-sequence-summary",
        type=Path,
        help=(
            "Optional batched CA-conditioned reconstruction summary JSON. "
            "If omitted, read the historical per-protein metadata files."
        ),
    )
    parser.add_argument(
        "--protein-ids-file",
        type=Path,
        help="JSON list or cohort manifest containing the scored protein IDs.",
    )
    parser.add_argument(
        "--protein-ids-key",
        default="scored_ids",
        help="Manifest key used when --protein-ids-file contains an object.",
    )
    parser.add_argument(
        "--jepa-postprocessed-root",
        type=Path,
        help=(
            "Optional root containing separately postprocessed predicted- and "
            "true-sequence JEPA PDB files."
        ),
    )
    parser.add_argument(
        "--jepa-sequence-summary",
        type=Path,
        default=audits / "oracle_aa_sidechains.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=audits / "ca_cond_vs_jepa_three_way_metrics.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=audits / "ca_cond_vs_jepa_three_way_metrics.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
