"""Compute matched Biotite reconstruction metrics for reconstructed PDBs.

The evaluator supports both C-alpha-anchored probes and fully latent decoders.
Metrics use the identical intersection of heavy atoms present in the native,
raw and postprocessed PDBs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from biotite.structure import lddt, rmsd, superimpose, tm_score
from biotite.structure.io import load_structure


DEFAULT_PROTEINS = (
    "1kfm_A",
    "1nkw_M",
    "3d3r_A",
    "5a13_G",
    "5fli_A",
    "5l0o_A",
    "5xr7_A",
    "6o14_A",
    "7cnw_A",
    "8cbv_A",
)
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _atom_key(array, index: int) -> tuple[str, int, str, str]:
    insertion_code = array.ins_code[index] if "ins_code" in array.get_annotation_categories() else ""
    return (
        str(array.chain_id[index]),
        int(array.res_id[index]),
        str(insertion_code),
        str(array.atom_name[index]),
    )


def _indices_by_key(array) -> dict[tuple[str, int, str, str], int]:
    result = {}
    for index in range(array.array_length()):
        if str(array.element[index]).upper() == "H":
            continue
        key = _atom_key(array, index)
        if key in result:
            raise ValueError(f"duplicate atom key {key}")
        result[key] = index
    return result


def _matched_arrays(native, raw, postprocessed):
    mappings = [_indices_by_key(array) for array in (native, raw, postprocessed)]
    common_keys = sorted(set(mappings[0]) & set(mappings[1]) & set(mappings[2]))
    arrays = []
    for array, mapping in zip((native, raw, postprocessed), mappings):
        arrays.append(array[np.asarray([mapping[key] for key in common_keys])])
    return common_keys, arrays


def _safe_lddt(reference, subject_coordinates) -> float | None:
    value = float(lddt(reference, subject_coordinates, aggregation="all"))
    return value if np.isfinite(value) else None


def _stage_metrics(reference, subject, ca_mask, backbone_mask) -> dict[str, float | int | None]:
    ca_reference = reference[ca_mask]
    ca_subject = subject[ca_mask]
    ca_indices = np.arange(ca_reference.array_length())
    ca_fitted, _ = superimpose(ca_reference, ca_subject)

    all_fitted, _ = superimpose(reference, subject)
    backbone_reference = reference[backbone_mask]
    backbone_subject = subject[backbone_mask]
    backbone_fitted, _ = superimpose(backbone_reference, backbone_subject)

    return {
        "matched_heavy_atoms": int(reference.array_length()),
        "matched_backbone_atoms": int(backbone_reference.array_length()),
        "matched_ca_atoms": int(ca_reference.array_length()),
        "tm_score_ca": float(
            tm_score(
                ca_reference,
                ca_fitted,
                ca_indices,
                ca_indices,
                reference_length=ca_reference.array_length(),
            )
        ),
        "lddt_all_common_heavy": _safe_lddt(reference, subject.coord),
        "lddt_backbone": _safe_lddt(backbone_reference, backbone_subject.coord),
        "lddt_ca": _safe_lddt(ca_reference, ca_subject.coord),
        # Direct RMSD is appropriate here because native C-alpha coordinates
        # establish the coordinate frame supplied to the decoder.
        "rmsd_all_common_heavy_ca_anchored_angstrom": float(rmsd(reference, subject)),
        "rmsd_backbone_ca_anchored_angstrom": float(
            rmsd(backbone_reference, backbone_subject)
        ),
        "rmsd_ca_angstrom": float(rmsd(ca_reference, ca_subject)),
        "rmsd_all_common_heavy_optimal_angstrom": float(rmsd(reference, all_fitted)),
        "rmsd_backbone_optimal_angstrom": float(
            rmsd(backbone_reference, backbone_fitted)
        ),
    }


def _mean(values) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _aggregate(records, stage: str, all_atom_only: bool) -> dict:
    selected = [
        record
        for record in records
        if not all_atom_only or record["has_all_atom_native_reference"]
    ]
    metrics = selected[0][stage].keys()
    aggregate = {
        "protein_count": len(selected),
        "protein_ids": [record["protein_id"] for record in selected],
    }
    for metric in metrics:
        if metric.startswith("matched_"):
            aggregate[f"total_{metric}"] = int(
                sum(record[stage][metric] for record in selected)
            )
        else:
            aggregate[f"mean_{metric}"] = _mean(
                [record[stage][metric] for record in selected]
            )

    # The pooled direct RMSD matches the existing atom-pooled reporting.
    squared_errors = []
    for record in selected:
        count = record[stage]["matched_heavy_atoms"]
        value = record[stage]["rmsd_all_common_heavy_ca_anchored_angstrom"]
        squared_errors.extend([value * value] * count)
    aggregate["pooled_rmsd_all_common_heavy_ca_anchored_angstrom"] = float(
        np.sqrt(np.mean(squared_errors))
    )
    return aggregate


def compute(
    root: Path,
    proteins: tuple[str, ...],
    raw_suffix: str = "reconstructed",
    postprocessed_suffix: str = "reconstructed_idealized",
    coordinate_mode: str = "ca-anchored",
    postprocessed_root: Path | None = None,
) -> dict:
    records = []
    for protein in proteins:
        protein_root = root / protein
        post_root = (postprocessed_root or root) / protein
        paths = {
            "native": protein_root / f"{protein}_native.pdb",
            "raw": protein_root / f"{protein}_{raw_suffix}.pdb",
            "postprocessed": post_root / f"{protein}_{postprocessed_suffix}.pdb",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        native, raw, postprocessed = [
            load_structure(path, model=1) for path in paths.values()
        ]
        common_keys, (native_common, raw_common, post_common) = _matched_arrays(
            native, raw, postprocessed
        )
        atom_names = np.asarray([key[3] for key in common_keys])
        ca_mask = atom_names == "CA"
        backbone_mask = np.isin(atom_names, list(BACKBONE_ATOMS))
        if np.count_nonzero(ca_mask) == 0:
            raise ValueError(f"{protein}: no matched C-alpha atoms")
        has_all_atom_native = bool(np.any(~ca_mask))
        records.append(
            {
                "protein_id": protein,
                "has_all_atom_native_reference": has_all_atom_native,
                "raw": _stage_metrics(
                    native_common, raw_common, ca_mask, backbone_mask
                ),
                "postprocessed": _stage_metrics(
                    native_common, post_common, ca_mask, backbone_mask
                ),
            }
        )

    return {
        "protocol": {
            "software": "Biotite",
            "tm_score": "Matched C-alpha atoms after optimal C-alpha superposition; normalized by matched residue count.",
            "lddt": "Standard Biotite lDDT on the identical native/raw/postprocessed heavy-atom intersection; 15 A inclusion radius and 0.5/1/2/4 A bins.",
            "rmsd_ca_anchored": "Direct RMSD in the native coordinate frame established by the supplied native C-alpha anchors.",
            "rmsd_optimal": "RMSD after Biotite optimal rigid-body superposition.",
            "coordinate_mode": coordinate_mode,
            "warning": (
                "Native C-alpha coordinates are supplied to this decoder. "
                "C-alpha TM-score/lDDT/RMSD therefore measure anchor "
                "preservation, not global-shape recovery from the latent "
                "representation."
                if coordinate_mode == "ca-anchored"
                else "The decoder receives no native C-alpha coordinates; "
                "C-alpha metrics measure recovered global structure."
            ),
        },
        "aggregate_all_10": {
            "raw": _aggregate(records, "raw", all_atom_only=False),
            "postprocessed": _aggregate(
                records, "postprocessed", all_atom_only=False
            ),
        },
        "aggregate_9_all_atom_native": {
            "raw": _aggregate(records, "raw", all_atom_only=True),
            "postprocessed": _aggregate(
                records, "postprocessed", all_atom_only=True
            ),
        },
        "proteins": records,
    }


def _write_csv(result: dict, path: Path) -> None:
    rows = []
    for protein in result["proteins"]:
        for stage in ("raw", "postprocessed"):
            rows.append(
                {
                    "protein_id": protein["protein_id"],
                    "stage": stage,
                    "has_all_atom_native_reference": protein[
                        "has_all_atom_native_reference"
                    ],
                    **protein[stage],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("non_code/pymol_proteins")
    )
    parser.add_argument("--proteins", nargs="+", default=DEFAULT_PROTEINS)
    parser.add_argument("--raw-suffix", default="reconstructed")
    parser.add_argument(
        "--postprocessed-suffix", default="reconstructed_idealized"
    )
    parser.add_argument("--postprocessed-root", type=Path)
    parser.add_argument(
        "--coordinate-mode",
        choices=("ca-anchored", "fully-latent"),
        default="ca-anchored",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "non_code/pymol_proteins/audits/biotite_reconstruction_metrics.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "non_code/pymol_proteins/audits/biotite_reconstruction_metrics.csv"
        ),
    )
    args = parser.parse_args()
    result = compute(
        args.root,
        tuple(args.proteins),
        raw_suffix=args.raw_suffix,
        postprocessed_suffix=args.postprocessed_suffix,
        coordinate_mode=args.coordinate_mode,
        postprocessed_root=args.postprocessed_root,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    _write_csv(result, args.csv_output)
    print(json.dumps(result["aggregate_9_all_atom_native"], indent=2))


if __name__ == "__main__":
    main()
