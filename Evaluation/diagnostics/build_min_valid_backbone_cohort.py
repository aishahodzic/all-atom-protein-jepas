"""Build a canonical-PDB cohort that excludes coordinate-sparse chains.

The source cohort and MMseqs split artifacts are left untouched.  The output
directory reuses the source processed graphs and clustering artifacts, but its
cluster-membership TSV omits entries with fewer than ``min_valid`` backbone-N
coordinates.  This preserves the original cluster split while preventing an
invalid member from being sampled during residue-level SIGReg.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR


def _inspect_entry(task: tuple[str, str, int]) -> tuple[str, int, int, str | None]:
    processed_dir, sequence_id, min_valid = task
    path = Path(processed_dir) / f"{sequence_id}.pt"
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
        atom_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        valid_backbone_n = int(atom_mask[:, 0].sum())
        length = int(atom_mask.shape[0])
        return sequence_id, valid_backbone_n, length, None
    except Exception as error:  # Preserve failures in the auditable manifest.
        return sequence_id, -1, -1, f"{type(error).__name__}: {error}"


def _link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"unexpected existing symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-valid", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if args.min_valid < 1:
        raise ValueError("--min-valid must be positive")
    output.mkdir(parents=True, exist_ok=True)

    cohort_csvs = sorted(source.glob("df_pdb_f1_minl50_maxl256_*.csv"))
    if len(cohort_csvs) != 1:
        raise RuntimeError(f"expected one canonical cohort CSV, found {cohort_csvs}")
    cohort_csv = cohort_csvs[0]
    frame = pd.read_csv(cohort_csv)
    sequence_ids = frame["id"].astype(str).tolist()

    tasks = (
        (str(source / "processed"), sequence_id, args.min_valid)
        for sequence_id in sequence_ids
    )
    records: list[tuple[str, int, int, str | None]] = []
    context = mp.get_context("fork")
    with context.Pool(processes=args.workers) as pool:
        for index, record in enumerate(
            pool.imap_unordered(_inspect_entry, tasks, chunksize=64), start=1
        ):
            records.append(record)
            if index % 10_000 == 0 or index == len(sequence_ids):
                print(f"inspected {index}/{len(sequence_ids)}", flush=True)

    invalid = sorted(
        record for record in records if record[1] < args.min_valid
    )
    failures = [record for record in invalid if record[3] is not None]
    if failures:
        raise RuntimeError(f"failed to inspect {len(failures)} entries: {failures[:10]}")
    excluded_ids = {record[0] for record in invalid}

    manifest = pd.DataFrame(
        invalid,
        columns=["id", "valid_backbone_n", "sequence_length", "error"],
    )
    manifest.to_csv(
        output / f"excluded_lt{args.min_valid}_valid_backbone_n.tsv",
        sep="\t",
        index=False,
    )
    (output / "excluded_ids.txt").write_text(
        "".join(f"{sequence_id}\n" for sequence_id in sorted(excluded_ids)),
        encoding="utf-8",
    )

    shutil.copy2(cohort_csv, output / cohort_csv.name)
    _link(source / "processed", output / "processed")

    for path in sorted(source.glob("seq_df_pdb_f1_minl50_maxl256_*.fasta")):
        _link(path, output / path.name)
    for path in sorted(source.glob("cluster_seqid_0.5_df_pdb_f1_minl50_maxl256_*_test.fasta")):
        _link(path, output / path.name)

    cluster_tsvs = sorted(
        source.glob("cluster_seqid_0.5_df_pdb_f1_minl50_maxl256_*_test.tsv")
    )
    if len(cluster_tsvs) != 1:
        raise RuntimeError(f"expected one cluster TSV, found {cluster_tsvs}")
    kept_lines = 0
    removed_lines = 0
    with cluster_tsvs[0].open("r", encoding="utf-8") as source_handle, (
        output / cluster_tsvs[0].name
    ).open("w", encoding="utf-8") as output_handle:
        for line in source_handle:
            _, member_id = line.rstrip("\n").split("\t", maxsplit=1)
            if member_id in excluded_ids:
                removed_lines += 1
            else:
                output_handle.write(line)
                kept_lines += 1

    summary = (
        f"source_entries={len(sequence_ids)}\n"
        f"min_valid_backbone_n={args.min_valid}\n"
        f"excluded_entries={len(excluded_ids)}\n"
        f"excluded_fraction={len(excluded_ids) / len(sequence_ids):.8f}\n"
        f"cluster_tsv_kept_lines={kept_lines}\n"
        f"cluster_tsv_removed_lines={removed_lines}\n"
    )
    (output / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
