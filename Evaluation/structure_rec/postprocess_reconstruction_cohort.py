"""Run deterministic reconstruction postprocessing in parallel by protein."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> None:
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )


def _process(protein_id: str, root: Path, modes: tuple[str, ...]) -> str:
    protein_root = root / protein_id
    for mode in modes:
        stem = f"{protein_id}_reconstructed_{mode}_sequence"
        source = protein_root / f"{stem}.pdb"
        projected = protein_root / f"{stem}_projected.pdb"
        rotamer = protein_root / f"{stem}_rotamer.pdb"
        final = protein_root / f"{stem}_postprocessed.pdb"
        if not source.is_file():
            raise FileNotFoundError(source)
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "Evaluation/structure_rec/project_pdb_ideal_geometry.py"),
                "--input",
                str(source),
                "--output",
                str(projected),
            ]
        )
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "Evaluation/structure_rec/rotamer_clash_repair.py"),
                "--input",
                str(projected),
                "--output",
                str(rotamer),
                "--metadata",
                str(protein_root / f"{protein_id}_{mode}_rotamer_repair.json"),
            ]
        )
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "Evaluation/structure_rec/restrained_geometry_repair.py"),
                "--input",
                str(rotamer),
                "--output",
                str(final),
                "--metadata",
                str(protein_root / f"{protein_id}_{mode}_geometry_repair.json"),
                "--strength",
                "500",
                "--steps",
                "2000",
                "--backbone-steps",
                "500",
                "--clash-strength",
                "300",
                "--peptide-tolerance",
                "0.03",
                "--peptide-angle-tolerance",
                "0.01",
            ]
        )
    return protein_id


def main(args: argparse.Namespace) -> None:
    manifest = json.loads(args.cohort.read_text())
    protein_ids = manifest[args.protein_ids_key]
    modes = tuple(args.sequence_modes)
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process, protein_id, args.root, modes): protein_id
            for protein_id in protein_ids
        }
        for future in concurrent.futures.as_completed(futures):
            protein_id = futures[future]
            future.result()
            completed.append(protein_id)
            print(f"completed {len(completed)}/{len(protein_ids)}: {protein_id}", flush=True)

    summary = {
        "root": str(args.root.resolve()),
        "cohort": str(args.cohort.resolve()),
        "protein_ids_key": args.protein_ids_key,
        "sequence_modes": list(modes),
        "workers": args.workers,
        "completed_ids": sorted(completed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--protein-ids-key", default="scored_ids")
    parser.add_argument(
        "--sequence-modes",
        nargs="+",
        choices=("predicted", "true"),
        default=("predicted", "true"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
