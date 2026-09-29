#!/usr/bin/env python3
"""Download archived thesis checkpoints and stage evaluation compatibility links."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "msc-thesis-org/ckpts"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO_ROOT / "StructTokenBench" / "100k-checkpoints"
EVALUATION_CONFIG = REPO_ROOT / "configs" / "evaluation" / "exp_config_jepa_nn130m.json"

# Stable archive path -> historical path consumed by main's evaluation registry.
MODELS = {
    "ca-cond": ("ae/ca-cond/100k.ckpt", "ae/ca-cond/100k-EMA.ckpt"),
    "fully-latent": ("ae/fully-latent/100k.ckpt", "ae/fully_latent/100k-EMA.ckpt"),
    "jepa-random-50": ("jepa/random-50/100k.ckpt", "jepa/random_masking_50pct/100k.ckpt"),
    "jepa-random-70": ("jepa/random-70/100k.ckpt", "jepa/random_masking_70pct/100k.ckpt"),
    "jepa-mutation": ("jepa/mutation/100k.ckpt", "jepa/mutation/100k.ckpt"),
    "jepa-sigreg-random-50-absolute": (
        "jepa_sigreg/absolute/random-50/100k.ckpt",
        "le-jepa/random_masking_50pct/100k.ckpt",
    ),
    "jepa-sigreg-random-70-absolute": (
        "jepa_sigreg/absolute/random-70/100k.ckpt",
        "le-jepa/random_masking_70pct/100k.ckpt",
    ),
    "jepa-sigreg-mutation-absolute": (
        "jepa_sigreg/absolute/mutation/100k.ckpt",
        "le-jepa/mutation/100k.ckpt",
    ),
    "jepa-sigreg-random-70-relative-rotation": (
        "jepa_sigreg/relative/random-70-rotation/100k.ckpt",
        "le-jepa/relative-position-and-rotation/100k.ckpt",
    ),
}

GROUPS = {"core": tuple(MODELS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(token: str | None) -> dict[str, dict[str, str]]:
    manifest_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename="manifest.csv",
            token=token,
        )
    )
    with manifest_path.open(newline="") as handle:
        return {row["hf_path"]: row for row in csv.DictReader(handle)}


def install_link(source: Path, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        if not force:
            raise FileExistsError(f"different symlink already exists: {destination}")
        destination.unlink()
    elif destination.exists():
        if not force:
            raise FileExistsError(
                f"file already exists: {destination}; use --force to replace it"
            )
        destination.unlink()
    destination.symlink_to(source.resolve())


def select_models(group: str | None, requested: list[str]) -> list[str]:
    selected = list(GROUPS[group]) if group else []
    selected.extend(requested)
    if not selected:
        selected.extend(GROUPS["core"])
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS))
    parser.add_argument("--model", action="append", default=[], choices=sorted(MODELS))
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = select_models(args.group, args.model)
    print(f"Hugging Face dataset: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Evaluation checkpoint root: {args.destination}")
    for model_id in selected:
        archive_path, staged_path = MODELS[model_id]
        print(f"  {model_id}: {archive_path} -> {staged_path}")
    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN")
    manifest = read_manifest(token)
    for model_id in selected:
        archive_path, staged_path = MODELS[model_id]
        if archive_path not in manifest:
            raise KeyError(f"{archive_path} is absent from the archive manifest")
        cached = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=archive_path,
                token=token,
                cache_dir=args.cache_dir,
            )
        )
        row = manifest[archive_path]
        if cached.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"size mismatch for {archive_path}")
        actual_sha = sha256(cached)
        if actual_sha != row["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {archive_path}: {actual_sha} != {row['sha256']}"
            )
        destination = args.destination / staged_path
        if (
            destination.exists()
            and not destination.is_symlink()
            and destination.stat().st_size == int(row["bytes"])
            and sha256(destination) == row["sha256"]
        ):
            print(f"already verified at compatibility path {model_id}: {destination}")
            continue
        install_link(cached, destination, force=args.force)
        print(f"verified and staged {model_id}: {destination}")

    config_destination = args.destination / "shared" / EVALUATION_CONFIG.name
    config_destination.parent.mkdir(parents=True, exist_ok=True)
    if config_destination.exists() or config_destination.is_symlink():
        same_config = (
            config_destination.resolve() == EVALUATION_CONFIG.resolve()
            or (
                config_destination.exists()
                and config_destination.read_bytes() == EVALUATION_CONFIG.read_bytes()
            )
        )
        if not same_config:
            if not args.force:
                raise FileExistsError(
                    f"different evaluation config exists: {config_destination}; use --force"
                )
            config_destination.unlink()
    if not config_destination.exists():
        config_destination.symlink_to(EVALUATION_CONFIG.resolve())
    print(f"staged evaluation config: {config_destination}")


if __name__ == "__main__":
    main()
