"""Thesis AA t-SNE: existing C-alpha AE, JEPA-70, uncropped LeJEPA + SO(3).

Reuses frozen 100k full-context latent caches, never runs an encoder or trains
a model. The AE baseline is EMA; the two predictive encoders use raw weights.
Uses the existing matched residue-selection and t-SNE implementations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

from run_tsne import (  # noqa: E402
    LINEAR_PROBING_DIR,
    CombinedSelection,
    common_protein_ids,
    coordinate_path,
    load_encoder_latents,
    load_labels,
    load_or_fit,
    sample_indices,
    scatter_panel,
)
from thesis_plot_style import BLACK, apply_thesis_style  # noqa: E402

PANELS = (
    ("ca-cond", r"C$\alpha$-conditioned AE", "ema"),
    ("jepa-rand-mask-70", "JEPA-rand70", "raw"),
    (
        "canonical-lejepa-v1-rotations-raw-step100000",
        "Uncropped LeJEPA + SO(3)",
        "raw",
    ),
)
STEM = "tsne_aa_ca_cond_jepa70_uncropped_lejepa_so3_100k"
OUTPUTS = (
    REPO / "non_code" / "main_thesis_figures" / "show",
    REPO / "thesis-latex" / "figures" / "results",
)
MANIFEST = REPO / "non_code" / "main_thesis_figures" / "data" / f"{STEM}.json"
EXPECTED_SAMPLE = "6281211e2d1f3364f54c33fbde061601e4d35560a56f75bd6864b993b775d9b8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(4)
    labels = load_labels()
    latent_paths = {
        name: LINEAR_PROBING_DIR / "latents" / f"{name}.pt"
        for name, _, _ in PANELS
    }
    have_latents = all(path.exists() for path in latent_paths.values())
    caches = (
        {name: load_encoder_latents(name) for name, _, _ in PANELS}
        if have_latents else {}
    )
    protein_ids = common_protein_ids(caches) if have_latents else sorted(labels)
    if have_latents:
        assert all(set(cache) == set(protein_ids) for cache in caches.values())
    selection = CombinedSelection(protein_ids, labels)
    sampled = sample_indices(selection.n, n_points=15_000, seed=42)
    physical_ids = np.concatenate([
        np.asarray([f"{pid}:{int(i)}" for i in indices], dtype=object)
        for pid, indices in selection.keys
    ])[sampled]
    sample_sha = hashlib.sha256("\n".join(physical_ids.tolist()).encode()).hexdigest()
    assert len(protein_ids) == 5_224 and selection.n == 748_402
    assert sample_sha == EXPECTED_SAMPLE, sample_sha
    aa = selection.aa[sampled]
    print(f"Verified identical {len(sampled):,} residues across all three encoders.", flush=True)

    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    previous_encoders = {
        entry["checkpoint_id"]: entry for entry in previous["encoders"]
    } if previous else {}
    embeddings, provenance = {}, []
    for name, label, weight_source in PANELS:
        latent_path = latent_paths[name]
        if have_latents:
            X = selection.build_X(caches[name])[sampled]
            assert X.shape == (15_000, 8) and np.isfinite(X).all()
            with threadpool_limits(limits=4):
                embedding, cached_coordinate_path, input_sha = load_or_fit(
                    name, X, perplexity=30.0, seed=42, max_iter=1000, force=False,
                )
            latent_sha = sha256(latent_path)
        else:
            assert name in previous_encoders, f"Missing prior provenance for {name}"
            prior = previous_encoders[name]
            cached_coordinate_path = coordinate_path(name, 15_000, 30.0, 42)
            assert cached_coordinate_path.exists(), cached_coordinate_path
            cached = np.load(cached_coordinate_path, allow_pickle=False)
            embedding = cached["embedding"]
            input_sha = str(cached["input_sha256"].item())
            assert input_sha == prior["sampled_latents_sha256"]
            assert sha256(cached_coordinate_path) == prior["coordinate_cache_sha256"]
            latent_sha = prior["latent_cache_sha256"]
        assert embedding.shape == (15_000, 2) and np.isfinite(embedding).all()
        embeddings[name] = embedding
        provenance.append({
            "checkpoint_id": name,
            "label": label,
            "encoder_training_step": 100_000,
            "weight_source": weight_source,
            "latent_dim": 8,
            "latent_cache": str(latent_path.relative_to(REPO)),
            "latent_cache_sha256": latent_sha,
            "sampled_latents_sha256": input_sha,
            "coordinate_cache": str(cached_coordinate_path.relative_to(REPO)),
            "coordinate_cache_sha256": sha256(cached_coordinate_path),
        })

    lm_family = apply_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, (name, label, _) in zip(axes, PANELS):
        scatter_panel(ax, embeddings[name], aa, view="aa20", annotate_aa=name == "ca-cond")
        if name == "ca-cond":
            assert len(ax.texts) == 20
            for annotation in ax.texts:
                annotation.set_fontsize(7.5)
                annotation.set_color(BLACK)
                annotation.set_fontfamily(lm_family)
                annotation.set_fontweight(600)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.035)
    assert not fig.legends
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.025, wspace=0.07)
    # Figure-level panel labels stay on one baseline despite each projection's
    # different aspect ratio. There is deliberately no overall figure title.
    for ax, (_, label, _) in zip(axes, PANELS):
        box = ax.get_position(original=True)
        fig.text((box.x0 + box.x1) / 2, 0.955, label,
                 ha="center", va="top", fontsize=18)
    output_paths = []
    for output in OUTPUTS:
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{STEM}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        output_paths.append(str(path.relative_to(REPO)))
        print(f"Wrote {path}", flush=True)
    plt.close(fig)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "description": "AA-coloured t-SNE; model labels and AA labels on C-alpha AE clusters only; no overall title or legend",
        "readout": "full-context per-residue final latent mean",
        "selection": {
            "shared_proteins": len(protein_ids), "eligible_residues": selection.n,
            "sampled_residues": len(sampled), "physical_sample_sha256": sample_sha,
        },
        "tsne": {
            "metric": "cosine", "perplexity": 30.0, "seed": 42,
            "max_iter": 1000, "init": "random", "learning_rate": "auto",
            "early_exaggeration": 12.0, "method": "barnes_hut", "angle": 0.5,
        },
        "encoders": provenance,
        "outputs": output_paths,
        "note": "AE EMA baseline retained by author request; JEPA and LeJEPA use raw weights. Independently fitted t-SNE axes and inter-island distances are not comparable across models.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
