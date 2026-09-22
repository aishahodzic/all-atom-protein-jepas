"""B3/B4: one nn.Linear (== softmax/linear regression trained to convergence
-- using sklearn's convex solvers instead of a hand-rolled SGD loop so a
lazy optimizer can never masquerade as "near chance" and trip the B5 sanity
gate falsely) per (encoder x property). Reports probe score, majority/mean
baseline (skipped for bfactor -- see below), and a random-latent control.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

PROPERTIES = {
    "aa": dict(label_key="aa", valid_key="aa_valid", kind="classification"),
    "ss": dict(label_key="ss", valid_key="ss_valid", kind="classification"),
    "rsa": dict(label_key="rsa", valid_key="rsa_valid", kind="regression_r2"),
    "bfactor": dict(label_key="bfactor_z", valid_key="bfactor_valid", kind="regression_spearman"),
}


@dataclass
class ProbeResult:
    probe_score: float
    baseline: Optional[float]  # None for bfactor (Spearman has no meaningful constant-prediction baseline)
    random_control: float
    n_train: int
    n_test: int


def _gather(protein_ids, probe_split, split_name, labels_by_protein, label_key, valid_key,
            latents_by_protein: Optional[Dict[str, torch.Tensor]] = None,
            random_dim: Optional[int] = None, rng: Optional[np.random.Generator] = None):
    """Flattens residue-level (X, y) for one probe_split ("probe_train"/"probe_test"),
    respecting per-protein validity masks. Pass either latents_by_protein (real latents)
    or random_dim+rng (random-latent control) -- exactly one of the two."""
    xs, ys = [], []
    for pid in protein_ids:
        if probe_split[pid] != split_name:
            continue
        lab = labels_by_protein.get(pid)
        if lab is None:
            continue
        valid = lab[valid_key]
        if valid.sum() == 0:
            continue
        if latents_by_protein is not None:
            z = latents_by_protein.get(pid)
            if z is None:
                continue
            x = z[torch.from_numpy(valid)].numpy()
        else:
            x = rng.standard_normal((int(valid.sum()), random_dim)).astype(np.float32)
        y = lab[label_key][valid]
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def _fit_eval(X_train, y_train, X_test, y_test, kind: str, seed: int) -> tuple:
    """Returns (probe_score, baseline_or_None)."""
    if kind == "classification":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        counts = np.bincount(y_test.astype(np.int64))
        baseline = counts.max() / counts.sum()
    elif kind == "regression_r2":
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        baseline = r2_score(y_test, np.full_like(y_test, y_train.mean()))
    elif kind == "regression_spearman":
        model = LinearRegression()
        model.fit(X_train, y_train)
        score, _ = spearmanr(y_test, model.predict(X_test))
        baseline = None
    else:
        raise ValueError(kind)
    return score, baseline


def run_probe(
    property_name: str,
    protein_ids,
    probe_split,
    labels_by_protein,
    latents_by_protein: Dict[str, torch.Tensor],
    random_control_score: Optional[float] = None,
    seed: int = SEED,
) -> ProbeResult:
    spec = PROPERTIES[property_name]
    X_train, y_train = _gather(protein_ids, probe_split, "probe_train", labels_by_protein,
                                spec["label_key"], spec["valid_key"], latents_by_protein=latents_by_protein)
    X_test, y_test = _gather(protein_ids, probe_split, "probe_test", labels_by_protein,
                              spec["label_key"], spec["valid_key"], latents_by_protein=latents_by_protein)

    score, baseline = _fit_eval(X_train, y_train, X_test, y_test, spec["kind"], seed)

    if random_control_score is None:
        random_control_score = compute_random_control(property_name, protein_ids, probe_split,
                                                        labels_by_protein, X_train.shape[-1], seed)

    return ProbeResult(
        probe_score=float(score),
        baseline=None if baseline is None else float(baseline),
        random_control=float(random_control_score),
        n_train=len(y_train),
        n_test=len(y_test),
    )


def compute_random_control(property_name: str, protein_ids, probe_split, labels_by_protein,
                            latent_dim: int, seed: int = SEED) -> float:
    """Trains + evaluates the same probe on standard-normal noise of the same
    shape as the real latents. Encoder-independent by construction (noise
    doesn't come from any encoder) -- call once per property, reuse the
    result across all 8 encoders' rows."""
    spec = PROPERTIES[property_name]
    rng = np.random.default_rng(seed)
    X_train, y_train = _gather(protein_ids, probe_split, "probe_train", labels_by_protein,
                                spec["label_key"], spec["valid_key"], random_dim=latent_dim, rng=rng)
    X_test, y_test = _gather(protein_ids, probe_split, "probe_test", labels_by_protein,
                              spec["label_key"], spec["valid_key"], random_dim=latent_dim, rng=rng)
    score, _ = _fit_eval(X_train, y_train, X_test, y_test, spec["kind"], seed)
    return float(score)
