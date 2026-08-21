"""One-off check: does the LogisticRegression probe's max_iter=2000 cap
undertrain the probe? Refits at 25x the iterations and 100x tighter
tolerance and compares accuracy directly, rather than arguing from theory.
Uses cached latents/labels only, no encoder re-runs.
"""

import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from loguru import logger

from extract_latents import extract
from probe import PROPERTIES, _gather

CHECKS = ["jepa-mutation", "ca-cond"]  # near-baseline case + a clearly-learned case for contrast


def run():
    pool, labels, latents = extract(checkpoint_ids=CHECKS)
    spec = PROPERTIES["aa"]

    for checkpoint_id in CHECKS:
        X_train, y_train = _gather(pool.protein_ids, pool.probe_split, "probe_train", labels,
                                    spec["label_key"], spec["valid_key"], latents_by_protein=latents[checkpoint_id])
        X_test, y_test = _gather(pool.protein_ids, pool.probe_split, "probe_test", labels,
                                  spec["label_key"], spec["valid_key"], latents_by_protein=latents[checkpoint_id])

        t0 = time.time()
        model_default = LogisticRegression(max_iter=2000, random_state=42)
        model_default.fit(X_train, y_train)
        acc_default = accuracy_score(y_test, model_default.predict(X_test))
        n_iter_default = model_default.n_iter_[0]
        t1 = time.time()

        model_extended = LogisticRegression(max_iter=50000, tol=1e-6, random_state=42)
        model_extended.fit(X_train, y_train)
        acc_extended = accuracy_score(y_test, model_extended.predict(X_test))
        n_iter_extended = model_extended.n_iter_[0]
        t2 = time.time()

        line = (
            f"{checkpoint_id:16s} default(max_iter=2000): acc={acc_default:.6f} "
            f"used {n_iter_default} iters ({t1-t0:.1f}s)  |  "
            f"extended(max_iter=50000, tol=1e-6): acc={acc_extended:.6f} "
            f"used {n_iter_extended} iters ({t2-t1:.1f}s)  |  "
            f"delta={acc_extended - acc_default:+.6f}"
        )
        print(line)
        logger.info(line)


if __name__ == "__main__":
    run()
