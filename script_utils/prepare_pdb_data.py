"""Prepare the shared full-PDB dataset metadata before distributed training."""

import os
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    load_dotenv(project_root / ".env")

    if not os.environ.get("DATA_PATH"):
        raise RuntimeError("DATA_PATH must point to the La-Proteina data directory")

    with initialize_config_dir(config_dir=str(project_root / "configs"), version_base=None):
        cfg = compose(config_name="training_ae_partlatent")

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.setup(stage="fit")
    print(
        "Prepared full-PDB splits:",
        f"train={len(datamodule.train_ds)}",
        f"val={len(datamodule.val_ds)}",
        f"test={len(datamodule.test_ds)}",
    )


if __name__ == "__main__":
    main()
