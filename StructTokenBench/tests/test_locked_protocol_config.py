import importlib.machinery
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPOSITORY = Path(__file__).resolve().parents[2]
SRC = REPOSITORY / "StructTokenBench" / "src"
SCRIPT = SRC / "script"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT))
sys.path.insert(0, str(REPOSITORY))

deepspeed_stub = types.ModuleType("deepspeed")
deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
deepspeed_stub.__version__ = "0.0.0"
deepspeed_stub.utils = types.SimpleNamespace(
    safe_get_full_grad=lambda parameter: parameter.grad
)
sys.modules.setdefault("deepspeed", deepspeed_stub)

from run_supervised_task import validate_locked_protocol_config  # noqa: E402


def compose_locked_config(*extra_overrides):
    overrides = [
        "tokenizer=WrappedLaProteinaTokenizer",
        "model.d_model=8",
        "trainer.devices=[0]",
        "optimization.optimizer.lr=0.001",
        "data.target_field=binding_label",
        "data.num_workers=8",
        "data.prefetch_factor=2",
        "data.filter_length=600",
        "default_data_dir=/tmp/data",
        "data.pdb_data_dir=/tmp/pdb",
        "trainer.default_root_dir=/tmp/logs",
        "tokenizer_pretrained_ckpt_path=/tmp/fake.ckpt",
        "tokenizer_checkpoint_type=jepa",
        "tokenizer_encoder_cfg_path=/tmp/fake.json",
        "tokenizer_encoder_source=online",
        "++tokenizer_eval_protocol=upstream-l600-v1",
        "tokenizer_cache_tag=upstream-l600-v1",
        *extra_overrides,
    ]
    config_dir = str(SCRIPT / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="interpro", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def compose_locked_smoke(*extra_overrides):
    return compose_locked_config(
        "max_steps=2",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=2",
        "trainer.check_val_every_n_epoch=1",
        "lightning.callbacks.checkpoint.every_n_epochs=1",
        *extra_overrides,
    )


class LockedProtocolConfigTest(unittest.TestCase):
    def test_launcher_full_configuration_uses_upstream_step_ceiling(self):
        cfg = compose_locked_config()
        with mock.patch.dict(os.environ, {"STB_SMOKE": "0"}):
            policy = validate_locked_protocol_config(cfg)

        self.assertEqual(policy.name, "upstream-l600-v1")
        self.assertEqual(cfg.max_steps, 10000)
        self.assertEqual(cfg.trainer.check_val_every_n_epoch, 2)
        self.assertEqual(cfg.lightning.callbacks.early_stopping.patience, 10)
        self.assertEqual(cfg.lightning.callbacks.checkpoint.every_n_epochs, 1)

    def test_launcher_smoke_configuration_satisfies_lock(self):
        cfg = compose_locked_smoke()
        with mock.patch.dict(os.environ, {"STB_SMOKE": "1"}):
            policy = validate_locked_protocol_config(cfg)

        self.assertEqual(policy.name, "upstream-l600-v1")
        self.assertEqual(cfg.trainer.check_val_every_n_epoch, 1)

    def test_scientific_override_cannot_keep_locked_protocol_name(self):
        cfg = compose_locked_smoke("optimization.seed=999")
        with mock.patch.dict(os.environ, {"STB_SMOKE": "1"}):
            with self.assertRaisesRegex(ValueError, "optimization.seed"):
                validate_locked_protocol_config(cfg)


if __name__ == "__main__":
    unittest.main()
