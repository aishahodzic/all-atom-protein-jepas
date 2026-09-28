import unittest

import torch
from torch import nn

from laproteina_readout import encode_readout, readout_metadata, validate_readout_block


class AddBlock(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class FixtureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layers = nn.ModuleList(AddBlock(i) for i in range(1, 13))
        self.latent_decoder_mean_n_log_scale = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
        self.fail = False

    def forward(self, batch):
        x = batch["x"]
        for block in self.transformer_layers:
            x = block(x)
        if self.fail:
            raise RuntimeError("fixture failure")
        projected = self.latent_decoder_mean_n_log_scale(x)
        return {"mean": projected[..., :2]}


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.encoder = FixtureEncoder().eval()
        self.batch = {"x": torch.zeros(1, 3, 4)}

    def test_one_based_blocks_capture_correct_depths(self):
        for block in (1, 6, 12):
            output = encode_readout(self.encoder, self.batch, block)
            self.assertTrue(torch.equal(output, torch.full((1, 3, 4), block * (block + 1) / 2)))
            self.assertEqual(readout_metadata(self.encoder, block)["module_index"], block - 1)
            self.assertEqual(len(self.encoder.transformer_layers[block - 1]._forward_hooks), 0)

    def test_default_mean_is_unchanged(self):
        self.assertTrue(torch.equal(encode_readout(self.encoder, self.batch), self.encoder(self.batch)["mean"]))
        self.assertEqual(readout_metadata(self.encoder)["dimension"], 2)
        self.assertEqual(readout_metadata(self.encoder, 6)["dimension"], 4)

    def test_post_layernorm_is_exact_and_distinct(self):
        block_12 = encode_readout(self.encoder, self.batch, 12)
        expected = self.encoder.latent_decoder_mean_n_log_scale[0](block_12)
        actual = encode_readout(
            self.encoder, self.batch, post_layernorm=True
        )
        torch.testing.assert_close(actual, expected)
        metadata = readout_metadata(self.encoder, post_layernorm=True)
        self.assertEqual(metadata["kind"], "post_latent_layernorm")
        self.assertEqual(metadata["dimension"], 4)
        self.assertEqual(
            len(self.encoder.latent_decoder_mean_n_log_scale[0]._forward_hooks), 0
        )

    def test_post_layernorm_and_block_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            encode_readout(self.encoder, self.batch, 12, post_layernorm=True)

    def test_invalid_numbers_fail(self):
        for value in (0, -1, True, 1.5, "6"):
            with self.assertRaises(ValueError):
                validate_readout_block(value, 12)
        with self.assertRaises(ValueError):
            encode_readout(self.encoder, self.batch, 13)

    def test_hook_removed_after_encoder_failure(self):
        self.encoder.fail = True
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            encode_readout(self.encoder, self.batch, 6)
        self.assertEqual(len(self.encoder.transformer_layers[5]._forward_hooks), 0)

    def test_repeated_invocation_is_rejected(self):
        self.encoder.transformer_layers[6] = self.encoder.transformer_layers[5]
        with self.assertRaisesRegex(RuntimeError, "fired 2 times"):
            encode_readout(self.encoder, self.batch, 6)


if __name__ == "__main__":
    unittest.main()
