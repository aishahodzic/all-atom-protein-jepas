"""Check the pinned 100k EMA artifact and post-block adapter on one cached chain.

This is an integration fixture, not a task evaluation or a held-out score.
"""

import argparse
import json
from pathlib import Path

import torch

from evaluation_manifest import sha256_file
from laproteina_readout import encode_readout
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list
from tokenizer import WrappedLaProteinaTokenizer


REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "store/canonical_lejepa_v1_100k/evaluation/canonical-lejepa-v1-step100000-ema-staged.ckpt"
CONFIG = REPO / "StructTokenBench/100k-checkpoints/shared/exp_config_jepa_nn130m.json"
CHECKPOINT_SHA = "49aaae3dccc3835005bb6470b0f483a57cee4e28bead2408994693974389e273"
CONFIG_SHA = "98e0350d418e7d840fa245ddfd7150064e87a9ac74c72c9150a946f0f66ccd9d"
SOURCE_SHA = "4f09dff610adaa75c8ac51b4cf01acb001c236c72dd91ac31568ffa191484527"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, choices=(1, 6, 12), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assert sha256_file(CHECKPOINT) == CHECKPOINT_SHA
    assert sha256_file(CONFIG) == CONFIG_SHA
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    provenance = saved["canonical_lejepa_evaluation"]
    assert provenance["encoder_source"] == "ema"
    assert provenance["source_global_step"] == 100_000
    assert provenance["source_checkpoint_sha256"] == SOURCE_SHA
    assert provenance["latent_dim"] == 8 and provenance["projector"] is None
    del saved

    tokenizer = WrappedLaProteinaTokenizer(
        device=args.device, pretrained_ckpt_path=str(CHECKPOINT),
        checkpoint_type="jepa", encoder_cfg_path=str(CONFIG),
        evaluation_protocol="upstream-l600-v1", readout_block=args.block,
    )
    assert tokenizer.latent_z_dim == 768
    assert not any(p.requires_grad for p in tokenizer.encoder.parameters())
    assert tokenizer._in_pretrained_cache("1kn0", "G"), "Read-only integration fixture missing"
    # A cache hit uses the real PDBDataset atom reordering and evaluation transforms.
    graph = tokenizer._parse_graph("1kn0.cif", "G")
    batch = dense_padded_from_data_list([graph])
    batch["mask"] = batch["mask_dict"]["coords"][..., 0, 0]
    if tokenizer.needs_chain_break_feature:
        batch["chain_breaks_per_residue"] = graph.chain_breaks_per_residue.unsqueeze(0)
    batch = batch.to(args.device)
    encoder = tokenizer.encoder.eval()
    captured = []

    def capture(_module, _args, output):
        captured.append(output.detach().clone())

    handle = encoder.transformer_layers[args.block - 1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            baseline_mean = encoder(batch)["mean"]
    finally:
        handle.remove()
    assert len(captured) == 1
    result, indices, sequence = tokenizer._encode_graph(graph)
    torch.testing.assert_close(result, captured[0].squeeze(0), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(encode_readout(encoder, batch), baseline_mean, rtol=1e-5, atol=1e-6)
    assert result.shape == (len(sequence), 768)
    assert len(indices) == len(sequence)
    assert torch.isfinite(result).all()
    assert not encoder.transformer_layers[args.block - 1]._forward_hooks
    print(json.dumps({
        "status": "passed", "fixture": "1kn0_G", "residues": len(sequence),
        "readout": tokenizer.readout, "encoder_provenance": provenance,
        "device": args.device,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
