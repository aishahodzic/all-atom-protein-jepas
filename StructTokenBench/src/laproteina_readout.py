"""Frozen post-block readouts; block numbers are one-based, never code indices."""

from numbers import Integral

import torch


def validate_readout_block(block, num_blocks=None):
    if block is None:
        return None
    if isinstance(block, bool) or not isinstance(block, Integral) or block < 1:
        raise ValueError("readout_block must be a positive one-based integer or None")
    if num_blocks is not None and block > num_blocks:
        raise ValueError(f"readout_block={block} exceeds {num_blocks} encoder blocks")
    return int(block)


def readout_metadata(encoder, block=None, post_layernorm=False):
    block = validate_readout_block(block, len(encoder.transformer_layers))
    if block is not None and post_layernorm:
        raise ValueError("block and post_layernorm readouts are mutually exclusive")
    projection = encoder.latent_decoder_mean_n_log_scale[1]
    if post_layernorm:
        return {
            "kind": "post_latent_layernorm",
            "block_number": None,
            "module_index": 0,
            "dimension": int(projection.in_features),
            "context": "full_unmasked_evaluation",
            "extra_projection": "final_layernorm_only",
        }
    return {
        "kind": "latent_mean" if block is None else "post_transformer_block",
        "block_number": block,
        "module_index": None if block is None else block - 1,
        "dimension": int(projection.out_features // 2 if block is None else projection.in_features),
        "context": "full_unmasked_evaluation",
        "extra_projection": None,
    }


@torch.no_grad()
def encode_readout(encoder, batch, block=None, post_layernorm=False):
    """Run the unchanged encoder and return its mean or one cloned block output.

    Cloning at the hook prevents later in-place operations from changing the
    captured state. The hook is removed even when the encoder raises an error.
    """
    metadata = readout_metadata(encoder, block, post_layernorm)
    if block is None and not post_layernorm:
        return encoder(batch)["mean"]
    captured = []

    def capture(_module, _args, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Expected a Tensor from the selected Transformer block")
        captured.append(output.detach().clone())

    module = (
        encoder.latent_decoder_mean_n_log_scale[0]
        if post_layernorm
        else encoder.transformer_layers[block - 1]
    )
    handle = module.register_forward_hook(capture)
    try:
        encoder(batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Selected readout fired {len(captured)} times; expected exactly once")
    result = captured[0]
    if result.ndim != 3 or result.shape[-1] != metadata["dimension"]:
        raise RuntimeError(f"Unexpected readout shape: {tuple(result.shape)}")
    return result
