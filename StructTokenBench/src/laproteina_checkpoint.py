"""Small, dependency-free helpers for LaProteina JEPA checkpoints."""


def extract_jepa_encoder_state_dict(state_dict, encoder_source="online"):
    """Extract the requested encoder branch from a JEPA Lightning state dict.

    ``target`` intentionally accepts both the canonical ``target_encoder`` name
    and this repository's ``ema_encoder`` name.  The returned prefix records
    which one was actually found so callers can log the exact branch.
    """
    encoder_source = str(encoder_source).lower()
    if encoder_source == "online":
        candidate_prefixes = ("online_encoder.",)
    elif encoder_source == "target":
        candidate_prefixes = ("target_encoder.", "ema_encoder.")
    else:
        raise ValueError(
            f"Unknown JEPA encoder_source {encoder_source!r}; expected 'online' or 'target'"
        )

    for prefix in candidate_prefixes:
        weights = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if weights:
            return weights, prefix[:-1]

    available = sorted({
        key.split(".", 1)[0]
        for key in state_dict
        if "encoder" in key.split(".", 1)[0]
    })
    raise KeyError(
        f"Checkpoint has no {encoder_source!r} JEPA encoder branch "
        f"(tried {candidate_prefixes}); available encoder branches: {available or 'none'}"
    )
