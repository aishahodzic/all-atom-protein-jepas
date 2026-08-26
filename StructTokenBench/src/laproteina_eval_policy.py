"""Small, dependency-free helpers for LaProteina evaluation policies.

The default/``None`` protocol deliberately preserves the historical local
integration, including checkpoint-training distribution filters.  The named
publication protocol is additive: it locks the choices needed for a comparable
StructTokenBench run without changing any other tokenizer's behaviour.
"""

from dataclasses import dataclass
from numbers import Integral
import re


UPSTREAM_L600_PROTOCOL = "upstream-l600-v1"
SUPPORTED_EVALUATION_PROTOCOLS = frozenset({UPSTREAM_L600_PROTOCOL})


@dataclass(frozen=True)
class LaProteinaEvaluationPolicy:
    """Resolved LaProteina-specific choices for one STB invocation."""

    name: str | None
    encoder_source: str
    evaluation_max_length: int | None
    cache_tag: str | None
    stb_filter_length: int | None
    apply_checkpoint_training_filters: bool
    strict_cohort: bool
    write_immutable_manifest: bool


def _normalize_optional_string(value, field_name):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def resolve_laproteina_evaluation_policy(
    protocol=None,
    *,
    encoder_source="online",
    evaluation_max_length=None,
    cache_tag=None,
):
    """Resolve either the legacy integration or the locked publication mode.

    ``upstream-l600-v1`` means:

    * the online/student encoder;
    * StructTokenBench's selected-residue ``filter_length=600``;
    * no LaProteina checkpoint-training cohort filters at all;
    * a fixed cache namespace and immutable run manifest.

    An old ``evaluation_max_length=600`` request is accepted for migration but
    intentionally resolved to ``None``: applying 600 to the *whole chain*
    would incorrectly remove STB examples whose selected domain is <=600.
    """

    protocol = _normalize_optional_string(protocol, "evaluation protocol")
    source = str(encoder_source).lower()
    if source not in {"online", "target"}:
        raise ValueError("encoder_source must be 'online' or 'target'")

    if protocol is None:
        return LaProteinaEvaluationPolicy(
            name=None,
            encoder_source=source,
            evaluation_max_length=evaluation_max_length,
            cache_tag=cache_tag,
            stb_filter_length=None,
            apply_checkpoint_training_filters=True,
            strict_cohort=False,
            write_immutable_manifest=False,
        )

    if protocol not in SUPPORTED_EVALUATION_PROTOCOLS:
        raise ValueError(
            f"Unknown LaProteina evaluation protocol {protocol!r}; expected one of "
            f"{sorted(SUPPORTED_EVALUATION_PROTOCOLS)}"
        )
    if source != "online":
        raise ValueError(
            f"{UPSTREAM_L600_PROTOCOL} locks encoder_source='online', got {source!r}"
        )
    if evaluation_max_length not in (None, 600):
        raise ValueError(
            f"{UPSTREAM_L600_PROTOCOL} uses STB filter_length=600 and does not "
            f"allow whole-chain evaluation_max_length={evaluation_max_length!r}"
        )
    if cache_tag not in (None, "", UPSTREAM_L600_PROTOCOL):
        raise ValueError(
            f"{UPSTREAM_L600_PROTOCOL} locks cache_tag={UPSTREAM_L600_PROTOCOL!r}, "
            f"got {cache_tag!r}"
        )

    return LaProteinaEvaluationPolicy(
        name=UPSTREAM_L600_PROTOCOL,
        encoder_source="online",
        # Critical: the STB limit applies after residue/domain selection.  A
        # tokenizer whole-chain max would be an extra LaProteina-only filter.
        evaluation_max_length=None,
        cache_tag=UPSTREAM_L600_PROTOCOL,
        stb_filter_length=600,
        apply_checkpoint_training_filters=False,
        strict_cohort=True,
        write_immutable_manifest=True,
    )


def resolve_evaluation_max_length(checkpoint_max_length, evaluation_max_length):
    """Return the selector max length used during downstream evaluation.

    ``None`` preserves the checkpoint's training-time selector exactly.  An
    explicit value replaces only that selector's ``max_length`` criterion;
    callers continue to pass through all its other criteria unchanged.
    """
    if evaluation_max_length is None:
        return checkpoint_max_length
    if isinstance(evaluation_max_length, bool) or not isinstance(
        evaluation_max_length, Integral
    ):
        raise TypeError("evaluation_max_length must be a positive integer or None")
    if evaluation_max_length <= 0:
        raise ValueError("evaluation_max_length must be positive")
    return int(evaluation_max_length)


def evaluation_length_cache_suffix(evaluation_max_length, effective_max_length):
    """Fingerprint an explicit evaluation-length override in token caches.

    The legacy/default policy deliberately retains its historical cache name.
    Explicit long-protein runs get a distinct suffix, preventing a token cache
    produced with the checkpoint's usual <=256 selection from being reused.
    """
    if evaluation_max_length is None:
        return ""
    return f"_evalmax{effective_max_length}"


def evaluation_cache_tag_suffix(cache_tag):
    """Return a safe explicit cache namespace for preprocessing revisions."""
    if cache_tag is None or cache_tag == "":
        return ""
    if not isinstance(cache_tag, str):
        raise TypeError("evaluation cache tag must be a string or None")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_tag):
        raise ValueError(
            "evaluation cache tag may contain only letters, numbers, '.', '_' and '-'"
        )
    return f"_cache-{cache_tag}"
