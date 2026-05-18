"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

from typing import Any

import transformers


# Default tokenizer revisions for models whose latest Hub tokenizer is known
# broken under transformers >=5.x. The bug: transformers v5 has
# `model_type == "kimi_k25"` on its `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`
# blocklist and unconditionally routes those checkpoints through
# `TokenizersBackend` (auto-converting `tiktoken.model` via a regex with the
# wrong special-token pattern). The fast-convert collapses the gaps in
# `added_tokens_decoder` and shifts every special-token ID by 1-3 vs the
# canonical mapping (e.g. trains write `163604` thinking it's `</think>`;
# inference decodes the same ID as `<|media_end|>`). The pinned revisions
# below ship a precompiled `tokenizer.json` + a `TikTokenTokenizerFast`
# wrapper so AutoTokenizer loads them directly without triggering the
# conversion. See FIR2-1631.
_DEFAULT_TOKENIZER_REVISIONS: dict[str, str] = {
    "moonshotai/kimi-k2.6": "81bcaaa79473ace391bcb4b6e6e08a87263767c8",
}


def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``. For models on the
    ``_DEFAULT_TOKENIZER_REVISIONS`` list, an unset revision falls back to
    the pinned-known-good commit instead of ``main``.
    """
    effective_revision = tokenizer_revision or None
    if effective_revision is None and tokenizer_model:
        effective_revision = _DEFAULT_TOKENIZER_REVISIONS.get(tokenizer_model.lower())

    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_model,
        revision=effective_revision,
        trust_remote_code=True,
    )


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
    )
