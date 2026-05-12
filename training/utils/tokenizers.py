"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

from typing import Any

import transformers


def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``.
    """
    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_model,
        revision=tokenizer_revision or None,
        trust_remote_code=True,
    )


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
    )
