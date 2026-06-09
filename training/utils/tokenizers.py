"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

import logging
from typing import Any

import transformers

logger = logging.getLogger(__name__)


def _is_dsv4_tokenizer_path(tokenizer_model: str | None) -> bool:
    if not tokenizer_model:
        return False
    normalized = tokenizer_model.lower().replace("_", "-")
    return "deepseek-v4" in normalized or "deepseekv4" in normalized


def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``.
    """
    if not tokenizer_model:
        raise ValueError("tokenizer_model is required")

    try:
        return transformers.AutoTokenizer.from_pretrained(
            tokenizer_model,
            revision=tokenizer_revision or None,
            trust_remote_code=True,
        )
    except Exception as exc:
        if not _is_dsv4_tokenizer_path(tokenizer_model):
            raise
        logger.warning(
            "AutoTokenizer failed for %s (%s); falling back to tokenizer.json only",
            tokenizer_model,
            exc,
        )

    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    tokenizer_file = hf_hub_download(
        tokenizer_model,
        "tokenizer.json",
        revision=tokenizer_revision or None,
    )
    logger.info(
        "Loaded DSV4 tokenizer from tokenizer.json only (skipped unsupported config.json)"
    )
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
    )
