"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

import re
from typing import Any

import transformers


_HTTP_STATUS_PATTERN = re.compile(r"\b([45]\d\d)\b")


def _huggingface_http_status_code(exc: BaseException) -> int | None:
    """Find a Hugging Face HTTP status in a wrapped exception graph."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                parsed_status_code = int(status_code)
            except (TypeError, ValueError):
                pass
            else:
                if 400 <= parsed_status_code <= 599:
                    return parsed_status_code

        message = str(current)
        normalized_message = message.lower()
        if (
            "huggingface" in normalized_message
            or "huggingface.co" in normalized_message
            or "hf hub" in normalized_message
        ):
            match = _HTTP_STATUS_PATTERN.search(message)
            if match is not None:
                return int(match.group(1))

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``.
    """
    try:
        return transformers.AutoTokenizer.from_pretrained(
            tokenizer_model,
            revision=tokenizer_revision or None,
            trust_remote_code=True,
        )
    except Exception as exc:
        status_code = _huggingface_http_status_code(exc)
        if status_code is None:
            raise
        raise RuntimeError(
            "Hugging Face Hub request failed while loading tokenizer "
            f"{tokenizer_model!r} (HTTP {status_code})."
        ) from exc


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
    )
