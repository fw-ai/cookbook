"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import wraps
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


_TokenizerLoader = Callable[[str | None, str | None, bool | None], Any]


def _propagate_huggingface_http_status(loader: _TokenizerLoader) -> _TokenizerLoader:
    """Preserve a wrapped Hugging Face HTTP status at the tokenizer boundary."""

    @wraps(loader)
    def wrapped(
        tokenizer_model: str | None,
        tokenizer_revision: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> Any:
        try:
            return loader(tokenizer_model, tokenizer_revision, trust_remote_code)
        except Exception as exc:
            status_code = _huggingface_http_status_code(exc)
            if status_code is None:
                raise
            raise RuntimeError(
                "Hugging Face Hub request failed while loading tokenizer "
                f"{tokenizer_model!r} (HTTP {status_code})."
            ) from exc

    return wrapped


@_propagate_huggingface_http_status
def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``. ``None`` preserves
    the legacy remote-code policy (enabled), while a reviewed tokenizer plan
    can explicitly enable or disable it.
    """
    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_model,
        revision=tokenizer_revision or None,
        trust_remote_code=True if trust_remote_code is None else trust_remote_code,
    )


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
        getattr(deployment, "tokenizer_trust_remote_code", None),
    )
