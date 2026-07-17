"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import transformers


logger = logging.getLogger(__name__)

_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TOKENIZER_LOAD_ATTEMPTS = 3
_TOKENIZER_RETRY_INITIAL_BACKOFF_SECONDS = 2.0
_HTTP_STATUS_PATTERN = re.compile(r"\b(408|429|500|502|503|504)\b")


def _transient_huggingface_status_code(exc: BaseException) -> int | None:
    """Find a retryable Hugging Face HTTP status in a wrapped exception chain."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                parsed_status_code = int(status_code)
            except (TypeError, ValueError):
                pass
            else:
                if parsed_status_code in _TRANSIENT_HTTP_STATUS_CODES:
                    return parsed_status_code
                return None

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

        current = current.__cause__ or current.__context__
    return None


def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``.
    """
    for attempt in range(1, _TOKENIZER_LOAD_ATTEMPTS + 1):
        try:
            return transformers.AutoTokenizer.from_pretrained(
                tokenizer_model,
                revision=tokenizer_revision or None,
                trust_remote_code=True,
            )
        except Exception as exc:
            status_code = _transient_huggingface_status_code(exc)
            if status_code is None:
                raise
            if attempt == _TOKENIZER_LOAD_ATTEMPTS:
                raise RuntimeError(
                    "Hugging Face Hub is unavailable while loading tokenizer "
                    f"{tokenizer_model!r} (HTTP {status_code}); exhausted "
                    f"{_TOKENIZER_LOAD_ATTEMPTS} attempts. Retry the training job "
                    "when Hugging Face is available or use a staged tokenizer artifact."
                ) from exc

            backoff_seconds = _TOKENIZER_RETRY_INITIAL_BACKOFF_SECONDS * 2 ** (
                attempt - 1
            )
            logger.warning(
                "Transient Hugging Face tokenizer load failure for %r (HTTP %d); "
                "retrying attempt %d/%d in %.1fs",
                tokenizer_model,
                status_code,
                attempt + 1,
                _TOKENIZER_LOAD_ATTEMPTS,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)

    raise AssertionError("unreachable")


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    return load_tokenizer(
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
    )
