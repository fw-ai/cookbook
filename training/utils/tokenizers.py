"""Shared HuggingFace tokenizer loading helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

from tokenizers import Tokenizer
import transformers

_HTTP_STATUS_PATTERN = re.compile(r"\b([45]\d\d)\b")
_GEMMA4_UNUSED_TOKEN_PATTERN = re.compile(r"^<unused\d+>$")
# Cross-repo contract consumed by fireworks.text.tokenizer at inference startup.
GEMMA4_RESERVED_TOKEN_MAP_CONFIG_KEY = "fireworks_gemma4_reserved_token_map"


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


_TokenizerLoader = Callable[
    [str | None, str | None, bool | None, Mapping[str, str] | None],
    Any,
]


def _propagate_huggingface_http_status(loader: _TokenizerLoader) -> _TokenizerLoader:
    """Preserve a wrapped Hugging Face HTTP status at the tokenizer boundary."""

    @wraps(loader)
    def wrapped(
        tokenizer_model: str | None,
        tokenizer_revision: str | None = None,
        trust_remote_code: bool | None = None,
        gemma4_reserved_token_map: Mapping[str, str] | None = None,
    ) -> Any:
        try:
            return loader(
                tokenizer_model,
                tokenizer_revision,
                trust_remote_code,
                gemma4_reserved_token_map,
            )
        except Exception as exc:
            status_code = _huggingface_http_status_code(exc)
            if status_code is None:
                raise
            raise RuntimeError(
                "Hugging Face Hub request failed while loading tokenizer "
                f"{tokenizer_model!r} (HTTP {status_code})."
            ) from exc

    return wrapped


def configure_gemma4_reserved_tokens(
    tokenizer: Any,
    token_map: Mapping[str, str],
) -> dict[str, int]:
    """Map customer token strings onto existing Gemma 4 ``<unusedN>`` rows.

    The operation rewrites token *names*, not token IDs: every alias takes the
    ID of its selected reserved slot and the tokenizer length must stay fixed.
    The aliases are then registered as special ``AddedToken`` instances so
    ordinary text encoding recognizes each complete string atomically.

    ``token_map`` maps the desired wire string to a Gemma 4 reserved token, for
    example ``{"<start_search>": "<unused123>"}``. Mapping a reserved token to
    itself (``{"<unused123>": "<unused123>"}``) only activates the stock name.
    """
    if not isinstance(token_map, Mapping) or not token_map:
        raise ValueError("gemma4_reserved_token_map must be a non-empty mapping")

    normalized: dict[str, str] = {}
    for alias, reserved_token in token_map.items():
        if not isinstance(alias, str) or not alias:
            raise ValueError("Gemma 4 reserved-token aliases must be non-empty strings")
        if "\x00" in alias:
            raise ValueError("Gemma 4 reserved-token aliases cannot contain null bytes")
        if not isinstance(
            reserved_token, str
        ) or not _GEMMA4_UNUSED_TOKEN_PATTERN.fullmatch(reserved_token):
            raise ValueError(
                "Gemma 4 reserved-token targets must use the exact '<unusedN>' spelling; "
                f"got {reserved_token!r} for alias {alias!r}"
            )
        if alias in normalized and normalized[alias] != reserved_token:
            raise ValueError(f"Gemma 4 token alias {alias!r} is mapped more than once")
        normalized[alias] = reserved_token

    targets = list(normalized.values())
    if len(set(targets)) != len(targets):
        raise ValueError(
            "Each Gemma 4 <unusedN> slot may be assigned to only one alias"
        )

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(tokenizer, "_tokenizer"):
        raise ValueError(
            "Gemma 4 reserved-token mapping requires a fast Hugging Face tokenizer"
        )

    backend_config = json.loads(backend.to_str())
    model_config = backend_config.get("model")
    if not isinstance(model_config, dict) or model_config.get("type") != "BPE":
        raise ValueError(
            "Gemma 4 reserved-token mapping requires the Gemma 4 BPE tokenizer"
        )
    vocab = model_config.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("Gemma 4 tokenizer JSON is missing model.vocab")

    original_size = len(tokenizer)
    resolved_ids: dict[str, int] = {}
    needs_backend_rebuild = False
    for alias, reserved_token in normalized.items():
        reserved_id = vocab.get(reserved_token)
        alias_id = vocab.get(alias)

        if alias == reserved_token:
            if reserved_id is None:
                raise ValueError(
                    f"{reserved_token!r} does not exist in this tokenizer vocabulary"
                )
            resolved_ids[alias] = int(reserved_id)
            continue

        if reserved_id is None:
            # A tokenizer saved after this mapping has already replaced the
            # reserved spelling with the alias. Treat an existing atomic alias
            # as idempotently configured.
            if alias_id is None:
                raise ValueError(
                    f"{reserved_token!r} does not exist in this tokenizer vocabulary"
                )
            resolved_ids[alias] = int(alias_id)
            continue
        if alias_id is not None:
            raise ValueError(
                f"Cannot map {alias!r}: that string already has vocabulary ID {alias_id}"
            )

        del vocab[reserved_token]
        vocab[alias] = reserved_id
        resolved_ids[alias] = int(reserved_id)
        needs_backend_rebuild = True

    if len(set(resolved_ids.values())) != len(resolved_ids):
        raise ValueError(
            "Each Gemma 4 reserved token mapping must resolve to a unique ID"
        )

    if needs_backend_rebuild:
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(backend_config))

    added_tokens = [
        transformers.AddedToken(
            alias,
            normalized=False,
            special=True,
        )
        for alias in normalized
    ]
    tokenizer.add_tokens(added_tokens, special_tokens=True)

    if len(tokenizer) != original_size:
        raise ValueError(
            "Gemma 4 reserved-token mapping unexpectedly changed tokenizer size "
            f"from {original_size} to {len(tokenizer)}; refusing to continue because "
            "the model embedding matrix is fixed"
        )

    for alias, expected_id in resolved_ids.items():
        actual_ids = tokenizer.encode(alias, add_special_tokens=False)
        if actual_ids != [expected_id]:
            raise ValueError(
                f"Gemma 4 token alias {alias!r} did not become atomic: "
                f"expected [{expected_id}], got {actual_ids}"
            )

    # Unknown tokenizer_config keys are retained in init_kwargs by HF and are
    # written back by save_pretrained(). Fireworks inference consumes this same
    # key, so a customized base carries the mapping across promotion.
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs[GEMMA4_RESERVED_TOKEN_MAP_CONFIG_KEY] = normalized
    return resolved_ids


@_propagate_huggingface_http_status
def load_tokenizer(
    tokenizer_model: str | None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool | None = None,
    gemma4_reserved_token_map: Mapping[str, str] | None = None,
) -> Any:
    """Load a tokenizer with cookbook defaults.

    ``tokenizer_revision`` is optional; empty strings are treated as unset so
    existing configs keep resolving HuggingFace ``main``. ``None`` preserves
    the legacy remote-code policy (enabled), while a reviewed tokenizer plan
    can explicitly enable or disable it.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model,
        revision=tokenizer_revision or None,
        trust_remote_code=True if trust_remote_code is None else trust_remote_code,
    )
    configured_map = gemma4_reserved_token_map
    if configured_map is None:
        configured_map = getattr(tokenizer, "init_kwargs", {}).get(
            GEMMA4_RESERVED_TOKEN_MAP_CONFIG_KEY
        )
    if configured_map:
        configure_gemma4_reserved_tokens(tokenizer, configured_map)
    return tokenizer


def load_deployment_tokenizer(deployment: Any) -> Any:
    """Load the tokenizer configured on a deployment config-like object."""
    args = [
        getattr(deployment, "tokenizer_model", None),
        getattr(deployment, "tokenizer_revision", None),
        getattr(deployment, "tokenizer_trust_remote_code", None),
    ]
    token_map = getattr(deployment, "gemma4_reserved_token_map", None)
    if token_map:
        return load_tokenizer(*args, gemma4_reserved_token_map=token_map)
    return load_tokenizer(*args)
