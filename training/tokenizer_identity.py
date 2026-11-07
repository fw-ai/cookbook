"""Cookbook reviewed-model tokenizer identity (thin public wrapper).

Delegates to ``fireworks.training.sdk.tokenizer_identity`` when the installed
SDK carries a matching ``TOKENIZER_IDENTITY_POLICY_CAPABILITY``; otherwise
falls back to the generated snapshot in ``training/_tokenizer_identity_snapshot.py``.
Only ``ModuleNotFoundError`` naming the requested module or one of its parents
is treated as absence; any other import failure propagates unchanged.
"""

from __future__ import annotations

import importlib
from typing import Any

_POLICY_CAPABILITY = "deepseek-v4-v41+kimi-k3-k31-tokenizer-identity/saved-native-profiles-v3"
_SDK_IDENTITY_MODULE = "fireworks.training.sdk.tokenizer_identity"

# Which backend supplied the identity implementation: "sdk" (delegation) or
# "snapshot" (older SDK floor fallback). Exposed for diagnostics and tests.
TOKENIZER_IDENTITY_BACKEND: str


def _load_sdk_identity() -> Any:
    """Return the SDK tokenizer-identity module, or None when it is absent."""
    try:
        return importlib.import_module(_SDK_IDENTITY_MODULE)
    except ModuleNotFoundError as exc:
        _requested = _SDK_IDENTITY_MODULE.split(".")
        _missing = (exc.name or "").split(".")
        if not _missing or _missing != _requested[: len(_missing)]:
            raise
        return None


def _select_backend() -> tuple[str, Any]:
    """Pick the identity implementation for this interpreter.

    "sdk" requires both the module import and the capability marker, so an
    SDK that ships the module without the final policy falls back to the
    snapshot instead of inheriting the older behavior.
    """
    sdk_identity = _load_sdk_identity()
    if sdk_identity is not None and (
        getattr(sdk_identity, "TOKENIZER_IDENTITY_POLICY_CAPABILITY", None) == _POLICY_CAPABILITY
    ):
        return "sdk", sdk_identity
    # lazy: compatibility source is needed only for older SDKs.
    from training import _tokenizer_identity_snapshot as snapshot

    return "snapshot", snapshot


TOKENIZER_IDENTITY_BACKEND, _identity = _select_backend()

DEEPSEEK_V4_MODEL_TYPE = _identity.DEEPSEEK_V4_MODEL_TYPE
DEEPSEEK_V4_REPOSITORIES = _identity.DEEPSEEK_V4_REPOSITORIES
DEEPSEEK_V4_TOKENIZER_JSON_SHA256 = _identity.DEEPSEEK_V4_TOKENIZER_JSON_SHA256
DEEPSEEK_V41_MODEL_RE = _identity.DEEPSEEK_V41_MODEL_RE
DEEPSEEK_V41_TOKENIZER_JSON_SHA256 = _identity.DEEPSEEK_V41_TOKENIZER_JSON_SHA256


def is_code_free_model_name(model_name: str | None) -> bool:
    """Recognize restrictive V4/V4.1 context without selecting a renderer."""
    return _identity.is_code_free_model_name(model_name)

KIMI_K3_MODEL_TYPE = _identity.KIMI_K3_MODEL_TYPE
KIMI_K31_MODEL_TYPE = _identity.KIMI_K31_MODEL_TYPE
KIMI_K3_K31_MODEL_TYPES = _identity.KIMI_K3_K31_MODEL_TYPES
KIMI_K3_REPOSITORY = _identity.KIMI_K3_REPOSITORY
KIMI_K31_REPOSITORY = _identity.KIMI_K31_REPOSITORY
KIMI_K3_K31_REPOSITORIES = _identity.KIMI_K3_K31_REPOSITORIES
KIMI_K3_K31_TIKTOKEN_MODEL_SHA256 = _identity.KIMI_K3_K31_TIKTOKEN_MODEL_SHA256
KIMI_K3_K31_TOKENIZER_CONFIG_SHA256 = _identity.KIMI_K3_K31_TOKENIZER_CONFIG_SHA256
KIMI_K3_MODEL_NAMES = _identity.KIMI_K3_MODEL_NAMES
KIMI_K31_MODEL_NAMES = _identity.KIMI_K31_MODEL_NAMES
KIMI_K3_K31_MODEL_NAMES = _identity.KIMI_K3_K31_MODEL_NAMES


def is_deepseek_v4_model_name(model_name: str | None) -> bool:
    """Recognize the V4 family in an independently selected model name."""
    return _identity.is_deepseek_v4_model_name(model_name)


def is_kimi_k3_k31_model_name(model_name: str | None) -> bool:
    """Recognize the Kimi K3/K3.1 family in an independently selected model name."""
    return _identity.is_kimi_k3_k31_model_name(model_name)


def is_kimi_k3_k31_tokenizer_source(
    tokenizer_model: str | None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    """Recognize the Kimi K3/K3.1 family without claiming other families."""
    return _identity.is_kimi_k3_k31_tokenizer_source(
        tokenizer_model,
        revision=revision,
        local_files_only=local_files_only,
    )


def is_deepseek_v4_tokenizer_source(
    tokenizer_model: str | None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    """Recognize semantic V4 family without claiming explicit other families."""
    return _identity.is_deepseek_v4_tokenizer_source(
        tokenizer_model,
        revision=revision,
        local_files_only=local_files_only,
    )


def requires_code_free_tokenizer_loading(
    tokenizer_model: str | None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    """Restrict reviewed data loading without changing renderer identity."""
    return _identity.requires_code_free_tokenizer_loading(
        tokenizer_model,
        revision=revision,
        local_files_only=local_files_only,
    )
