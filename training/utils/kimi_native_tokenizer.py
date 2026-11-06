"""Cookbook Kimi K3/K3.1 native tokenizer loading (thin public wrapper).

Delegates to ``fireworks.training.sdk.kimi_native_tokenizer`` when the
installed SDK carries a matching ``KIMI_NATIVE_TOKENIZER_CAPABILITY``;
otherwise falls back to the generated snapshot in
``training/utils/_kimi_native_tokenizer_snapshot.py``. Only
``ModuleNotFoundError`` naming the requested module or one of its parents is
treated as absence; any other import failure propagates unchanged.

The loaded tokenizer is built entirely from reviewed installed code and
hash-pinned data files; checkpoint Python (``tokenization_kimi.py``) is
hash-checked as provenance evidence but never imported.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

_CAPABILITY = "kimi-k3-k31-native-tokenizer/chat-and-saved-bundles-v3"
_SDK_MODULE = "fireworks.training.sdk.kimi_native_tokenizer"

# Which backend supplied the implementation: "sdk" (delegation) or
# "snapshot" (older SDK floor fallback). Exposed for diagnostics and tests.
KIMI_NATIVE_TOKENIZER_BACKEND: str


def _load_sdk_module() -> Any:
    """Return the SDK native-tokenizer module, or None when it is absent."""
    try:
        return importlib.import_module(_SDK_MODULE)
    except ModuleNotFoundError as exc:
        _requested = _SDK_MODULE.split(".")
        _missing = (exc.name or "").split(".")
        if not _missing or _missing != _requested[: len(_missing)]:
            raise
        return None


def _select_backend() -> tuple[str, Any]:
    """Pick the implementation for this interpreter.

    "sdk" requires both the module import and the capability marker, so an
    SDK that ships the module without the final capability falls back to the
    snapshot instead of inheriting the older behavior.
    """
    sdk_module = _load_sdk_module()
    if sdk_module is not None and getattr(sdk_module, "KIMI_NATIVE_TOKENIZER_CAPABILITY", None) == _CAPABILITY:
        return "sdk", sdk_module
    # lazy: compatibility source is needed only for older SDKs.
    from training.utils import _kimi_native_tokenizer_snapshot as snapshot

    return "snapshot", snapshot


KIMI_NATIVE_TOKENIZER_BACKEND, _impl = _select_backend()

KIMI_K3_PROFILE = _impl.KIMI_K3_PROFILE
KIMI_K31_PROFILE = _impl.KIMI_K31_PROFILE
KIMI_K3_REPOSITORY = _impl.KIMI_K3_REPOSITORY
KIMI_K3_REVISION = _impl.KIMI_K3_REVISION
KIMI_K31_REPOSITORY = _impl.KIMI_K31_REPOSITORY
KIMI_K31_REVISION = _impl.KIMI_K31_REVISION
KIMI_K3_SOURCE_SHA256 = _impl.KIMI_K3_SOURCE_SHA256
KIMI_K31_SOURCE_SHA256 = _impl.KIMI_K31_SOURCE_SHA256
KIMI_K3_K31_PAT_STR = _impl.KIMI_K3_K31_PAT_STR
KIMI_BASE_VOCAB_SIZE = _impl.KIMI_BASE_VOCAB_SIZE
KIMI_NUM_RESERVED_SPECIAL_TOKENS = _impl.KIMI_NUM_RESERVED_SPECIAL_TOKENS


def verify_source_dir(source_dir: str | Path, *, profile: str | None = None) -> str:
    """Fail closed unless every pinned source file is present and unchanged."""
    return _impl.verify_source_dir(source_dir, profile=profile)


def special_tokens(source_dir: str | Path) -> list[tuple[str, int]]:
    """Every special token in id order, named exactly as ``TikTokenTokenizer`` names them."""
    return _impl.special_tokens(source_dir)


def build_fast_tokenizer(source_dir: str | Path, *, profile: str | None = None) -> Any:
    """Convert the pinned tiktoken vocabulary into an id-identical fast tokenizer."""
    return _impl.build_fast_tokenizer(source_dir, profile=profile)


def load_native_tokenizer(
    tokenizer_model: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    profile: str | None = None,
) -> Any:
    """Build the native fast tokenizer for a reviewed Kimi K3/K3.1 source."""
    return _impl.load_native_tokenizer(
        tokenizer_model,
        revision=revision,
        local_files_only=local_files_only,
        profile=profile,
    )


__all__ = [
    "KIMI_BASE_VOCAB_SIZE",
    "KIMI_K3_PROFILE",
    "KIMI_K3_REPOSITORY",
    "KIMI_K3_REVISION",
    "KIMI_K3_SOURCE_SHA256",
    "KIMI_K31_PROFILE",
    "KIMI_K31_REPOSITORY",
    "KIMI_K31_REVISION",
    "KIMI_K31_SOURCE_SHA256",
    "KIMI_K3_K31_PAT_STR",
    "KIMI_NATIVE_TOKENIZER_BACKEND",
    "KIMI_NUM_RESERVED_SPECIAL_TOKENS",
    "build_fast_tokenizer",
    "load_native_tokenizer",
    "special_tokens",
    "verify_source_dir",
]
