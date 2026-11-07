"""Wrapper routing only; policy behavior belongs to the SDK's own suite."""

import importlib
import sys
import types
from unittest.mock import Mock

import pytest

from training import tokenizer_identity as identity

_SDK_MODULE = "fireworks.training.sdk.tokenizer_identity"
_PUBLIC = (
    "DEEPSEEK_V4_MODEL_TYPE",
    "DEEPSEEK_V4_REPOSITORIES",
    "DEEPSEEK_V4_TOKENIZER_JSON_SHA256",
    "DEEPSEEK_V41_MODEL_RE",
    "DEEPSEEK_V41_TOKENIZER_JSON_SHA256",
    "is_code_free_model_name",
    "is_deepseek_v4_model_name",
    "is_deepseek_v4_tokenizer_source",
    "requires_code_free_tokenizer_loading",
)


def test_selected_backend_and_public_surface() -> None:
    backend, module = identity._select_backend()
    assert backend == identity.TOKENIZER_IDENTITY_BACKEND
    assert module is identity._identity
    for name in _PUBLIC:
        assert getattr(identity, name) is not None
    for name in _PUBLIC[5:]:
        assert getattr(identity, name).__module__ == "training.tokenizer_identity"
    if backend == "sdk":
        assert module is importlib.import_module(_SDK_MODULE)
        assert module.TOKENIZER_IDENTITY_POLICY_CAPABILITY == identity._POLICY_CAPABILITY
        for name in _PUBLIC[:5]:
            assert getattr(identity, name) is getattr(module, name)


@pytest.mark.parametrize(
    "missing_name,falls_back", [(_SDK_MODULE, True), ("fireworks.training.sdk", True), ("torch", False)]
)
def test_missing_module_guard(monkeypatch: pytest.MonkeyPatch, missing_name: str, falls_back: bool) -> None:
    real_import = importlib.import_module

    def guarded(name: str, package: str | None = None) -> types.ModuleType:
        if name == _SDK_MODULE:
            raise ModuleNotFoundError(name=missing_name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded)
    if falls_back:
        backend, module = identity._select_backend()
        assert backend == "snapshot"
        assert module.__name__ == "training._tokenizer_identity_snapshot"
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            identity._select_backend()
        assert error.value.name == missing_name


@pytest.mark.parametrize(
    "capability",
    [
        None,
        "deepseek-v4-tokenizer-identity/exact-id-and-data-only-rule",
        "deepseek-v4-tokenizer-identity/exact-id-data-v41-isolation",
        "deepseek-v4-tokenizer-identity/explicit-family-before-digest",
        identity._POLICY_CAPABILITY,
    ],
)
def test_capability_selects_only_matching_sdk(monkeypatch: pytest.MonkeyPatch, capability: str | None) -> None:
    fake = types.ModuleType(_SDK_MODULE)
    if capability is not None:
        fake.TOKENIZER_IDENTITY_POLICY_CAPABILITY = capability
    monkeypatch.setitem(sys.modules, _SDK_MODULE, fake)
    backend, module = identity._select_backend()
    if capability == identity._POLICY_CAPABILITY:
        assert backend == "sdk" and module is fake
    else:
        assert backend == "snapshot"
        assert module.TOKENIZER_IDENTITY_POLICY_CAPABILITY == identity._POLICY_CAPABILITY


def test_wrappers_forward_source_and_download_options(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = Mock()
    monkeypatch.setattr(identity, "_identity", backend)
    for name in ("is_deepseek_v4_model_name", "is_code_free_model_name"):
        method = getattr(backend, name)
        assert getattr(identity, name)("model") is method.return_value
        method.assert_called_once_with("model")
    for name in ("is_deepseek_v4_tokenizer_source", "requires_code_free_tokenizer_loading"):
        method = getattr(backend, name)
        kwargs = dict(revision="pinned", local_files_only=True)
        assert getattr(identity, name)("source", **kwargs) is method.return_value
        method.assert_called_once_with("source", **kwargs)
