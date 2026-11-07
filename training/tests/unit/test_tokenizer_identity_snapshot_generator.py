"""Parity checks for the generated tokenizer-identity fallback snapshot.

The snapshot (``training/_tokenizer_identity_snapshot.py``) must remain a
byte-exact generated copy of the SDK source of truth behind a static header,
so the cookbook fallback can never drift silently from the SDK policy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

GENERATOR_PATH = Path(__file__).resolve().parents[1] / "_tokenizer_identity_snapshot_generator.py"
PUBLIC_REPOS = Path(__file__).resolve().parents[4]
SDK_PATH = PUBLIC_REPOS / "python-sdk" / "src" / "fireworks" / "training" / "sdk" / "tokenizer_identity.py"
SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "_tokenizer_identity_snapshot.py"
KIMI_SDK_PATH = PUBLIC_REPOS / "python-sdk" / "src" / "fireworks" / "training" / "sdk" / "kimi_native_tokenizer.py"
KIMI_SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "utils" / "_kimi_native_tokenizer_snapshot.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_is_byte_exact_regeneration() -> None:
    if not SDK_PATH.exists():
        pytest.skip("SDK source of truth absent (standalone cookbook checkout)")
    generator = _load_module(GENERATOR_PATH, "tokenizer_identity_snapshot_generator")
    sdk_source = SDK_PATH.read_text(encoding="utf-8")
    assert SNAPSHOT_PATH.read_text(encoding="utf-8") == generator.generate_snapshot_text(sdk_source)


def test_snapshot_marks_temporary_duplication() -> None:
    first_line = SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert "TEMPORARY DUPLICATION" in first_line


def test_kimi_snapshot_is_byte_exact_regeneration() -> None:
    if not KIMI_SDK_PATH.exists():
        pytest.skip("SDK source of truth absent (standalone cookbook checkout)")
    generator = _load_module(GENERATOR_PATH, "tokenizer_identity_snapshot_generator")
    sdk_source = KIMI_SDK_PATH.read_text(encoding="utf-8")
    assert KIMI_SNAPSHOT_PATH.read_text(encoding="utf-8") == generator.generate_kimi_snapshot_text(sdk_source)


def test_kimi_snapshot_marks_temporary_duplication() -> None:
    first_line = KIMI_SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert "TEMPORARY DUPLICATION" in first_line


def test_sdk_encoders_are_deterministic_lint_adaptations() -> None:
    if not KIMI_SDK_PATH.exists():
        pytest.skip("SDK source of truth absent (standalone cookbook checkout)")
    generator = _load_module(GENERATOR_PATH, "tokenizer_identity_snapshot_generator")
    for profile, relative in (
        ("kimi_k3", "renderer/_vendored/kimi_k3/encoding_k3.py"),
        ("kimi_k31", "renderer/tito/_vendored/kimi_k31/encoding_k3.py"),
    ):
        source = (KIMI_SNAPSHOT_PATH.parents[1] / relative).read_text(encoding="utf-8")
        packaged = KIMI_SDK_PATH.parent / "_vendored" / profile / "encoding_k3.py"
        assert packaged.read_text(encoding="utf-8") == generator.generate_sdk_encoder_text(source, profile)
