"""Public selection contract for the Nemotron 3.5 Super VL renderer."""

from __future__ import annotations

import pytest

from training.renderer.nemotron35 import RENDERER_NAMES
from training.utils.supervised import resolve_renderer_name

_PUBLIC_NAMES = (
    "nemotron35_vl",
    "nemotron35_vl_disable_thinking",
)


def test_selectable_names_are_the_public_modes() -> None:
    assert RENDERER_NAMES == _PUBLIC_NAMES


def test_vl_checkpoints_select_nemotron35_vl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.utils.supervised.resolve_renderer_name_from_plugins",
        lambda _tokenizer_model: None,
    )
    for model in (
        "accounts/fireworks/models/nemotron-3p5-super-vl",
        "accounts/fireworks-e2e-tests/models/nemotron-3p5-super-vl",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
        "nemotron-3p5-super-vl",
    ):
        assert resolve_renderer_name(model) == "nemotron35_vl"
    assert resolve_renderer_name("any-model", "nemotron35_vl_disable_thinking") == "nemotron35_vl_disable_thinking"


def test_text_nemotron_checkpoints_stay_on_the_text_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.utils.supervised.resolve_renderer_name_from_plugins",
        lambda _tokenizer_model: None,
    )
    for model in (
        "nvidia/NVIDIA-Nemotron-3-Super-120B",
        "accounts/pyroworks/models/nemotron-3-5-super-ea-0911-textonly",
        "accounts/fireworks/models/nemotron-3p5-super-vl-unreviewed",
        "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
    ):
        expected = "nemotron3_ultra" if "lightning" in model else "nemotron3"
        assert resolve_renderer_name(model) == expected


def test_internal_selectors_are_not_inferred_as_a_public_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "training.utils.supervised.resolve_renderer_name_from_plugins",
        lambda _tokenizer_model: None,
    )
    with pytest.raises(ValueError):
        resolve_renderer_name("fireworks-internal/nemotron-preview")
