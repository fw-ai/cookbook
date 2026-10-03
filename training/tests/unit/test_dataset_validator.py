"""Renderer-owned dataset rules stay small and family-scoped."""

from __future__ import annotations

from training.dataset_validator import image_issue, register_image_rule


def test_qwen_family_rejects_extreme_aspect_ratio() -> None:
    message = image_issue("qwen3_5_interleaved", width=1, height=630)
    assert message is not None
    assert "630.0" in message
    assert "200" in message


def test_qwen_family_accepts_the_aspect_ratio_cap() -> None:
    assert image_issue("qwen3_vl_instruct", width=1, height=200) is None


def test_other_vision_families_are_not_subject_to_the_qwen_cap() -> None:
    assert image_issue("muse_glimmer", width=1, height=630) is None
    assert image_issue("", width=1, height=630) is None


def test_registered_rule_applies_only_to_its_renderer_family() -> None:
    def _too_wide(width: int, height: int) -> str | None:
        if width > height * 3:
            return "custom family rejects this image"
        return None

    register_image_rule(("custom_vl",), _too_wide)
    assert image_issue("custom_vl_preview", width=10, height=1) == (
        "custom family rejects this image"
    )
    assert image_issue("qwen3_5_interleaved", width=10, height=1) is None
