"""Lightweight renderer-owned dataset checks.

FireTitan walks the JSONL and enforces row shape. This module answers the
questions that belong to a renderer family and can be decided without
rendering the row, such as whether an image's dimensions fit the vision
model. Render remains the backstop for images whose bytes are not embedded
in the dataset.
"""

from __future__ import annotations

from collections.abc import Callable

from training.image_geometry import (
    QWEN_VL_RENDERER_PREFIXES,
    qwen_vl_aspect_ratio_message,
)

ImageRule = Callable[[int, int], str | None]

_IMAGE_RULES: list[tuple[tuple[str, ...], ImageRule]] = []


def register_image_rule(renderer_prefixes: tuple[str, ...], rule: ImageRule) -> None:
    """Register a dimension check for renderer names that start with ``renderer_prefixes``."""
    if not renderer_prefixes or any(not prefix for prefix in renderer_prefixes):
        raise ValueError("renderer_prefixes must be non-empty strings")
    if not callable(rule):
        raise TypeError("image rule must be callable")
    _IMAGE_RULES.append((renderer_prefixes, rule))


def image_issue(renderer_name: str, *, width: int, height: int) -> str | None:
    """Return public dataset-error copy when a registered rule rejects the image."""
    for prefixes, rule in _IMAGE_RULES:
        if not renderer_name.startswith(prefixes):
            continue
        message = rule(width, height)
        if message:
            return message
    return None


register_image_rule(QWEN_VL_RENDERER_PREFIXES, qwen_vl_aspect_ratio_message)
