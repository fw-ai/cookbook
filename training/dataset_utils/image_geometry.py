"""Qwen-VL image geometry checks for dataset rendering.

Qwen2-VL and the Qwen3 vision models that reuse its ``smart_resize`` reject
images whose long side is more than 200 times the short side. Detect that
here, before the processor raises a plain ``ValueError``, so training can
report a dataset error instead of an internal failure.
"""

from __future__ import annotations

QWEN_VL_MAX_ASPECT_RATIO = 200
QWEN_VL_RENDERER_PREFIXES = (
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
    "qwen3_6",
    "qwen3_8",
)

_QWEN_PROCESSOR_NAME_TOKENS = (
    "Qwen2VL",
    "Qwen2_5_VL",
    "Qwen3VL",
    "Qwen3_5",
)
_QWEN_PROCESSOR_MODULE_TOKENS = (
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
)


class ImageGeometryError(Exception):
    """An image's dimensions are incompatible with the selected vision model."""


def qwen_vl_aspect_ratio_message(width: int, height: int) -> str | None:
    """Return public copy when ``width`` x ``height`` exceeds the Qwen-VL cap."""
    if width <= 0 or height <= 0:
        return None
    ratio = max(width, height) / min(width, height)
    if ratio <= QWEN_VL_MAX_ASPECT_RATIO:
        return None
    return (
        f"Image aspect ratio {ratio:.1f} exceeds the maximum of "
        f"{QWEN_VL_MAX_ASPECT_RATIO} for this vision model. Crop or resize the "
        "image so the long side is less than "
        f"{QWEN_VL_MAX_ASPECT_RATIO} times the short side, then retry."
    )


def processor_enforces_qwen_vl_aspect_ratio(image_processor: object) -> bool:
    """Return whether ``image_processor`` uses the Qwen2-VL aspect-ratio cap."""
    processor_type = type(image_processor)
    if any(token in processor_type.__name__ for token in _QWEN_PROCESSOR_NAME_TOKENS):
        return True
    return any(token in processor_type.__module__ for token in _QWEN_PROCESSOR_MODULE_TOKENS)


def reject_qwen_vl_aspect_ratio(
    image_processor: object,
    *,
    width: int,
    height: int,
) -> None:
    """Raise ``ImageGeometryError`` when a Qwen-VL processor cannot resize the image."""
    if not processor_enforces_qwen_vl_aspect_ratio(image_processor):
        return
    message = qwen_vl_aspect_ratio_message(width, height)
    if message is not None:
        raise ImageGeometryError(message)
