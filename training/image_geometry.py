"""Compatibility imports for checks now owned by ``training.dataset_utils.image_geometry``."""

from training.dataset_utils.image_geometry import (
    QWEN_VL_MAX_ASPECT_RATIO,
    QWEN_VL_RENDERER_PREFIXES,
    ImageGeometryError,
    processor_enforces_qwen_vl_aspect_ratio,
    qwen_vl_aspect_ratio_message,
    reject_qwen_vl_aspect_ratio,
)

__all__ = [
    "QWEN_VL_MAX_ASPECT_RATIO",
    "QWEN_VL_RENDERER_PREFIXES",
    "ImageGeometryError",
    "processor_enforces_qwen_vl_aspect_ratio",
    "qwen_vl_aspect_ratio_message",
    "reject_qwen_vl_aspect_ratio",
]
