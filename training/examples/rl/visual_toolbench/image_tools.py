"""PIL-backed image manipulation tools for VisualToolBench rollouts.

VisualToolBench tasks are designed so the model must *transform* the image
(crop into a small region, zoom, rotate, enhance contrast) to uncover details
it cannot read from the raw pixels alone.  This module exposes:

* ``TOOL_SPECS`` -- OpenAI-format tool schemas handed to the renderer's
  ``create_conversation_prefix_with_tools`` so the model sees the tools.
* ``execute_tool_call`` -- pure-PIL executor.  Bad arguments never raise; the
  model receives the error text as the tool response and can retry, which is
  itself learnable behavior under RL.

Images travel as base64 data URLs (the Fireworks completions ``images``
payload format).  Every tool result is re-encoded as JPEG and capped to
``max_dim`` so multi-turn trajectories keep a bounded token footprint.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional, Tuple

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "crop_image",
        "description": (
            "Crop a rectangular region from an image and return it as a new "
            "image. Coordinates are fractions of width/height in [0, 1], "
            "with (0, 0) at the top-left corner. Use this to isolate and "
            "inspect a small region in detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Which image to crop (0 = the first image shown).",
                },
                "x_min": {
                    "type": "number",
                    "description": "Left edge, fraction of width.",
                },
                "y_min": {
                    "type": "number",
                    "description": "Top edge, fraction of height.",
                },
                "x_max": {
                    "type": "number",
                    "description": "Right edge, fraction of width.",
                },
                "y_max": {
                    "type": "number",
                    "description": "Bottom edge, fraction of height.",
                },
            },
            "required": ["image_index", "x_min", "y_min", "x_max", "y_max"],
        },
    },
    {
        "name": "zoom_image",
        "description": (
            "Zoom into the center of an image by the given scale factor. The "
            "central region is enlarged to the image's original display size; "
            "use crop_image when the region of interest is not centered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Which image to zoom (0 = the first image shown).",
                },
                "factor": {
                    "type": "number",
                    "description": "Zoom factor; 2.0 enlarges the central half of the image. Range [1.1, 4].",
                },
            },
            "required": ["image_index", "factor"],
        },
    },
    {
        "name": "rotate_image",
        "description": "Rotate an image counter-clockwise by the given degrees (e.g. 90, 180, 270, or arbitrary angles).",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Which image to rotate (0 = the first image shown).",
                },
                "degrees": {
                    "type": "number",
                    "description": "Counter-clockwise rotation in degrees.",
                },
            },
            "required": ["image_index", "degrees"],
        },
    },
    {
        "name": "adjust_image",
        "description": (
            "Adjust brightness, contrast, and/or sharpness of an image. "
            "1.0 keeps a property unchanged; 2.0 doubles it; 0.5 halves it. "
            "Use this to recover detail from dark, washed-out, or blurry images."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Which image to adjust (0 = the first image shown).",
                },
                "brightness": {
                    "type": "number",
                    "description": "Brightness multiplier in [0.2, 4].",
                },
                "contrast": {
                    "type": "number",
                    "description": "Contrast multiplier in [0.2, 4].",
                },
                "sharpness": {
                    "type": "number",
                    "description": "Sharpness multiplier in [0.2, 4].",
                },
            },
            "required": ["image_index"],
        },
    },
]

TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}

_JPEG_QUALITY = 88


def _require_pil():
    try:
        from PIL import Image, ImageEnhance  # noqa: F401

        return Image, ImageEnhance
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "visual_toolbench requires pillow. Install it with "
            "`pip install pillow` (or `pip install -e .[eval]`)."
        ) from exc


def decode_data_url(data_url: str):
    """Decode a ``data:<mime>;base64,...`` payload into a PIL image."""
    Image, _ = _require_pil()
    _prefix, _sep, payload = str(data_url).partition(";base64,")
    if not payload:
        raise ValueError("image is not a base64 data URL")
    image = Image.open(io.BytesIO(base64.b64decode(payload)))
    return image.convert("RGB")


def encode_data_url(image, *, max_dim: int = 1024) -> str:
    """Encode a PIL image as a JPEG data URL, capping the longest side."""
    Image, _ = _require_pil()
    if max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        new_size = (
            max(1, round(image.size[0] * scale)),
            max(1, round(image.size[1] * scale)),
        )
        image = image.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=_JPEG_QUALITY)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed
    raise ValueError("tool arguments must be a JSON object")


def execute_tool_call(
    name: str,
    arguments: Any,
    workspace_images: List[str],
    *,
    max_dim: int = 1024,
) -> Tuple[Optional[str], str]:
    """Execute one tool call against the image workspace.

    Returns ``(new_image_data_url | None, result_text)``.  Invalid tool names
    or arguments produce ``(None, <error text>)`` so the rollout can surface
    the failure to the model instead of crashing the trajectory.
    """
    Image, ImageEnhance = _require_pil()
    try:
        args = _parse_arguments(arguments)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"Error: could not parse tool arguments ({exc})."

    if name not in TOOL_NAMES:
        return None, (
            f"Error: unknown tool {name!r}. Available tools: "
            + ", ".join(sorted(TOOL_NAMES))
            + "."
        )

    try:
        index = int(args.get("image_index", 0))
    except (TypeError, ValueError):
        return None, "Error: image_index must be an integer."
    if not 0 <= index < len(workspace_images):
        return None, (
            f"Error: image_index {index} out of range; there are "
            f"{len(workspace_images)} image(s), indexed from 0."
        )

    try:
        image = decode_data_url(workspace_images[index])
    except Exception as exc:
        return None, f"Error: could not decode image {index} ({exc})."

    width, height = image.size
    try:
        if name == "crop_image":
            x_min = _clamp(args["x_min"], 0.0, 1.0)
            y_min = _clamp(args["y_min"], 0.0, 1.0)
            x_max = _clamp(args["x_max"], 0.0, 1.0)
            y_max = _clamp(args["y_max"], 0.0, 1.0)
            if x_max - x_min < 0.01 or y_max - y_min < 0.01:
                return None, (
                    "Error: crop region is empty; require x_min < x_max and "
                    "y_min < y_max (fractions of the image size)."
                )
            box = (
                round(x_min * width),
                round(y_min * height),
                round(x_max * width),
                round(y_max * height),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                return None, (
                    "Error: crop region is empty after conversion to image pixels; "
                    "choose a larger region."
                )
            result = image.crop(box)
            # Upscale small crops so fine details stay legible after JPEG.
            if max(result.size) < max_dim // 2:
                factor = (max_dim // 2) / max(result.size)
                result = result.resize(
                    (
                        max(1, round(result.size[0] * factor)),
                        max(1, round(result.size[1] * factor)),
                    ),
                    Image.LANCZOS,
                )
            summary = f"Cropped image {index} to region ({x_min:.2f}, {y_min:.2f})-({x_max:.2f}, {y_max:.2f})."
        elif name == "zoom_image":
            factor = _clamp(args["factor"], 1.1, 4.0)
            crop_width = max(1, round(width / factor))
            crop_height = max(1, round(height / factor))
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            result = image.crop((left, top, left + crop_width, top + crop_height))
            target_long_side = min(max(width, height), max_dim)
            resize_scale = target_long_side / max(result.size)
            result = result.resize(
                (
                    max(1, round(result.size[0] * resize_scale)),
                    max(1, round(result.size[1] * resize_scale)),
                ),
                Image.LANCZOS,
            )
            summary = f"Zoomed into the center of image {index} by {factor:.2f}x."
        elif name == "rotate_image":
            degrees = float(args["degrees"]) % 360.0
            result = image.rotate(degrees, expand=True, fillcolor=(255, 255, 255))
            summary = (
                f"Rotated image {index} by {degrees:.1f} degrees counter-clockwise."
            )
        else:  # adjust_image
            result = image
            applied = []
            for prop, enhancer in (
                ("brightness", ImageEnhance.Brightness),
                ("contrast", ImageEnhance.Contrast),
                ("sharpness", ImageEnhance.Sharpness),
            ):
                if prop in args and args[prop] is not None:
                    factor = _clamp(args[prop], 0.2, 4.0)
                    result = enhancer(result).enhance(factor)
                    applied.append(f"{prop}={factor:.2f}")
            if not applied:
                return None, (
                    "Error: adjust_image needs at least one of brightness, "
                    "contrast, or sharpness."
                )
            summary = f"Adjusted image {index} ({', '.join(applied)})."
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"Error: invalid arguments for {name} ({exc})."

    new_data_url = encode_data_url(result, max_dim=max_dim)
    new_index = len(workspace_images)
    return new_data_url, (
        f"{summary} The result is image {new_index} "
        f"({result.size[0]}x{result.size[1]} before re-encoding), shown below."
    )
