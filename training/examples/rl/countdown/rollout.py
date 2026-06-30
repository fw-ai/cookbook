"""Renderer-backed Countdown rollouts for text and vision prompts."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from tinker_cookbook.renderers import get_text_content

from training.examples.rl.countdown.reward import composite_reward
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.recipes.async_rl_loop import RolloutFn, RolloutSetup
from training.utils.rl.rollout import (
    RolloutRun,
    sample_vision_completion,
    single_turn_renderer_rollout,
)
from training.utils.supervised import build_renderer

import training.renderer  # noqa: F401  Registers cookbook-local renderers.

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a math puzzle solver. Given a target number and a set of available "
    "numbers, find an arithmetic expression using each number exactly once with "
    "operations +, -, *, / to reach the target.\n\n"
    "Show your reasoning inside <think>...</think> tags, then put your final "
    "equation inside <answer>...</answer> tags.\n\n"
    "Example:\n"
    "Target: 24, Numbers: [1, 2, 3, 4]\n"
    "<think>I need to reach 24. Let me try 1 * 2 * 3 * 4 = 24.</think>\n"
    "<answer>1 * 2 * 3 * 4</answer>"
)


def _numbers_text(row: dict[str, Any]) -> str:
    numbers = row.get("numbers") or row.get("nums")
    if numbers is None:
        raise KeyError("Countdown row must include 'numbers' or 'nums'")
    return "[" + ", ".join(str(int(value)) for value in numbers) + "]"


def _render_numbers_image(row: dict[str, Any]) -> str:
    if Image is None or ImageDraw is None:
        raise RuntimeError(
            "Vision Countdown requires Pillow. Install the cookbook with "
            "`pip install -e '.[eval]'` or `pip install -e '.[dev]'`."
        )

    numbers = _numbers_text(row)
    image = Image.new("RGB", (520, 180), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 510, 170), outline="black", width=3)
    draw.text((36, 46), "Available numbers:", fill="black")
    draw.text((36, 92), numbers, fill="black")

    out = io.BytesIO()
    image.save(out, format="PNG")
    payload = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_countdown_messages(row: dict[str, Any], *, variant: str) -> list[dict[str, Any]]:
    """Build a text-only or image-based Countdown prompt."""
    target = int(row["target"])
    if variant == "text":
        user_content: str | list[dict[str, Any]] = (
            f"Using the numbers {_numbers_text(row)}, create an equation that "
            f"equals {target}. You can use +, -, *, / and each number must be "
            "used exactly once."
        )
    elif variant == "vision":
        user_content = [
            {
                "type": "text",
                "text": (
                    "Using the numbers shown in the image, create an equation "
                    f"that equals {target}. You can use +, -, *, / and each "
                    "number must be used exactly once."
                ),
            },
            {"type": "image", "image": _render_numbers_image(row)},
        ]
    else:
        raise ValueError(f"Unknown Countdown variant: {variant}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parsed_text(parsed_message: Any) -> str:
    try:
        return str(get_text_content(parsed_message))
    except (KeyError, TypeError, AttributeError):
        if isinstance(parsed_message, dict):
            return str(parsed_message.get("content", ""))
        return str(parsed_message or "")


async def _reward_fn(
    row: dict[str, Any],
    parsed_message: Any,
    parse_success: bool,
) -> float:
    if not parse_success:
        return 0.0
    return composite_reward(_parsed_text(parsed_message), row)


def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    """Create a Countdown rollout function for ``async_rl_loop``."""
    variant = str(setup.extras.get("variant", "vision")).strip() or "vision"
    if variant not in {"text", "vision"}:
        raise ValueError("Countdown variant must be 'text' or 'vision'")

    sampler = build_deployment_sampler(setup)
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id)
    sample_kwargs = dict(setup.sample_kwargs)

    async def sample_with_vision(
        *,
        prompt_text: str,
        images: list[str],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        stop: list[str] | None = None,
        **extra_kwargs: Any,
    ) -> Any:
        return await sample_vision_completion(
            prompt_text=prompt_text,
            images=images,
            inference_base_url=setup.inference_base_url,
            api_key=setup.api_key,
            deployment_model=setup.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extra_kwargs=extra_kwargs,
        )

    async def rollout_fn(sample_prompt: dict[str, Any]) -> RolloutRun | None:
        try:
            return await single_turn_renderer_rollout(
                sample_prompt,
                renderer=renderer,
                sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
                sample_with_vision=sample_with_vision if variant == "vision" else None,
                message_builder=lambda row: build_countdown_messages(
                    row,
                    variant=variant,
                ),
                reward_fn=_reward_fn,
                sample_kwargs=sample_kwargs,
                tokenizer=setup.tokenizer,
                inference_base_url=setup.inference_base_url,
                api_key=setup.api_key,
                deployment_model=setup.model,
            )
        except Exception as exc:
            logger.warning("Countdown %s rollout failed: %s", variant, exc)
            return None

    return rollout_fn
