"""Compact DeepSeek-V4.1 delta over the V4 renderer."""

from __future__ import annotations

import copy
import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tinker
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from training._vendor.tinker_cookbook_0_4_3.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    TrainOnWhat,
    image_to_chunk,
)
from training.renderer import register_renderer
from training.renderer.deepseek_v4 import (
    _RESPONSE_FORMAT_TEMPLATE,
    DeepseekV4Renderer,
    _extract_reasoning_and_text,
    _merge_tool_messages,
    _sort_tool_results_by_call_order,
    _to_json,
)
from training.renderer.message_weights import untrained_synthesized_context
from training.renderer.reasoning_fields import (
    original_reasoning,
    original_reasoning_content,
)
from training.renderer.tokenizer import Tokenizer

_SYSTEM_SP = "<｜System｜>"
_REASONING_EFFORT = (
    "Reasoning Effort: 75 "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
_THINKING_START = "_deepseek_v41_thinking_start"
_IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def _without_image_placeholder(text: str) -> str:
    if _IMAGE_PLACEHOLDER in text:
        raise ValueError(
            "Raw DeepSeek V4.1 image placeholders are unsupported; use an image content part"
        )
    return text


class DeepseekV41ImageTokenCounter:
    def __init__(
        self,
        *,
        patch_size: int,
        downsample_ratio: int,
        max_image_tokens: int,
        min_pixels: int,
        max_wh_ratio: float | None,
    ) -> None:
        self.patch_size = int(patch_size)
        self.downsample_ratio = int(downsample_ratio)
        self.max_image_tokens = int(max_image_tokens)
        self.min_pixels = int(min_pixels)
        self.max_wh_ratio = None if max_wh_ratio is None else float(max_wh_ratio)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "DeepseekV41ImageTokenCounter":
        model_path = Path(model_name)
        config_path = (
            model_path / "config.json"
            if model_path.is_dir()
            else Path(hf_hub_download(model_name, "config.json"))
        )
        vision = json.loads(config_path.read_text())["vision_config"]
        return cls(
            patch_size=vision["patch_size"],
            downsample_ratio=vision["downsample_ratio"],
            max_image_tokens=vision.get("max_image_tokens", 1024),
            min_pixels=vision.get("min_pixels", 295936),
            max_wh_ratio=vision.get("max_wh_ratio"),
        )

    def media_tokens_calculator(self, media: dict[str, Any]) -> int:
        image = media.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError("DeepSeek V4.1 image media must contain a PIL image")
        raw_width, height = image.size
        width = float(raw_width)
        if self.max_wh_ratio is not None and width > height * self.max_wh_ratio:
            width = height * self.max_wh_ratio
        if width * height < self.min_pixels:
            scale = math.sqrt(self.min_pixels / (width * height))
            width, height = int(width * scale), int(height * scale)

        patch = self.patch_size
        target_width = math.ceil(width / patch) * patch
        target_height = math.ceil(height / patch) * patch

        def grid(canvas_height: int, canvas_width: int) -> tuple[int, int]:
            return (
                math.ceil((canvas_height // patch) / self.downsample_ratio),
                math.ceil((canvas_width // patch) / self.downsample_ratio),
            )

        llm_height, llm_width = grid(target_height, target_width)
        if llm_height * (llm_width + 1) + 2 > self.max_image_tokens:
            ratio = height / width
            max_width = math.sqrt((self.max_image_tokens - 2) / ratio + 0.25) - 0.5
            max_height = max_width * ratio
            cell = patch * self.downsample_ratio
            if max_width < 1:
                target_height, target_width = (
                    (self.max_image_tokens - 2) // 2 * cell,
                    cell,
                )
            elif max_height < 1:
                target_height, target_width = (
                    cell,
                    (self.max_image_tokens - 3) * cell,
                )
            else:
                scale = min(
                    math.floor(max_width) * cell / width,
                    math.floor(max_height) * cell / height,
                )
                target_height = math.floor(height * scale / patch) * patch
                target_width = math.floor(width * scale / patch) * patch
            llm_height, llm_width = grid(target_height, target_width)
        return llm_height * (llm_width + 1) + 2

    def get_resize_config(self, media: dict[str, Any]) -> dict[str, int]:
        return {"num_tokens": self.media_tokens_calculator(media)}


def _assistant_reasoning_and_text(message: Mapping[str, Any]) -> tuple[str, str]:
    explicit = message.get("reasoning_content")
    if not isinstance(explicit, str) or not explicit:
        has_reasoning, reasoning = original_reasoning(message)
        has_reasoning_content, reasoning_content = original_reasoning_content(message)
        explicit = (
            reasoning
            if has_reasoning and reasoning
            else reasoning_content
            if has_reasoning_content and reasoning_content
            else ""
        )
    extracted, visible = _extract_reasoning_and_text(message.get("content"))
    return _without_image_placeholder(
        explicit or extracted
    ), _without_image_placeholder(visible)


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": _without_image_placeholder(content)}]
    if not isinstance(content, list):
        raise TypeError("DeepSeek V4.1 user content must be text or content parts")
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            raise TypeError("DeepSeek V4.1 content parts must be objects")
        kind = part.get("type")
        if kind == "text":
            blocks.append(
                {
                    "type": "text",
                    "text": _without_image_placeholder(str(part.get("text", ""))),
                }
            )
        elif kind == "image":
            blocks.append({"type": "image", "image": part.get("image")})
        else:
            raise ValueError(f"Unsupported DeepSeek V4.1 user content part: {kind!r}")
    return blocks


class DeepseekV41Renderer(DeepseekV4Renderer):
    _user_sp = "<｜User｜>"
    _tool_calls_name = " calls"
    _tool_invoke_name = " invoke"
    _tool_parameter_name = " parameter"
    _tool_argument_decode_passes = 2
    _strict_tool_parsing = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        image_processor: Any | None = None,
    ) -> None:
        super().__init__(tokenizer)
        self.image_processor = image_processor

    def _last_assistant_turn_boundary(self, messages: list[Message]) -> int:
        boundary = -1
        for index, message in enumerate(messages):
            if message.get("role") == "system" and index > 0:
                boundary = index
            elif message.get("role") == "user" and any(
                block.get("type") in {"text", "image"}
                for block in message.get("content_blocks", [])
            ):
                boundary = index
        return boundary

    def _is_disaggregation_turn_boundary(
        self,
        message: Mapping[str, Any],
        index: int,
    ) -> bool:
        return message.get("role") == "user" or (
            message.get("role") == "system" and index > 0
        )

    def _preprocess(self, messages: list[Message]) -> list[Message]:
        normalized: list[Message] = []
        for original in messages:
            message = copy.deepcopy(original)
            if message.get("role") == "assistant":
                reasoning, visible = _assistant_reasoning_and_text(message)
                message["content"] = visible
                if reasoning:
                    message["reasoning_content"] = reasoning
            normalized.append(message)

        merged = _sort_tool_results_by_call_order(
            _merge_tool_messages(normalized, user_blocks=_user_blocks)
        )
        self._effective_strip = self.strip_thinking_from_history and not any(
            message.get("tools") for message in merged
        )
        boundary = self._last_assistant_turn_boundary(merged)
        for index, message in enumerate(merged):
            if message.get("role") != "assistant":
                continue
            starts_thinking = not self._effective_strip or index > boundary
            message[_THINKING_START] = starts_thinking
            if not starts_thinking:
                message.pop("reasoning_content", None)
        return untrained_synthesized_context(merged)

    def _assistant_starts_thinking(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> bool:
        del ctx
        return bool(message.get(_THINKING_START))

    def _assistant_emits_header(self, message: Message, ctx: RenderContext) -> bool:
        del message
        if ctx.prev_message is None:
            return False
        role = ctx.prev_message.get("role")
        return role == "user" or (role == "system" and ctx.idx > 1)

    def _assistant_header_str(self, message: Message, ctx: RenderContext) -> str:
        prefix = _SYSTEM_SP + _REASONING_EFFORT if ctx.idx == 0 else ""
        return prefix + super()._assistant_header_str(message, ctx)

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        role = message["role"]
        if role == "system":
            prefix = _SYSTEM_SP + (_REASONING_EFFORT if ctx.idx == 0 else "")
            blocks = _user_blocks(message.get("content"))
            if any(block["type"] != "text" for block in blocks):
                raise ValueError("DeepSeek V4.1 system messages support text only")
            body = "\n\n".join(block["text"] for block in blocks)
            if message.get("tools"):
                body += "\n\n" + self._render_tools_section(list(message["tools"]))
            if message.get("response_format"):
                body += "\n\n" + _RESPONSE_FORMAT_TEMPLATE.format(
                    schema=_to_json(message["response_format"])
                )
            return RenderedMessage(
                header=tinker.types.EncodedTextChunk(tokens=self._encode(prefix)),
                output=[tinker.types.EncodedTextChunk(tokens=self._encode(body))],
            )
        if role == "user":
            prefix = (
                _SYSTEM_SP + _REASONING_EFFORT if ctx.idx == 0 else ""
            ) + self._user_sp
            # Encode each run between images as one string: BPE merges across
            # block separators (".\n\n", ">\n\n") must match the publisher prompt.
            output: list[tinker.types.ModelInputChunk] = []
            pending = ""
            for index, block in enumerate(message.get("content_blocks", [])):
                if index:
                    pending += "\n\n"
                kind = block.get("type")
                if kind == "text":
                    text = str(block.get("text", ""))
                elif kind == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        if any(part.get("type") != "text" for part in content):
                            raise ValueError(
                                "DeepSeek V4.1 tool-result images are unsupported"
                            )
                        content = "\n\n".join(
                            str(part.get("text", "")) for part in content
                        )
                    text = f"<tool_result>{_without_image_placeholder(str(content))}</tool_result>"
                elif kind == "image":
                    if self.image_processor is None:
                        raise ValueError(
                            "DeepSeek V4.1 image messages require an image processor"
                        )
                    image = block.get("image")
                    if isinstance(image, bytes):
                        image = Image.open(io.BytesIO(image))
                        image.load()
                    if pending:
                        output.append(
                            tinker.types.EncodedTextChunk(tokens=self._encode(pending))
                        )
                        pending = ""
                    output.append(image_to_chunk(image, self.image_processor))
                    continue
                else:
                    raise ValueError(
                        f"Unsupported DeepSeek V4.1 content block: {kind!r}"
                    )
                pending += text
            if pending:
                output.append(
                    tinker.types.EncodedTextChunk(tokens=self._encode(pending))
                )
            return RenderedMessage(
                header=tinker.types.EncodedTextChunk(tokens=self._encode(prefix)),
                output=output,
            )
        return super().render_message(message, ctx)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        prepared = self._preprocess(messages)
        has_mid_system = any(
            message.get("role") == "system" and index > 0
            for index, message in enumerate(prepared)
        )
        has_user = any(message.get("role") == "user" for message in prepared)
        if train_on_what == TrainOnWhat.LAST_ASSISTANT_TURN and (
            has_mid_system
            or not has_user
            or any(message.get("tools") for message in prepared)
        ):
            model_input, weights = self._build_tool_turn_example(prepared)
        else:
            model_input, weights = Renderer.build_supervised_example(
                self,
                prepared,
                train_on_what=train_on_what,
            )
        offset = 0
        for chunk in model_input.chunks:
            if isinstance(chunk, tinker.types.ImageChunk):
                weights[offset : offset + chunk.length] = 0
            offset += chunk.length
        return model_input, weights


def _deepseek_v41_factory(
    tokenizer: Tokenizer,
    image_processor: Any = None,
) -> DeepseekV41Renderer:
    return DeepseekV41Renderer(tokenizer, image_processor=image_processor)


register_renderer("deepseek_v41", _deepseek_v41_factory)
