"""Beam Harmony renderer backed by the tokenizer's XML chat template."""

from __future__ import annotations

import re
from typing import Any

import torch
from fireworks.training import sdk

from training.renderer import register_renderer
from training.renderer.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolSpec,
    TrainOnWhat,
)
from training.renderer.errors import RendererError
from training.renderer.reasoning_fields import (
    original_reasoning,
    original_reasoning_content,
)

from training.renderer.tokenizer import Tokenizer

_CONTROL_MARKER = re.compile(r"<\|(?:start|end|return|channel|message|call|constrain)\|>")

_WIRE = "_beam_wire"
_HEADER = "<|start|>assistant"
_SEGMENT = re.compile(
    r"(?:<\|start\|>)?assistant<\|channel\|>(analysis|final)<\|message\|>(.*?)(<\|end\|>|<\|return\|>)",
    re.DOTALL,
)

_PARTIAL_SEGMENT = re.compile(
    r"(?:<\|start\|>)?assistant<\|channel\|>(analysis|final)<\|message\|>(.*)",
    re.DOTALL,
)


class BeamRenderer(Renderer):
    """Preserve reasoning history; generate both channel headers after ``assistant``."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)
        if not getattr(tokenizer, "chat_template", None):
            raise RendererError("Beam requires its tokenizer chat_template.jinja.")

    @property
    def has_extension_property(self) -> bool:
        return True

    def get_stop_sequences(self) -> list[str]:
        return ["<|return|>"]

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        if (
            message["role"] not in {"system", "developer", "user", "assistant"}
            or message.get("tools")
            or message.get("tool_calls")
            or message.get("unparsed_tool_calls")
        ):
            raise RendererError("Beam tool calling is not supported by this adapter.")
        result = super().to_openai_message(message)
        content = message["content"]
        if isinstance(content, list):
            if any(part["type"] not in {"text", "thinking"} for part in content):
                raise RendererError("Beam supports text and thinking content only.")
            result["content"] = "".join(p["text"] for p in content if p["type"] == "text")
            result["reasoning_content"] = "".join(p["thinking"] for p in content if p["type"] == "thinking")
        has_reasoning, reasoning = original_reasoning(message)
        has_content, reasoning_content = original_reasoning_content(message)
        if has_reasoning or has_content:
            result["reasoning_content"] = reasoning_content or reasoning
        if "thinking" in message:
            result["thinking"] = message["thinking"]
        return result

    def _template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

    def _encode(self, text: str) -> sdk.EncodedTextChunk:
        return sdk.EncodedTextChunk(tokens=self.tokenizer.encode(text, add_special_tokens=False))

    def _prepare(self, messages: list[Message]) -> list[Message]:
        leading = 0
        while leading < len(messages) and messages[leading]["role"] in {
            "system",
            "developer",
        }:
            leading += 1
        instructions = messages[:leading]
        # A user sentinel lets even an empty instruction list render its preamble.
        sentinel = {"role": "user", "content": ""}
        wire = self._template([self.to_openai_message(m) for m in instructions] + [sentinel])
        wire = wire.removesuffix("<|start|>user<|message|><|end|>")
        preamble: Message = {"role": "system", "content": "", _WIRE: wire}
        flags = {m["trainable"] for m in instructions if "trainable" in m}
        if len(flags) > 1:
            raise RendererError("Beam combines leading instructions; their trainable flags must agree.")
        if any("trainable" in m for m in messages):
            preamble["trainable"] = next(iter(flags), False)
        prepared = [preamble]
        for message in messages[leading:]:
            if message["role"] in {"system", "developer"}:
                # The canonical template ignores instructions after the first user.
                continue
            prepared.append(message)
        return prepared

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> sdk.ModelInput:
        if role != "assistant":
            raise RendererError("Beam generation requires the assistant role.")
        return super().build_generation_prompt(self._prepare(messages), role, prefill)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[sdk.ModelInput, torch.Tensor]:
        return super().build_supervised_example(self._prepare(messages), train_on_what)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        return self.tokenizer.encode(_HEADER, add_special_tokens=False)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        if _WIRE in message:
            wire = message[_WIRE]
            # Treat injected context as a header so assistant-targeted training
            # masks it. Dataset instructions retain normal message loss policy.
            boundary = wire.find("<|start|>developer<|message|>")
            if boundary < 0:
                return RenderedMessage(header=self._encode(wire), output=[])
            boundary += len("<|start|>developer<|message|>")
            return RenderedMessage(
                header=self._encode(wire[:boundary]),
                output=[self._encode(wire[boundary:])],
            )
        raw = self.to_openai_message(message)
        wire = self._template([raw])
        # Each standalone template render injects the fixed system preamble.
        wire = wire.split("<|end|>", 1)[1]
        if message["role"] == "assistant":
            header = _HEADER
        else:
            header = wire.split("<|message|>", 1)[0] + "<|message|>"
        return RenderedMessage(header=self._encode(header), output=[self._encode(wire[len(header) :])])

    def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
        if tools:
            raise RendererError("Beam tool calling is not supported by this adapter.")
        return [{"role": "system", "content": system_prompt}] if system_prompt else []

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        text = self.tokenizer.decode(response, skip_special_tokens=False)
        if not text.startswith("<|start|>"):
            text = _HEADER + text
        message: Message = {"role": "assistant", "content": []}
        position = 0
        terminal = None
        for match in _SEGMENT.finditer(text):
            if match.start() != position or terminal is not None:
                break
            channel, body, ending = match.groups()
            if (channel == "final") != (ending == "<|return|>"):
                break
            if body:
                message["content"].append(
                    {"type": "thinking", "thinking": body} if channel == "analysis" else {"type": "text", "text": body}
                )
            position = match.end()
            if ending == "<|return|>":
                terminal = ending
        if position != len(text) or terminal is None:
            # Recover only a recognized channel body. An incomplete header or
            # invalid trailing wire is not visible answer text.
            partial = _PARTIAL_SEGMENT.match(text, position) if terminal is None else None
            if partial:
                channel, body = partial.groups()
                body = _CONTROL_MARKER.split(body, maxsplit=1)[0]
                if body:
                    message["content"].append(
                        {"type": "thinking", "thinking": body}
                        if channel == "analysis"
                        else {"type": "text", "text": body}
                    )
            return message, ParseTermination.MALFORMED
        return message, ParseTermination.STOP_SEQUENCE


def _factory(tokenizer: Tokenizer, image_processor: Any = None) -> BeamRenderer:
    return BeamRenderer(tokenizer)


register_renderer("beam", _factory)
