"""Renderer for ZhipuAI GLM-5.1 chat template.

Handles the GLM-5.1 chat format and has been validated end-to-end via
SFT training on ``accounts/fireworks/models/glm-5p1``. Other GLM
versions are out of scope for this renderer.

Layout for a user/assistant turn::

    [gMASK]<sop><|user|>
    {user_content}<|assistant|>
    <think></think>
    {assistant_content}<|endoftext|>

Notes:

- ``[gMASK]<sop>`` is emitted once at the very start of the conversation
  (no trailing newline — the upstream template strips it).
- Role tags are ``<|system|>``, ``<|user|>``, ``<|assistant|>``,
  ``<|observation|>``; each is followed by a literal newline except
  ``<|assistant|>`` whose newline is produced as part of the assistant
  output so the model generates it.
- Every assistant turn begins with ``<think>...</think>``. Historical
  assistant turns (before the last user turn) and non-thinking-mode
  outputs collapse to ``<think></think>``.
- The upstream Jinja template does *not* emit ``<|endoftext|>`` at
  message boundaries. This renderer appends the eos token only to the
  *last* message in the conversation, so training on the final assistant
  turn teaches the model when to stop while historical assistant turns
  still match the Jinja byte-for-byte.

Only text-only chat, with and without thinking, is implemented here.
Tools and multimodal content are left for a future extension.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tinker
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    parse_response_for_stop_token,
    parse_think_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_BOS_TEXT = "[gMASK]<sop>"


def _visible_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif (
                isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _extract_reasoning_and_text(content: Any) -> tuple[str, str]:
    """Return ``(reasoning, visible_text)`` for an assistant message."""
    if isinstance(content, str):
        if "</think>" not in content:
            return "", content
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
        visible = content.split("</think>")[-1].lstrip("\n")
        return reasoning, visible

    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_parts.append(part["thinking"])
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    return "".join(reasoning_parts), "".join(text_parts)


class GLM5Renderer(Renderer):
    """Renderer for ZhipuAI GLM-5.1 instruct models."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = True,
    ) -> None:
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    @property
    def has_extension_property(self) -> bool:
        return not self.strip_thinking_from_history

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode(_BOS_TEXT, add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            raise RuntimeError(
                "GLM5Renderer requires tokenizer.eos_token_id to be set "
                "(expected <|endoftext|>)."
            )
        return int(eos)

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        if role == "assistant":
            return self._render_assistant(message, ctx)
        if role == "user":
            header_str = "<|user|>\n"
            output_str = _visible_text(message["content"])
        elif role == "system":
            header_str = "<|system|>\n"
            output_str = _visible_text(message["content"])
        elif role == "tool":
            header_str = "<|observation|>\n"
            output_str = (
                f"<tool_response>\n{_visible_text(message['content'])}\n</tool_response>"
            )
        else:
            raise ValueError(f"GLM5Renderer: unsupported role {role!r}")

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False),
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _render_assistant(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        # Header is just the role tag. The leading "\n" and the <think>
        # block live in `output` so they are part of the training target
        # for this assistant turn.
        header_str = "<|assistant|>"

        is_historical = (
            self.strip_thinking_from_history
            and ctx.last_user_index >= 0
            and ctx.idx < ctx.last_user_index
        )

        reasoning, visible = _extract_reasoning_and_text(message["content"])
        if is_historical or not reasoning:
            think_block = "<think></think>"
        else:
            think_block = f"<think>{reasoning.strip()}</think>"

        parts = ["\n", think_block]
        visible_stripped = visible.strip()
        if visible_stripped:
            parts.append("\n")
            parts.append(visible_stripped)
        output_str = "".join(parts)

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)
        # Append <|endoftext|> only on the final message in the conversation.
        # Historical assistant turns are delimited only by the next role tag
        # in the GLM Jinja template, so emitting EOS there would add a token
        # the template never produces. The final turn still gets EOS so the
        # trained model learns when to stop.
        if ctx.is_last:
            output_tokens.append(self._end_message_token)

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=output_tokens),
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del ctx
        suffix_str = f"<|{role}|>" if role in ("assistant", "user", "system") else f"<|{role}|>"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, ok = parse_response_for_stop_token(
            response=response,
            tokenizer=self.tokenizer,
            stop_token=self._end_message_token,
        )
        if not ok:
            return assistant_message, False
        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"].lstrip("\n")
        parts = parse_think_blocks(content)
        assistant_message["content"] = parts if parts is not None else content
        return assistant_message, True


def _glm5_factory(tokenizer: Tokenizer, image_processor=None) -> GLM5Renderer:
    del image_processor
    return GLM5Renderer(tokenizer)


register_renderer("glm5", _glm5_factory)
