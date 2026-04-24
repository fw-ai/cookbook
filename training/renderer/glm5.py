"""Renderer for ZhipuAI GLM-5.1 chat template.

Handles the GLM-5.1 chat format as shipped with ``zai-org/GLM-5.1``
(and its FP8 variant ``zai-org/GLM-5.1-FP8``, which ships an identical
tokenizer and chat template).

Token-level layout matches ``tokenizer.apply_chat_template`` byte-for-byte
(verified by the unit tests in ``test_glm5_renderer.py``), modulo the
intentional training EOS on the terminal assistant message.

Role tag layout (as the shipped Jinja template emits them):

- ``<|system|>{content}``  — no newline after the tag, no newline before the next tag
- ``<|user|>{content}``    — same
- ``<|assistant|>...``     — see below
- ``<|observation|>{content}`` — same

Assistant turn layout:

- **Terminal turn, ``enable_thinking=True`` (default), no reasoning content**::

      <|assistant|><think></think>{content}

- **Terminal turn, reasoning content provided**::

      <|assistant|><think>{reasoning}</think>{content}

- **Terminal turn, ``enable_thinking=False``** (or non-thinking mode) — the
  shipped template emits ``</think>`` alone so the model skips the think
  phase::

      <|assistant|></think>{content}

- **Historical assistant turn** (any turn before the last user message) when
  ``strip_thinking_from_history=True`` (the default)::

      <|assistant|></think>{content}

  Historical turns drop any reasoning content to avoid leaking intermediate
  reasoning into later turns' context. Matches the shipped template's
  ``loop.index0 > ns.last_user_index`` branch.

Other invariants:

- ``[gMASK]<sop>`` is emitted once at the very start of the conversation.
- The shipped Jinja template does **not** emit ``<|endoftext|>`` at message
  boundaries. This renderer appends ``<|endoftext|>`` only to the *last*
  message in the conversation so the trained model learns when to stop;
  historical assistant turns still match the Jinja byte-for-byte without
  any EOS adjustment.
- Generation suffix (``add_generation_prompt=True`` in Jinja):
  ``<|assistant|><think>`` for thinking mode (default),
  ``<|assistant|></think>`` for non-thinking mode.

Only text-only chat, with and without thinking, is implemented here.
Tool calls and multimodal content are left for a future extension.
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
        # GLM-5.1 role tags do not have a trailing newline; content is
        # concatenated directly (e.g. ``<|user|>hello``).
        if role == "user":
            header_str = "<|user|>"
            output_str = _visible_text(message["content"])
        elif role == "system":
            header_str = "<|system|>"
            output_str = _visible_text(message["content"])
        elif role == "tool":
            header_str = "<|observation|>"
            output_str = (
                f"<tool_response>{_visible_text(message['content'])}</tool_response>"
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
        # The role tag ``<|assistant|>`` is the header; the ``<think>...
        # </think>`` block lives in ``output`` so it is part of the training
        # target for this assistant turn.
        header_str = "<|assistant|>"

        is_historical = (
            self.strip_thinking_from_history
            and ctx.last_user_index >= 0
            and ctx.idx < ctx.last_user_index
        )

        reasoning, visible = _extract_reasoning_and_text(message["content"])

        # Match the shipped HF chat template:
        # - Historical turns (strip_thinking): ``</think>`` alone, no
        #   ``<think>`` opener.
        # - Terminal turn with reasoning: ``<think>{reasoning}</think>``.
        # - Terminal turn without reasoning: ``<think></think>`` (empty
        #   think block, thinking-mode default).
        #
        # No newlines between the role tag, the think block, or the content.
        if is_historical:
            think_block = "</think>"
        elif reasoning:
            think_block = f"<think>{reasoning.strip()}</think>"
        else:
            think_block = "<think></think>"

        visible_stripped = visible.strip()
        output_str = think_block + visible_stripped

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
        # For the assistant role, match the shipped template's
        # ``add_generation_prompt=True`` thinking-mode output:
        # ``<|assistant|><think>``. The model produces the rest of the think
        # block + ``</think>`` + visible content itself.
        if role == "assistant":
            suffix_str = "<|assistant|><think>"
        else:
            suffix_str = f"<|{role}|>"
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
