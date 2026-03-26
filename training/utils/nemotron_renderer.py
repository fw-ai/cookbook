"""Renderer for NVIDIA Nemotron-H models.

Nemotron-H uses the same ``<|im_start|>``/``<|im_end|>`` token format as Qwen3
but with different thinking-tag handling:

- Every assistant message gets ``<think></think>`` prepended if no thinking
  tags are present (even when thinking is "disabled" — the empty tags signal
  the model to skip extended reasoning).
- History truncation replaces thinking *content* with ``<think></think>``
  rather than stripping the tags entirely (cf. Qwen3 which removes them).
- Truncation scope: only assistant messages *before* the last user message
  are truncated (not all non-last messages as in Qwen3).
"""

from __future__ import annotations

import json

import tinker
import torch
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Role,
    TrainOnWhat,
)
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


def _ensure_thinking_tags(content: str) -> str:
    """Prepend ``<think></think>`` if neither tag is present."""
    if "<think>" not in content and "</think>" not in content:
        return f"<think></think>{content}"
    return content


def _format_nemotron_tool_call(tool_call) -> str:
    """Format a single tool call in Nemotron's XML parameter format.

    Produces::

        <tool_call>
        <function=get_weather>
        <parameter=city>
        San Francisco
        </parameter>
        </function>
        </tool_call>
    """
    func = tool_call.function
    arguments = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
    parts = [f"<tool_call>\n<function={func.name}>\n"]
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            formatted = json.dumps(value)
        else:
            formatted = str(value)
        parts.append(f"<parameter={key}>\n{formatted}\n</parameter>\n")
    parts.append("</function>\n</tool_call>\n")
    return "".join(parts)


def _truncate_thinking(content: str) -> str:
    """Replace thinking content with empty tags for historical messages.

    Mirrors the Jinja template logic: first ensure tags exist, then
    take only the text after the last ``</think>`` and re-prepend
    empty tags.
    """
    content = _ensure_thinking_tags(content)
    if "<think>" in content and "</think>" in content:
        after_think = content.split("</think>")[-1]
        return f"<think></think>{after_think}".strip()
    return content.strip()


class NemotronRenderer(Qwen3Renderer):
    """Renderer for NVIDIA Nemotron-H instruct models with thinking support.

    Format::

        <|im_start|>system
        You are a helpful assistant<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        <think>reasoning</think>Hi there!<|im_end|>

    Inherits tokenizer handling, stop sequences, and ``parse_response`` from
    :class:`Qwen3Renderer`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = True,
    ):
        super().__init__(
            tokenizer,
            strip_thinking_from_history=strip_thinking_from_history,
        )

    @staticmethod
    def _ensure_system_message(messages: list[Message]) -> list[Message]:
        """Inject an empty system message if the conversation doesn't start with one.

        Nemotron's Jinja template always emits a system block; omitting it
        causes a token mismatch between training and inference.
        """
        if not messages or messages[0]["role"] != "system":
            return [Message(role="system", content="")] + list(messages)
        return messages

    @staticmethod
    def _tag_tool_groups(messages: list[Message]) -> list[Message]:
        """Tag consecutive ``role=tool`` messages with their group position.

        Nemotron's Jinja template renders consecutive tool responses under one
        ``<|im_start|>user`` block.  Each tool message gets a ``_tool_pos`` key:
        ``"only"`` (single), ``"first"``, ``"middle"``, or ``"last"``.
        """
        tagged = list(messages)
        i = 0
        while i < len(tagged):
            if tagged[i]["role"] == "tool":
                start = i
                while i < len(tagged) and tagged[i]["role"] == "tool":
                    i += 1
                end = i - 1
                for j in range(start, end + 1):
                    if start == end:
                        pos = "only"
                    elif j == start:
                        pos = "first"
                    elif j == end:
                        pos = "last"
                    else:
                        pos = "middle"
                    tagged[j] = {**tagged[j], "_tool_pos": pos}
            else:
                i += 1
        return tagged

    def _preprocess_messages(self, messages: list[Message]) -> list[Message]:
        messages = self._ensure_system_message(messages)
        return self._tag_tool_groups(messages)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(
            self._preprocess_messages(messages), role=role, prefill=prefill,
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(
            self._preprocess_messages(messages), train_on_what=train_on_what,
        )

    def _render_tool_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a tool response, grouping consecutive tool messages.

        The Nemotron template emits one ``<|im_start|>user`` header for the
        entire group and one ``<|im_end|>`` at the end.  The ``_tool_pos``
        tag (set by ``_tag_tool_groups``) controls header/footer emission.
        """
        pos = message.get("_tool_pos", "only")
        maybe_newline = "\n" if ctx.idx > 0 else ""

        if pos in ("first", "only"):
            header_str = f"{maybe_newline}<|im_start|>user\n"
        else:
            header_str = ""

        body = f"<tool_response>\n{message['content']}\n</tool_response>\n"
        if pos in ("last", "only"):
            body += "<|im_end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False) if header_str else [],
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(body, add_special_tokens=False),
            ),
        ]
        return RenderedMessage(header=header, output=output)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        if message["role"] == "tool":
            return self._render_tool_message(message, ctx)
        if message["role"] != "assistant":
            return super().render_message(message, ctx)

        maybe_newline = "\n" if ctx.idx > 0 else ""
        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        content = message["content"]
        should_truncate = (
            self.strip_thinking_from_history
            and ctx.last_user_index >= 0
            and ctx.idx < ctx.last_user_index
        )

        if isinstance(content, list):
            has_thinking = any(p.get("type") == "thinking" for p in content)
            rendered_parts: list[str] = []

            if should_truncate:
                rendered_parts.append("<think></think>")
                for p in content:
                    if p["type"] == "text":
                        rendered_parts.append(p["text"])
            else:
                if not has_thinking:
                    rendered_parts.append("<think></think>")
                for p in content:
                    if p["type"] == "thinking":
                        rendered_parts.append(f"<think>{p['thinking']}</think>")
                    elif p["type"] == "text":
                        rendered_parts.append(p["text"])

            output_content = "".join(rendered_parts)
        else:
            if should_truncate:
                output_content = _truncate_thinking(content)
            else:
                output_content = _ensure_thinking_tags(content)

        if "tool_calls" in message:
            output_content += "\n" + "".join(
                _format_nemotron_tool_call(tc)
                for tc in message["tool_calls"]
            )

        output_content += "<|im_end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False),
            ),
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: str, ctx: RenderContext) -> list[int]:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        suffix_str = f"{maybe_newline}<|im_start|>{role}\n<think>\n"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)
