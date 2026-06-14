"""Renderer registration for Moonshot Kimi K2.7 Code.

Kimi K2.7 Code keeps the K2.6 tokenizer, special tokens, and tool declaration
format, but its official chat template differs in two important ways:

* historical thinking is preserved by default and cannot be disabled by the
  tokenizer's ``apply_chat_template`` wrapper;
* no default system message is injected when the input starts with a user turn.

This local renderer reuses the upstream K2.6 preserve-thinking implementation
and only removes the default-system insertion so tokenization matches the
official K2.7 Code template.
"""

from __future__ import annotations

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import Message, ToolSpec
from tinker_cookbook.renderers.kimi_k2_5_tool_declaration_ts import (
    encode_tools_to_typescript_style,
)
from tinker_cookbook.renderers.kimi_k26 import KimiK26PreserveThinkingRenderer
from tinker_cookbook.tokenizer_utils import Tokenizer


class KimiK27CodeRenderer(KimiK26PreserveThinkingRenderer):
    """Kimi K2.7 Code renderer aligned with its HF chat template."""

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        return list(messages)

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        messages: list[Message] = []

        if tools:
            tools_payload = [{"type": "function", "function": tool} for tool in tools]
            tools_ts_str = encode_tools_to_typescript_style(tools_payload)
            messages.append(Message(role="tool_declare", content=tools_ts_str))

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        return messages


def _kimi_k27_code_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK27CodeRenderer:
    return KimiK27CodeRenderer(tokenizer, image_processor=image_processor)


register_renderer("kimi_k27_code", _kimi_k27_code_factory)
