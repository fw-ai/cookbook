"""Renderer registration for Moonshot Kimi K2.7 Code.

Kimi K2.7 Code keeps the K2.6 tokenizer and special tokens, but its official
chat template differs in three important ways:

* historical thinking is preserved by default and cannot be disabled by the
  tokenizer's ``apply_chat_template`` wrapper;
* no default system message is injected when the input starts with a user turn.
* different tokenizer backends may either auto-populate the template-only
  ``tools_ts_str`` variable or fall back to compact OpenAI JSON in
  ``tool_declare``.

This local renderer reuses the upstream K2.6 preserve-thinking implementation
and overrides those K2.7-specific pieces so tokenization matches the official
K2.7 Code template.
"""

from __future__ import annotations

import json
from typing import Any

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import Message, ToolSpec
from tinker_cookbook.renderers.kimi_k2_5_tool_declaration_ts import (
    encode_tools_to_typescript_style,
)
from tinker_cookbook.renderers.kimi_k26 import (
    KimiK26PreserveThinkingRenderer as _UpstreamKimiK26PreserveThinkingRenderer,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

from training.renderer.kimi_k26 import (
    KimiK26PreserveThinkingRenderer as _CookbookKimiK26PreserveThinkingRenderer,
)


_KIMI_TOOL_STYLE_PROBE = [
    {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def _tokenizer_tools_branch_uses_typescript(tokenizer: Any) -> bool:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return False

    try:
        rendered = apply_chat_template(
            [{"role": "user", "content": "probe"}],
            tools=_KIMI_TOOL_STYLE_PROBE,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return False

    return isinstance(rendered, str) and "namespace functions" in rendered


class _KimiK27CodeMixin:
    """K2.7-specific system/tool declaration behavior."""

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
            if _tokenizer_tools_branch_uses_typescript(self.tokenizer):
                content = encode_tools_to_typescript_style(tools_payload)
            else:
                content = json.dumps(
                    tools_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            messages.append(Message(role="tool_declare", content=content))

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        return messages


class KimiK27CodeRenderer(
    _KimiK27CodeMixin,
    _UpstreamKimiK26PreserveThinkingRenderer,
):
    """Legacy concrete ``kimi_k27_code`` behavior."""


class KimiK27CodePreservedRenderer(
    _KimiK27CodeMixin,
    _CookbookKimiK26PreserveThinkingRenderer,
):
    """K2.7's corrected PRESERVED mode, without unrolling."""


def _kimi_k27_code_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK27CodeRenderer:
    return KimiK27CodeRenderer(tokenizer, image_processor=image_processor)


def _kimi_k27_code_preserved_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK27CodePreservedRenderer:
    return KimiK27CodePreservedRenderer(
        tokenizer,
        image_processor=image_processor,
    )


register_renderer("kimi_k27_code", _kimi_k27_code_factory)
register_renderer("kimi_k27_code_preserved", _kimi_k27_code_preserved_factory)
