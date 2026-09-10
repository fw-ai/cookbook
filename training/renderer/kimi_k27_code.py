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

from training.renderer.image_processing import ImageProcessor
from training.renderer import register_renderer
from training._vendor.tinker_cookbook_0_4_3.renderers.base import Message, ToolSpec
from training._vendor.tinker_cookbook_0_4_3.renderers.kimi_k2_5_tool_declaration_ts import (
    encode_tools_to_typescript_style,
)
from training._vendor.tinker_cookbook_0_4_3.renderers.kimi_k26 import (
    KimiK26PreserveThinkingRenderer as _UpstreamKimiK26PreserveThinkingRenderer,
)
from training.renderer.tokenizer import Tokenizer

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
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


# Distinguishes "not resolved yet" from a resolved ``None``.
_UNRESOLVED = object()


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

    @property
    def image_placeholder_token_id(self) -> int | None:
        """Token id standing in for one image chunk, or ``None`` when text-only.

        K2.7 inherits a text-only renderer lineage from K2.6, so nothing up the
        MRO resolves this. Rollout multimodal rendering needs it to encode image
        chunks for token-in completions, and without it a vision-capable K2.7
        checkpoint samples zero multimodal prompt groups. Resolve it the way
        ``KimiK3VisionRenderer`` does -- from the tokenizer's ``<|media_pad|>``
        special token -- and stay ``None`` for tokenizers that lack it rather
        than returning an ``unk`` id that would silently render as text.
        """
        cached = getattr(self, "_image_placeholder_token_id_cache", _UNRESOLVED)
        if cached is not _UNRESOLVED:
            return cached

        # Imported lazily: ``renderer/__init__`` loads this module before
        # ``kimi_k3``, so a module-level import would close a cycle.
        from training.renderer.kimi_k3 import MEDIA_PAD_TOKEN

        resolved: int | None = None
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            try:
                candidate = convert(MEDIA_PAD_TOKEN)
            except (KeyError, ValueError):
                candidate = None
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                # A tokenizer without the token maps it to ``unk`` rather than
                # failing, and encoding that id would render the image chunk as
                # ordinary text instead of a placeholder.
                unk_id = getattr(self.tokenizer, "unk_token_id", None)
                if not (isinstance(unk_id, int) and candidate == unk_id):
                    resolved = candidate

        self._image_placeholder_token_id_cache = resolved
        return resolved

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
    DisaggregateMultiTurnMixin,
    _UpstreamKimiK26PreserveThinkingRenderer,
):
    """Legacy concrete ``kimi_k27_code`` behavior.

    The upstream base reports no extension property and unrolls multi-turn rows
    with its own splitter, which hands each per-user-turn prefix the caller's
    mode verbatim. A row carrying per-message weights therefore arrived as
    ``CUSTOMIZED`` and re-trained every earlier assistant turn in every later
    prefix. The mixin reduces each prefix to its own terminal turn instead, and
    is byte-identical to the upstream splitter for rows without weights.
    """


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
