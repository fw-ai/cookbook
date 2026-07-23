"""Cookbook-local Kimi K2.5 and K2.6 history-mode renderers.

Moonshot's K2.5 and K2.6 templates do not synthesize a system message.  The
upstream Tinker K2.5 renderer does, so both local INTERLEAVED renderers
override that behavior while retaining Tinker's SFT-only terminal-target
adaptation and per-user-turn unrolling.

K2.6 additionally exposes the official ``preserve_thinking=True`` branch.
Tinker represents that branch as ``strip_thinking_from_history=False``; this
wrapper also advertises the resulting extension property and consequently
keeps a multi-turn conversation in one supervised datum.
"""

from __future__ import annotations

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    ToolSpec,
)
from tinker_cookbook.renderers.kimi_k25 import (
    KimiK25Renderer as _TinkerKimiK25Renderer,
)
from tinker_cookbook.renderers.kimi_k26 import (
    KimiK26PreserveThinkingRenderer as _TinkerKimiK26PreserveThinkingRenderer,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.reasoning_fields import original_reasoning


class _NoImplicitSystemMessageMixin:
    """Match the official K2.5+ templates: render only supplied messages."""

    # The generic tool-prefix assembler normally folds a leading system
    # message into ``system_prompt``. An explicit empty system message would
    # otherwise become indistinguishable from no system message at all.
    _preserves_explicit_empty_system_with_tools = True

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        return list(messages)

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        prefix = super().create_conversation_prefix_with_tools(  # type: ignore[misc]
            tools,
            system_prompt=system_prompt,
        )
        if system_prompt:
            return prefix
        return [message for message in prefix if message["role"] != "system"]


class _KimiReasoningFieldPrecedenceMixin:
    """Apply Kimi's field-presence precedence after generic normalization."""

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        has_reasoning, reasoning = original_reasoning(message)
        if has_reasoning and message["role"] == "assistant":
            copied = dict(message)
            content = copied.get("content", "")
            if isinstance(content, list):
                visible_parts = [
                    part
                    for part in content
                    if not (
                        isinstance(part, dict) and part.get("type") == "thinking"
                    )
                ]
            else:
                visible_parts = [{"type": "text", "text": content}]
            copied["content"] = [
                {"type": "thinking", "thinking": reasoning},
                *visible_parts,
            ]
            message = copied  # type: ignore[assignment]
        return super().render_message(message, ctx)  # type: ignore[misc]


class KimiK25InterleavedRenderer(
    _KimiReasoningFieldPrecedenceMixin,
    _NoImplicitSystemMessageMixin,
    _TinkerKimiK25Renderer,
):
    """K2.5's only history mode: INTERLEAVED."""


class KimiK26InterleavedRenderer(KimiK25InterleavedRenderer):
    """K2.6 INTERLEAVED mode, equivalent to ``preserve_thinking=False``."""


class KimiK26PreserveThinkingRenderer(
    _KimiReasoningFieldPrecedenceMixin,
    DisaggregateMultiTurnMixin,
    _NoImplicitSystemMessageMixin,
    _TinkerKimiK26PreserveThinkingRenderer,
):
    """K2.6 PRESERVED mode with no multi-turn unrolling."""

    @property
    def has_extension_property(self) -> bool:
        return True


def _kimi_k25_interleaved_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK25InterleavedRenderer:
    return KimiK25InterleavedRenderer(tokenizer, image_processor=image_processor)


def _kimi_k26_interleaved_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK26InterleavedRenderer:
    return KimiK26InterleavedRenderer(tokenizer, image_processor=image_processor)


def _kimi_k26_preserve_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK26PreserveThinkingRenderer:
    return KimiK26PreserveThinkingRenderer(
        tokenizer,
        image_processor=image_processor,
    )


# ``kimi_k25`` is an existing upstream concrete name. Do not shadow it: direct
# callers and persisted legacy jobs retain the upstream behavior (including its
# implicit system message). The corrected official-template variants use new,
# immutable names that Managed Training may safely materialize.
register_renderer("kimi_k25_interleaved", _kimi_k25_interleaved_factory)
register_renderer("kimi_k26_interleaved", _kimi_k26_interleaved_factory)
register_renderer("kimi_k26_preserve_thinking", _kimi_k26_preserve_factory)
