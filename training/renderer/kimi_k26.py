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

from training.renderer.image_processing import ImageProcessor
from training.renderer import register_renderer
from training._vendor.tinker_cookbook_0_4_3.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    ToolSpec,
)
from training._vendor.tinker_cookbook_0_4_3.renderers.kimi_k25 import (
    KimiK25Renderer as _TinkerKimiK25Renderer,
)
from training._vendor.tinker_cookbook_0_4_3.renderers.kimi_k26 import (
    KimiK26PreserveThinkingRenderer as _TinkerKimiK26PreserveThinkingRenderer,
)
from training.renderer.tokenizer import Tokenizer

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
                    if not (isinstance(part, dict) and part.get("type") == "thinking")
                ]
            else:
                visible_parts = [{"type": "text", "text": content}]
            copied["content"] = [
                {"type": "thinking", "thinking": reasoning},
                *visible_parts,
            ]
            message = copied  # type: ignore[assignment]
        return super().render_message(message, ctx)  # type: ignore[misc]


# Distinguishes "not resolved yet" from a resolved ``None``.
_UNRESOLVED = object()


class _KimiMediaPadImagePlaceholderMixin:
    """Resolve the image-placeholder token id for token-in vision completions.

    The K2.5/K2.6 renderer lineage is text-only, so nothing up the MRO
    supplies this. Rollout multimodal rendering needs it to encode image
    chunks for token-in completions, and without it a vision-capable
    checkpoint samples zero multimodal prompt groups and RL fails the "no
    trained multimodal prompt group" gate. Resolve it the way
    ``KimiK3VisionRenderer`` does -- from the tokenizer's ``<|media_pad|>``
    special token -- and stay ``None`` for tokenizers that lack it rather
    than returning an ``unk`` id that would silently render as text.
    """

    @property
    def image_placeholder_token_id(self) -> int | None:
        """Token id standing in for one image chunk, or ``None`` when text-only."""
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


class KimiK25InterleavedRenderer(
    _KimiMediaPadImagePlaceholderMixin,
    _KimiReasoningFieldPrecedenceMixin,
    DisaggregateMultiTurnMixin,
    _NoImplicitSystemMessageMixin,
    _TinkerKimiK25Renderer,
):
    """K2.5's only history mode: INTERLEAVED.

    Tinker's own unrolling splits on the same per-user-turn prefixes, but hands
    each prefix the caller's mode verbatim. A row carrying per-message weights
    reaches it as ``CUSTOMIZED``, which then re-trains every earlier assistant
    turn once per later prefix — with its thinking already stripped from
    history, the train/inference mismatch the unrolling exists to prevent. The
    mixin reduces each prefix to its own terminal turn instead.
    """


class KimiK26InterleavedRenderer(KimiK25InterleavedRenderer):
    """K2.6 INTERLEAVED mode, equivalent to ``preserve_thinking=False``."""


class KimiK26PreserveThinkingRenderer(
    _KimiMediaPadImagePlaceholderMixin,
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
