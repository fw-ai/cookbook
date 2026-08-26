"""Local Nemotron3 renderers with multi-turn SFT disaggregate support.

Upstream ``tinker_cookbook.renderers.nemotron3`` ships ``Nemotron3Renderer``
(and ``Nemotron3DisableThinkingRenderer``) which inherit Qwen3_5Renderer's
strip-from-history behavior but don't carry a ``build_supervised_examples``
override. Re-register both upstream names with local subclasses that mix in
``DisaggregateMultiTurnMixin``.

Nemotron-3's HF template always emits a system block, so upstream synthesizes
an empty system message when the row has none. That message is template
context, and under per-message weighting it has to say so.

Preserve-thinking mode (Nano / Super)
-------------------------------------
The served Nemotron-3 Nano / Super chat templates gate historical
reasoning on ``truncate_history_thinking`` (default ``True`` = strip prior
turns). Serving maps ``reasoning_history="preserved"`` →
``truncate_history_thinking=False`` (#40028). This module registers
``nemotron3_preserve_thinking`` so SFT can match that contract: historical
``reasoning_content`` is replayed inside ``<think>`` and the renderer
satisfies the sequence extension property (no per-user-turn unrolling).

Think-block boundary
--------------------
Nemotron-3 inherits Qwen3.5's assistant split, which puts the prefilled
``<think>`` in the trainable span and the model-generated ``</think>`` in the
masked header. Every thinking-enabled variant therefore mixes in
:class:`training.renderer._think_prefill.ThinkPrefillWeightsMixin`.
``nemotron3_disable_thinking`` does not: its generation suffix prefills the
whole ``<think></think>`` wrapper, so masking both markers is already right.
"""

from __future__ import annotations

from tinker_cookbook.renderers import Message, register_renderer
from tinker_cookbook.renderers.base import RenderContext
from tinker_cookbook.renderers.nemotron3 import (
    Nemotron3DisableThinkingRenderer,
    Nemotron3LowThinkingRenderer,
    Nemotron3Renderer,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer._think_prefill import ThinkPrefillWeightsMixin
from training.renderer.message_weights import untrained_synthesized_context


class _UntrainedSynthesizedSystemMixin:
    """Declare Nemotron-3's synthesized empty system message untrained."""

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        return untrained_synthesized_context(super()._normalize_messages(messages))


class Nemotron3SplitRenderer(
    ThinkPrefillWeightsMixin,
    DisaggregateMultiTurnMixin,
    _UntrainedSynthesizedSystemMixin,
    Nemotron3Renderer,
):
    """Default / INTERLEAVED: strip thinking before the last user turn."""


class Nemotron3LowThinkingSplitRenderer(
    ThinkPrefillWeightsMixin,
    DisaggregateMultiTurnMixin,
    _UntrainedSynthesizedSystemMixin,
    Nemotron3LowThinkingRenderer,
):
    """Low-effort reasoning (Super only); thinking is still enabled."""


class Nemotron3DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin,
    _UntrainedSynthesizedSystemMixin,
    Nemotron3DisableThinkingRenderer,
):
    pass


class _Nemotron3PreserveThinkingMixin:
    """Match HF ``truncate_history_thinking=False``.

    Upstream ``Nemotron3Renderer`` always strips thinking for
    ``idx < last_user_index`` (interleaved default). Preserve mode keeps
    every assistant ``<think>`` block — including historical turns that
    precede a later user message — and therefore satisfies the extension
    property (``has_extension_property=True`` via
    ``strip_thinking_from_history=False``).
    """

    def __init__(self, tokenizer, image_processor=None):
        super().__init__(
            tokenizer,
            image_processor=image_processor,
            strip_thinking_from_history=False,
        )

    def _assistant_header_suffix(self, message: Message, ctx: RenderContext) -> str:
        """Do not emit empty ``<think></think>`` when thinking will be kept.

        With ``truncate_history_thinking=False`` the HF template includes the
        full think block for every assistant turn. Empty wrappers are only
        prepended when the assistant has no thinking content at all — the
        same rule as non-historical turns under the interleaved renderer.
        """
        content = message.get("content", "")
        has_think = False
        if isinstance(content, list):
            has_think = any(p["type"] == "thinking" for p in content)
        elif isinstance(content, str):
            has_think = "<think>" in content
        if has_think:
            return ""
        return "<think></think>"


class Nemotron3PreserveThinkingSplitRenderer(
    _Nemotron3PreserveThinkingMixin,
    ThinkPrefillWeightsMixin,
    DisaggregateMultiTurnMixin,
    _UntrainedSynthesizedSystemMixin,
    Nemotron3Renderer,
):
    """PRESERVED history: replay prior reasoning inside ``<think>``."""


register_renderer("nemotron3", lambda tok, ip=None: Nemotron3SplitRenderer(tok))
register_renderer(
    "nemotron3_interleaved",
    lambda tok, ip=None: Nemotron3SplitRenderer(tok),
)
register_renderer(
    "nemotron3_low_thinking",
    lambda tok, ip=None: Nemotron3LowThinkingSplitRenderer(tok),
)
register_renderer(
    "nemotron3_disable_thinking",
    lambda tok, ip=None: Nemotron3DisableThinkingSplitRenderer(tok),
)
register_renderer(
    "nemotron3_preserve_thinking",
    lambda tok, ip=None: Nemotron3PreserveThinkingSplitRenderer(tok),
)
register_renderer(
    "nemotron3_preserved",
    lambda tok, ip=None: Nemotron3PreserveThinkingSplitRenderer(tok),
)
