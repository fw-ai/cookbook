"""Local Nemotron3 renderers with multi-turn SFT disaggregate support.

Upstream ``tinker_cookbook.renderers.nemotron3`` ships ``Nemotron3Renderer``
(and ``Nemotron3DisableThinkingRenderer``) which inherit Qwen3_5Renderer's
strip-from-history behavior but don't carry a ``build_supervised_examples``
override. Re-register both upstream names with local subclasses that mix in
``DisaggregateMultiTurnMixin``.

Nemotron-3's HF template always emits a system block, so upstream synthesizes
an empty system message when the row has none. That message is template
context, and under per-message weighting it has to say so.
"""

from __future__ import annotations

from tinker_cookbook.renderers import Message, register_renderer
from tinker_cookbook.renderers.nemotron3 import (
    Nemotron3DisableThinkingRenderer,
    Nemotron3Renderer,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.message_weights import untrained_synthesized_context


class _UntrainedSynthesizedSystemMixin:
    """Declare Nemotron-3's synthesized empty system message untrained."""

    def _normalize_messages(self, messages: list[Message]) -> list[Message]:
        return untrained_synthesized_context(super()._normalize_messages(messages))


class Nemotron3SplitRenderer(
    DisaggregateMultiTurnMixin,
    _UntrainedSynthesizedSystemMixin,
    Nemotron3Renderer,
):
    pass


class Nemotron3DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin,
    _UntrainedSynthesizedSystemMixin,
    Nemotron3DisableThinkingRenderer,
):
    pass


register_renderer("nemotron3", lambda tok, ip=None: Nemotron3SplitRenderer(tok))
register_renderer(
    "nemotron3_disable_thinking",
    lambda tok, ip=None: Nemotron3DisableThinkingSplitRenderer(tok),
)
