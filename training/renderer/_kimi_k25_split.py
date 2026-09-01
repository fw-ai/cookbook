"""Local Kimi K2.5 renderer with weight-aware multi-turn SFT unrolling.

``kimi_k25`` deliberately keeps upstream's *rendering*, including the implicit
default system message that the corrected ``kimi_k25_interleaved`` variant
removes, because persisted legacy jobs depend on those exact tokens. Loss
placement is a separate matter, and upstream gets it wrong for weighted rows:
its own unrolling splits on per-user-turn prefixes but hands each prefix the
caller's mode verbatim, so a row carrying per-message weights arrives as
``CUSTOMIZED`` and every earlier assistant turn is re-trained in every later
prefix — with its thinking already stripped from history.

That is not a corner: ``resolve_renderer_snapshot`` returns this name for both
Kimi-K2.5 and Kimi-K2.6 whenever a job sets no explicit history mode, and the
row renders without error, so it mistrained silently.

``DisaggregateMultiTurnMixin`` reduces each prefix to its own terminal turn and
is byte-identical to upstream's splitter for rows that carry no weights, so the
legacy rendering contract holds. Upstream's synthesized system message
additionally has to declare itself untrained, or the ``CUSTOMIZED`` render fails
its contract outright.
"""

from __future__ import annotations

from training.renderer.image_processing import ImageProcessor
from training.renderer import Message, register_renderer
from training._vendor.tinker_cookbook_0_4_3.renderers.kimi_k25 import KimiK25Renderer as _TinkerKimiK25Renderer
from training.renderer.tokenizer import Tokenizer

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.message_weights import untrained_synthesized_context


class KimiK25SplitRenderer(DisaggregateMultiTurnMixin, _TinkerKimiK25Renderer):
    """Upstream K2.5 rendering with per-turn, weight-aware loss placement."""

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        return untrained_synthesized_context(super()._ensure_system_message(messages))


def _kimi_k25_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK25SplitRenderer:
    return KimiK25SplitRenderer(tokenizer, image_processor=image_processor)


register_renderer("kimi_k25", _kimi_k25_factory)
