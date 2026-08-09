"""Local Kimi K2.5 renderer that declares its synthesized system message.

``kimi_k25`` deliberately keeps upstream's rendering, including the implicit
default system message that the corrected ``kimi_k25_interleaved`` variant
removes, because persisted legacy jobs depend on those exact tokens. Upstream
synthesizes that message inside ``build_supervised_example``, long after the
dataset's per-message weights were resolved, so it reaches the base renderer
with no ``trainable`` field and fails the ``CUSTOMIZED`` contract outright.

That is not a corner: ``resolve_renderer_snapshot`` returns this name for both
Kimi-K2.5 and Kimi-K2.6 whenever a job sets no explicit history mode, so any
SFT row carrying ``weight`` on those models failed to render.

Subclass to flag the synthesized message untrained. Rendering is unchanged —
the helper is a no-op for rows that carry no weights.
"""

from __future__ import annotations

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import Message, register_renderer
from tinker_cookbook.renderers.kimi_k25 import KimiK25Renderer as _TinkerKimiK25Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer

from training.renderer.message_weights import untrained_synthesized_context


class KimiK25SplitRenderer(_TinkerKimiK25Renderer):
    """Upstream K2.5 rendering, with its synthesized system message declared."""

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        return untrained_synthesized_context(super()._ensure_system_message(messages))


def _kimi_k25_factory(
    tokenizer: Tokenizer,
    image_processor: ImageProcessor | None = None,
) -> KimiK25SplitRenderer:
    return KimiK25SplitRenderer(tokenizer, image_processor=image_processor)


register_renderer("kimi_k25", _kimi_k25_factory)
