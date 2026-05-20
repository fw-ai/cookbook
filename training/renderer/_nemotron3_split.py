"""Local Nemotron3 renderers with multi-turn SFT disaggregate support.

Upstream ``tinker_cookbook.renderers.nemotron3`` ships ``Nemotron3Renderer``
(and ``Nemotron3DisableThinkingRenderer``) which inherit Qwen3_5Renderer's
strip-from-history behavior but don't carry a ``build_supervised_examples``
override. Re-register both upstream names with local subclasses that mix in
``DisaggregateMultiTurnMixin``.
"""

from __future__ import annotations

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.nemotron3 import (
    Nemotron3DisableThinkingRenderer,
    Nemotron3Renderer,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin


class Nemotron3SplitRenderer(DisaggregateMultiTurnMixin, Nemotron3Renderer):
    pass


class Nemotron3DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin, Nemotron3DisableThinkingRenderer
):
    pass


register_renderer("nemotron3", lambda tok, ip=None: Nemotron3SplitRenderer(tok))
register_renderer(
    "nemotron3_disable_thinking",
    lambda tok, ip=None: Nemotron3DisableThinkingSplitRenderer(tok),
)
