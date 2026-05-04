"""Local Qwen3 / Qwen3.5 renderers with multi-turn SFT disaggregate support.

Upstream ``tinker_cookbook.renderers.qwen3`` and
``tinker_cookbook.renderers.qwen3_5`` ship without a
``build_supervised_examples`` override. Combined with
``has_extension_property=False`` (the default for thinking-mode
variants), the cookbook SFT dispatcher routes multi-turn
``ALL_ASSISTANT_MESSAGES`` rendering to the upstream base implementation
which raises ``NotImplementedError``.

This module re-registers the upstream renderer names with local
subclasses that mix in ``DisaggregateMultiTurnMixin``. Importing this
module (eagerly via ``training/renderer/__init__.py``) installs the
override before any caller resolves a renderer via ``get_renderer``.
"""

from __future__ import annotations

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.qwen3 import (
    Qwen3DisableThinkingRenderer,
    Qwen3Renderer,
    Qwen3VLInstructRenderer,
    Qwen3VLRenderer,
)
from tinker_cookbook.renderers.qwen3_5 import (
    Qwen3_5DisableThinkingRenderer,
    Qwen3_5Renderer,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin


class Qwen3SplitRenderer(DisaggregateMultiTurnMixin, Qwen3Renderer):
    pass


class Qwen3DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin, Qwen3DisableThinkingRenderer
):
    pass


class Qwen3VLSplitRenderer(DisaggregateMultiTurnMixin, Qwen3VLRenderer):
    pass


class Qwen3VLInstructSplitRenderer(
    DisaggregateMultiTurnMixin, Qwen3VLInstructRenderer
):
    pass


class Qwen3_5SplitRenderer(DisaggregateMultiTurnMixin, Qwen3_5Renderer):
    pass


class Qwen3_5DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin, Qwen3_5DisableThinkingRenderer
):
    pass


register_renderer("qwen3", lambda tok, ip=None: Qwen3SplitRenderer(tok))
register_renderer(
    "qwen3_disable_thinking",
    lambda tok, ip=None: Qwen3DisableThinkingSplitRenderer(tok),
)
register_renderer(
    "qwen3_vl",
    lambda tok, ip=None: Qwen3VLSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_vl_instruct",
    lambda tok, ip=None: Qwen3VLInstructSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_5",
    lambda tok, ip=None: Qwen3_5SplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_5_disable_thinking",
    lambda tok, ip=None: Qwen3_5DisableThinkingSplitRenderer(
        tok, image_processor=ip
    ),
)
