"""Local gpt-oss renderer with multi-turn SFT disaggregate support.

Upstream ``tinker_cookbook.renderers.gpt_oss.GptOssRenderer`` ships without
a ``build_supervised_examples`` override and ``has_extension_property``
defaults to False (base class default). Multi-turn ALL_ASSISTANT_MESSAGES
SFT therefore raises ``NotImplementedError``. Re-register all four
upstream gpt-oss reasoning-effort variants with a local subclass that
mixes in ``DisaggregateMultiTurnMixin``.
"""

from __future__ import annotations

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin


class GptOssSplitRenderer(DisaggregateMultiTurnMixin, GptOssRenderer):
    pass


register_renderer(
    "gpt_oss_no_sysprompt",
    lambda tok, ip=None: GptOssSplitRenderer(tok, use_system_prompt=False),
)
register_renderer(
    "gpt_oss_low_reasoning",
    lambda tok, ip=None: GptOssSplitRenderer(
        tok, use_system_prompt=True, reasoning_effort="low"
    ),
)
register_renderer(
    "gpt_oss_medium_reasoning",
    lambda tok, ip=None: GptOssSplitRenderer(
        tok, use_system_prompt=True, reasoning_effort="medium"
    ),
)
register_renderer(
    "gpt_oss_high_reasoning",
    lambda tok, ip=None: GptOssSplitRenderer(
        tok, use_system_prompt=True, reasoning_effort="high"
    ),
)
