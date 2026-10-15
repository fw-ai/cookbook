"""Local DeepSeek V3 thinking renderer with multi-turn SFT disaggregate support.

Upstream ``training.renderer.deepseek_v3.DeepSeekV3ThinkingRenderer``
strips ``<think>`` blocks from history (matching the shipped
``apply_chat_template`` default) but ships without a
``build_supervised_examples`` override. Multi-turn ALL_ASSISTANT_MESSAGES
SFT therefore raises ``NotImplementedError``.

Re-register the upstream name (``deepseekv3_thinking``) with a local
subclass that mixes in ``DisaggregateMultiTurnMixin``. The non-thinking
variants (``deepseekv3``, ``deepseekv3_disable_thinking``) override
``has_extension_property=True`` upstream and don't need this fix.
"""

from __future__ import annotations

from training.renderer import register_renderer
from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.deepseek_v3 import DeepSeekV3ThinkingRenderer


class DeepSeekV3ThinkingSplitRenderer(DisaggregateMultiTurnMixin, DeepSeekV3ThinkingRenderer):
    pass


register_renderer(
    "deepseekv3_thinking",
    lambda tok, ip=None: DeepSeekV3ThinkingSplitRenderer(tok),
)
