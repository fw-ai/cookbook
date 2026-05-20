"""Cookbook-local model renderers.

This package mirrors the layout of ``tinker_cookbook.renderers`` but holds
renderers that have not yet been upstreamed. Each module exposes a
``Renderer`` subclass and registers it under a short name via
``tinker_cookbook.renderers.register_renderer`` so it can be obtained with
``tinker_cookbook.renderers.get_renderer(name, tokenizer)``.

Importing this package eagerly imports every contained renderer module so
the registrations take effect.

The ``_*_split`` modules re-register upstream renderer names with local
subclasses that mix in ``DisaggregateMultiTurnMixin``, so multi-turn
ALL_ASSISTANT_MESSAGES SFT works for upstream renderers whose chat
templates strip historical thinking but ship without a
``build_supervised_examples`` override. They run last so the override
shadows the upstream registration.
"""

from training.renderer import deepseek_v4 as _deepseek_v4  # noqa: F401  (registers "deepseek_v4")
from training.renderer import gemma4 as _gemma4  # noqa: F401  (registers "gemma4")
from training.renderer import glm5 as _glm5  # noqa: F401  (registers "glm5")
from training.renderer import minimax_m2 as _minimax_m2  # noqa: F401  (registers "minimax_m2")
from training.renderer import nemotron as _nemotron  # noqa: F401  (registers "nemotron")

# Local overrides for upstream renderers that need disaggregate-per-user-turn
# multi-turn SFT support (no upstream ``build_supervised_examples`` override).
from training.renderer import _qwen3_split as _qwen3_split  # noqa: F401
from training.renderer import _deepseek_v3_split as _deepseek_v3_split  # noqa: F401
from training.renderer import _nemotron3_split as _nemotron3_split  # noqa: F401
from training.renderer import _gpt_oss_split as _gpt_oss_split  # noqa: F401
