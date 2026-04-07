"""Cookbook-local model renderers.

This package mirrors the layout of ``tinker_cookbook.renderers`` but holds
renderers that have not yet been upstreamed. Each module exposes a
``Renderer`` subclass and registers it under a short name via
``tinker_cookbook.renderers.register_renderer`` so it can be obtained with
``tinker_cookbook.renderers.get_renderer(name, tokenizer)``.

Importing this package eagerly imports every contained renderer module so
the registrations take effect.
"""

from training.renderer import minimax_m2 as _minimax_m2  # noqa: F401  (registers "minimax_m2")
from training.renderer import nemotron as _nemotron  # noqa: F401  (registers "nemotron")
