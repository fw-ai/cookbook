"""Local gpt-oss renderer override: declare ``has_extension_property=True``.

Unlike Qwen3 / GLM5 / DeepSeek-thinking renderers (which strip historical
thinking from rendered tokens), the upstream
``tinker_cookbook.renderers.gpt_oss.GptOssRenderer.render_message`` emits
the analysis channel for *any* assistant message that carries thinking
content, with no ``ctx.is_last`` / ``before_last_user`` guard. Its
rendered output therefore *does* satisfy the sequence extension
property: each successive render is a strict prefix of the next.

The base ``Renderer`` class defaults ``has_extension_property=False``,
which would route multi-turn ``ALL_ASSISTANT_MESSAGES`` SFT to the
plural dispatcher path; the upstream class also lacks a
``build_supervised_examples`` override, so the dispatch ends up at
``base.py:1515 raise NotImplementedError``.

Disaggregating per-user-turn would not help here — every per-prefix
datum would still preserve historical thinking (the renderer's
behavior, by-design from this class's perspective), just N times the
token cost. The correct dispatcher decision is to fast-path to
singular, which requires declaring ``has_extension_property=True``.

NOTE: gpt-oss's cookbook renderer disagrees with the shipped HF
``apply_chat_template`` — HF Jinja strips historical analysis via a
``future_final_message`` lookahead. The cookbook renderer does not.
That divergence is a real training/inference OOD bug, but it lives in
``render_message`` and is independent of the dispatcher routing
addressed here. Tracked separately.
"""

from __future__ import annotations

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer


class GptOssExtensionRenderer(GptOssRenderer):
    """gpt-oss subclass declaring extension-safe behavior.

    The upstream renderer's ``render_message`` does not strip
    historical thinking, so token sequences are extension-safe by
    construction. This override aligns ``has_extension_property`` with
    the actual rendering behavior so the SFT dispatcher
    fast-paths to the singular render path instead of trying to
    disaggregate (which would be wasteful and wouldn't fix the
    independent HF parity issue).
    """

    @property
    def has_extension_property(self) -> bool:
        return True


register_renderer(
    "gpt_oss_no_sysprompt",
    lambda tok, ip=None: GptOssExtensionRenderer(tok, use_system_prompt=False),
)
register_renderer(
    "gpt_oss_low_reasoning",
    lambda tok, ip=None: GptOssExtensionRenderer(
        tok, use_system_prompt=True, reasoning_effort="low"
    ),
)
register_renderer(
    "gpt_oss_medium_reasoning",
    lambda tok, ip=None: GptOssExtensionRenderer(
        tok, use_system_prompt=True, reasoning_effort="medium"
    ),
)
register_renderer(
    "gpt_oss_high_reasoning",
    lambda tok, ip=None: GptOssExtensionRenderer(
        tok, use_system_prompt=True, reasoning_effort="high"
    ),
)
