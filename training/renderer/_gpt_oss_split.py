"""Local gpt-oss renderer with multi-turn SFT disaggregate support and a
history-thinking strip patch on ``render_message``.

There are two layered bugs in upstream
``tinker_cookbook.renderers.gpt_oss.GptOssRenderer``:

1. ``render_message`` (gpt_oss.py:265-268) emits the analysis channel
   for any assistant message that carries thinking content, with no
   ``ctx.is_last`` / ``before_last_user`` guard. HF
   ``apply_chat_template`` strips historical analysis via a
   ``future_final_message`` Jinja lookahead — verified empirically.
   So upstream's renderer disagrees with the shipped chat template:
   training tokens carry historical analysis that inference never
   shows the model. Silent training/inference OOD.

2. The base ``Renderer`` class defaults
   ``has_extension_property=False`` and provides no
   ``build_supervised_examples`` override, so multi-turn
   ``ALL_ASSISTANT_MESSAGES`` SFT crashes with
   ``NotImplementedError``.

Fix both layers locally:

* ``GptOssSplitRenderer.render_message`` pre-processes assistant
  messages whose ``ctx.idx <= ctx.last_user_index`` (historical turns)
  to drop their ``{type: thinking}`` content parts before delegating
  to upstream, so upstream's ``if thinking_content:`` branch does not
  fire and no analysis channel is emitted for history turns. Current
  turn (``idx > last_user_index``) is unchanged — its analysis is
  emitted as usual.
* ``DisaggregateMultiTurnMixin`` provides the multi-turn dispatch.
  After fix (1), ``has_extension_property`` (base default False) is
  semantically correct again — historical thinking is now stripped, so
  rendered token sequences are no longer prefix-extension-safe across
  successive turns, and per-user-turn disaggregation is required to
  align training tokens with HF inference.
"""

from __future__ import annotations

from typing import Any

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import RenderContext, RenderedMessage
from tinker_cookbook.renderers.base import Message as RenderMessage
from tinker_cookbook.renderers.base import ensure_list
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin


class GptOssSplitRenderer(DisaggregateMultiTurnMixin, GptOssRenderer):
    """gpt-oss subclass that strips historical thinking before
    delegating to upstream's ``render_message``, and disaggregates
    multi-turn SFT data so training tokens match HF
    ``apply_chat_template`` rendering.
    """

    def render_message(
        self, message: RenderMessage, ctx: RenderContext
    ) -> RenderedMessage:
        if (
            message.get("role") == "assistant"
            and ctx.last_user_index >= 0
            and ctx.idx <= ctx.last_user_index
        ):
            content = message.get("content")
            if content is not None:
                parts = ensure_list(content)
                stripped = [
                    p
                    for p in parts
                    if not (isinstance(p, dict) and p.get("type") == "thinking")
                ]
                if len(stripped) != len(parts):
                    new_msg: dict[str, Any] = dict(message)
                    new_msg["content"] = stripped
                    message = new_msg  # type: ignore[assignment]
        return super().render_message(message, ctx)


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
