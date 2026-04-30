"""Renderer-backed tool-using multi-turn rollout — canonical example.

A concrete, hand-coded ``async def rollout_fn(row, ctx)`` for tool-using
agents.  Like its sibling ``multi_turn_minimal_renderer/rollout.py``, this is
NOT a framework helper — it is an example users read and copy.  Both
example rollouts call into the same shared private step coroutine
(``examples/rl/_renderer_turn_loop.py``) for the renderer + sampler +
assembler inner loop, so that core is single-sourced.

Pipeline (the shared step handles everything up to and including
``parse_response``)::

    shared step (extension-property guard, render, sample, assemble, parse)
    -> if tool_calls: env.execute(tool_call) -> tool_message  (loop)
       else:           env.step(parsed)              -> (next, reward, done)
    (done) -> pack_assembled_to_sample(assembler, total_reward=...)

Tool execution lives in the user-supplied env; the renderer's responsibility
ends at parsing tool calls from assistant tokens.  ``loss_mask=1`` is
assigned only to assistant tokens (enforced by ``TrajectoryAssembler``);
tool / user / template-suffix gap tokens carry ``loss_mask=0``.

Env shape (documented convention only; NOT an exported Protocol)
----------------------------------------------------------------

::

    async def initial_messages(self) -> list[Message]
    async def execute(self, tool_call: Any) -> Message
        # Returns a Message with role="tool" containing the tool's reply.
    async def step(self, parsed: Message)
        -> tuple[list[Message], float, bool]
        # Used when the assistant turn does not include a tool call.
        # Returns (next_messages, reward, done).
    async def reward(self, messages: list[Message]) -> float
        # Optional: terminal reward computed once done=True.

The ``parse_response`` return shape is renderer-specific; this example
expects a Tinker-style ``Message`` whose ``tool_calls`` attribute is either
empty / None (no tool call) or a list of tool-call records.
"""

from __future__ import annotations

import logging
from typing import Any

from training.examples.rl._renderer_turn_loop import (
    ExtensionPropertyError,
    pack_assembled_to_sample,
    renderer_turn_step,
)
from training.utils.rl.rollout import Rollout
from training.utils.rl.trajectory_assembler import TrajectoryAssembler


logger = logging.getLogger(__name__)


__all__ = ["rollout_fn"]


def _extract_tool_calls(parsed_message: Any) -> list[Any]:
    """Pull the tool-call list off a parsed assistant message, if any."""
    tool_calls = getattr(parsed_message, "tool_calls", None)
    if tool_calls is None and isinstance(parsed_message, dict):
        tool_calls = parsed_message.get("tool_calls")
    return list(tool_calls) if tool_calls else []


async def rollout_fn(row: dict, ctx: Any) -> Rollout | None:
    """Tool-using multi-turn renderer-backed rollout."""
    renderer = ctx.renderer
    sample_with_prompt_tokens = ctx.sample_with_prompt_tokens
    build_env = ctx.build_env
    max_tokens = getattr(ctx, "max_tokens", None)
    sample_kwargs = getattr(ctx, "sample_kwargs", None)

    env = build_env(row)
    messages = list(await env.initial_messages())

    assembler = TrajectoryAssembler(tokenizer_id=getattr(ctx, "tokenizer_id", None))
    assistant_turns_seen = 0
    last_step_reward: float = 0.0

    current_version_fn = getattr(ctx, "current_version", lambda: -1)
    last_finish_reason = "stop"

    while True:
        try:
            outcome = await renderer_turn_step(
                messages=messages,
                renderer=renderer,
                sample_with_prompt_tokens=sample_with_prompt_tokens,
                assembler=assembler,
                turns_seen=assistant_turns_seen,
                sample_kwargs=sample_kwargs,
                max_tokens=max_tokens,
                turn_version=current_version_fn(),
            )
        except ExtensionPropertyError:
            raise

        if outcome.dropped:
            logger.warning("dropping tool rollout: turn step returned dropped")
            return None
        if not outcome.parse_success:
            return None

        last_finish_reason = outcome.finish_reason
        assistant_turns_seen += 1
        tool_calls = _extract_tool_calls(outcome.parsed_message)

        if tool_calls:
            tool_messages: list[Any] = []
            for tc in tool_calls:
                tool_messages.append(await env.execute(tc))
            messages = list(messages) + [outcome.parsed_message] + tool_messages
            continue

        next_messages, step_reward, done = await env.step(outcome.parsed_message)
        last_step_reward = float(step_reward)
        messages = list(messages) + [outcome.parsed_message] + list(next_messages)

        if done:
            terminal_reward = last_step_reward
            terminal_fn = getattr(env, "reward", None)
            if terminal_fn is not None:
                terminal_reward = float(await terminal_fn(messages))
            sample = pack_assembled_to_sample(
                assembler,
                total_reward=terminal_reward,
                finish_reason=last_finish_reason,
            )
            return Rollout(samples=[sample], row_meta={"row_id": row.get("id")})
