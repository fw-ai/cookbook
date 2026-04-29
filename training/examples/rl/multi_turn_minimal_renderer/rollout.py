"""Renderer-backed multi-turn rollout — canonical example to copy.

This is a *concrete example file*, not a framework helper.  Multi-turn flows
in the cookbook ship as hand-coded ``async def rollout_fn(row, ctx)`` files
that users read and adapt to their environment, not as `utils/rl/` helpers
parameterized by `MessageEnv` / `ToolEnv` Protocols.  The trainer keeps its
slime-style contract: ``rollout_fn(row, ctx) -> Rollout | None``.

Pipeline per turn (sourced from the shared private step
``examples/rl/_renderer_turn_loop.py``)::

    extension-property guard (when turns_seen==1)
        -> renderer.build_generation_prompt
        -> model_input_to_token_ids
        -> sample_with_prompt_tokens(prompt_token_ids, ...)
        -> TrajectoryAssembler.add_call(InferenceCall(...))
        -> renderer.parse_response(out_tokens)
    -> env.step(parsed_message)
    (done) -> assembler.to_payload(total_reward=...) -> pack_payload_to_sample

The shared step (``renderer_turn_step``) is single-sourced for the
``multi_turn_minimal_renderer`` and ``multi_turn_tool`` example rollouts so
the renderer / token / masking / assembly inner loop is not duplicated.

Env shape (documented convention only; NOT an exported Protocol)
----------------------------------------------------------------

The ``env`` object passed to ``rollout_fn`` (built by ``ctx.build_env(row)``)
must expose two callables::

    async def initial_messages(self) -> list[Message]
    async def step(self, parsed: Message)
        -> tuple[list[Message], float, bool]   # (next_messages, reward, done)

When ``done=True`` the rollout terminates with the most recent ``reward`` as
``total_reward`` for the trajectory.  Tool-using flows are a separate sibling
example (``multi_turn_tool/rollout.py``) that adds an ``execute(tool_call)``
callable on the env and shares the same per-turn step coroutine.

Parse-failure / truncation handling
-----------------------------------

DROP and zero-reward are user-code patterns, not framework primitives:

* DROP: ``return None`` from this rollout function.
* Zero-reward: ``return Rollout(samples=[RolloutSample(reward=0.0, ...)])``.
* Length-as-terminal: ``finish_reason='length'`` flows through unchanged
  unless the user branches on it explicitly.

Extension-property guard
------------------------

Renderers without the sequence-extension property (e.g. Qwen3 with its
default ``strip_thinking_from_history=True``) cannot be safely flattened
into a single training datum across turns.  The shared step coroutine
raises :class:`ExtensionPropertyError` (a 3-line guard, no typed error
class exported from ``utils/rl/``) before the second sampling call.
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


async def rollout_fn(row: dict, ctx: Any) -> Rollout | None:
    """Multi-turn renderer-backed rollout.

    ``ctx`` is the trainer's :class:`RolloutContext` (extended by the
    example's wiring layer with ``renderer``, ``sample_with_prompt_tokens``,
    and ``build_env``).  ``RolloutContext`` itself is unchanged in this
    iteration — these fields are attached by the wiring code that constructs
    the trainer's per-row callable, not by adding fields to the dataclass.
    """
    renderer = ctx.renderer
    sample_with_prompt_tokens = ctx.sample_with_prompt_tokens
    build_env = ctx.build_env
    max_tokens = getattr(ctx, "max_tokens", None)
    sample_kwargs = getattr(ctx, "sample_kwargs", None)

    env = build_env(row)
    messages = list(await env.initial_messages())

    assembler = TrajectoryAssembler(tokenizer_id=getattr(ctx, "tokenizer_id", None))
    turns_seen = 0
    last_reward: float = 0.0

    current_version_fn = getattr(ctx, "current_version", lambda: -1)
    last_finish_reason = "stop"
    last_text = ""

    while True:
        try:
            outcome = await renderer_turn_step(
                messages=messages,
                renderer=renderer,
                sample_with_prompt_tokens=sample_with_prompt_tokens,
                assembler=assembler,
                turns_seen=turns_seen,
                sample_kwargs=sample_kwargs,
                max_tokens=max_tokens,
                turn_version=current_version_fn(),
            )
        except ExtensionPropertyError:
            raise

        if outcome.dropped:
            logger.warning("dropping multi-turn rollout: turn step returned dropped")
            return None
        if not outcome.parse_success:
            return None

        last_finish_reason = outcome.finish_reason
        next_messages, step_reward, done = await env.step(outcome.parsed_message)
        last_reward = float(step_reward)
        messages = list(messages) + [outcome.parsed_message] + list(next_messages)
        turns_seen += 1

        if done:
            sample = pack_assembled_to_sample(
                assembler,
                total_reward=last_reward,
                finish_reason=last_finish_reason,
                text=last_text,
            )
            return Rollout(samples=[sample], row_meta={"row_id": row.get("id")})
