"""Adapters that lower simpler user-supplied inputs to a :class:`MessageEnv`.

These keep the ergonomic entry points cheap for the common cases:

- :class:`SingleTurnEnv` wraps a plain ``reward_fn(completion, row) -> float``
  into a one-step :class:`MessageEnv`.  The 80% of recipes that just want a
  custom reward never subclass :class:`MessageEnv` directly.
- :func:`wrap_reward_fn` takes a sync or async reward callable and returns
  an ``env_builder`` suitable for :data:`rl_loop.Config.env_builder`.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Union

from training.utils.rl.env import Message, MessageEnv, MessageStepResult

RewardFn = Callable[[str, dict], Union[float, Awaitable[float]]]
"""Reward callable: ``(completion_text, row) -> float | Awaitable[float]``."""

EnvBuilder = Callable[[dict], MessageEnv]
"""Row → env factory used by the RL loop to construct one env per rollout."""


def _extract_text(content: Any) -> str:
    """Coerce OpenAI-style ``content`` (str or list-of-parts) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


class SingleTurnEnv(MessageEnv):
    """One-step :class:`MessageEnv` driven by a user-supplied ``reward_fn``.

    Takes a dataset row and a reward callable; runs one turn, extracts the
    assistant completion text, awaits the reward (if async), and terminates.

    The row is expected to carry a ``messages`` field (OpenAI-compatible
    chat list).  If absent, ``initial_messages`` returns an empty list so
    downstream rendering raises a clear error.
    """

    def __init__(
        self,
        row: dict,
        reward_fn: RewardFn,
        *,
        messages_field: str = "messages",
    ) -> None:
        self._row = row
        self._reward_fn = reward_fn
        self._messages_field = messages_field

    async def initial_messages(self) -> list[Message]:
        return list(self._row.get(self._messages_field, []))

    async def step(self, assistant_message: Message) -> MessageStepResult:
        completion = _extract_text(assistant_message.get("content"))
        result = self._reward_fn(completion, self._row)
        if inspect.isawaitable(result):
            result = await result
        reward = float(result)
        return MessageStepResult(reward=reward, episode_done=True)


def wrap_reward_fn(reward_fn: RewardFn, **single_turn_kwargs: Any) -> EnvBuilder:
    """Return an env-builder that wraps ``reward_fn`` into a :class:`SingleTurnEnv`.

    Usage::

        train(Config(..., env_builder=wrap_reward_fn(my_reward)))

    Equivalent to, but more explicit than, setting ``Config.reward_fn`` —
    the RL loop auto-wraps ``reward_fn`` internally on behalf of users who
    just want the default single-turn path.
    """
    if not callable(reward_fn):
        raise TypeError(f"reward_fn must be callable, got {type(reward_fn).__name__}")

    def _build(row: dict) -> MessageEnv:
        return SingleTurnEnv(row=row, reward_fn=reward_fn, **single_turn_kwargs)

    return _build
