"""Message-level RL environment abstraction.

This module defines the extension point that user recipes implement to plug
a custom task into ``rl_loop.py``.  The training loop never knows how a
reward was computed or whether a rollout was single- or multi-turn — it only
consumes :class:`Trajectory` objects produced by running a :class:`MessageEnv`.

Concepts
--------
- :class:`MessageEnv` — stateful, single-use environment operating at the
  chat-message level.  Subclass this for any custom task.
- :class:`MessageStepResult` — what :meth:`MessageEnv.step` returns: the
  reward for the assistant turn plus whether the episode is done and any
  messages to append before the next sample.
- :class:`Transition` — one completed turn: the assistant tokens, their
  per-token inference logprobs, the reward, and metrics.
- :class:`Trajectory` — ordered list of :class:`Transition`.  A single-turn
  task produces a trajectory with exactly one transition.

Extension points in :class:`Trajectory` are deliberately minimal.  Users do
not construct Transitions themselves; the rollout runner does that from the
sampler output and the :class:`MessageStepResult` the env returned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]
"""Chat message, OpenAI-compatible: ``{"role": ..., "content": ..., ...}``."""


@dataclass
class MessageStepResult:
    """Result of a single assistant turn.

    Attributes:
        reward: Immediate reward for the assistant turn.
        episode_done: Whether the episode has ended.  Single-turn envs always
            set this to ``True``.
        next_messages: Messages (e.g. tool results, simulator feedback) to
            append to the conversation *before* the next sample.  Empty when
            the episode is done.
        metrics: Per-turn scalar metrics merged into training logs.
        is_reward_valid: Set to ``False`` when the grader could not score the
            turn (e.g. remote grader returned ``transient=true``).  The
            rollout runner may drop rollouts whose final turn is invalid.
    """

    reward: float
    episode_done: bool
    next_messages: list[Message] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    is_reward_valid: bool = True


class MessageEnv(ABC):
    """Stateful, single-use environment operating at the chat-message level.

    Subclasses implement :meth:`initial_messages` (the starting conversation)
    and :meth:`step` (score one assistant turn and decide what happens next).
    The training loop creates a fresh env per rollout — do not assume any
    env is reused.

    Minimal single-turn example::

        class MathEnv(MessageEnv):
            def __init__(self, row):
                self.row = row

            async def initial_messages(self):
                return self.row["messages"]

            async def step(self, assistant_message):
                completion = assistant_message.get("content", "")
                reward = 1.0 if self.row["answer"] in completion else 0.0
                return MessageStepResult(reward=reward, episode_done=True)

    Multi-turn envs should keep episode state on ``self`` and terminate by
    returning ``episode_done=True``.
    """

    @abstractmethod
    async def initial_messages(self) -> list[Message]:
        """Return the conversation the agent sees before its first turn."""

    @abstractmethod
    async def step(self, assistant_message: Message) -> MessageStepResult:
        """Score one assistant turn and advance the environment."""


@dataclass
class Transition:
    """One completed turn within a :class:`Trajectory`.

    Attributes:
        prompt_tokens: Rendered prompt tokens the sampler saw at this turn.
        completion_tokens: Assistant tokens produced by the sampler.
        completion_text: Decoded assistant text, convenient for graders and
            trajectory logging.
        inference_logprobs: Per-token logprobs from the sampler (aligned with
            ``completion_tokens``).  ``None`` when the rollout source does
            not return them; the rollout builder will recover them via an
            ``echo=True`` prefill call.
        assistant_message: The structured assistant message (role/content/
            tool_calls) passed to :meth:`MessageEnv.step`.
        reward: Reward returned by :meth:`MessageEnv.step`.
        episode_done: Whether this transition ended the episode.
        finish_reason: Sampler-reported finish reason (``"stop"``, ``"length"``,
            ...).  Used to flag truncation.
        is_reward_valid: Propagated from :class:`MessageStepResult`.
        metrics: Per-turn scalars.
        routing_matrices: Optional MoE routing matrices for router replay.
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_text: str
    inference_logprobs: list[float] | None
    assistant_message: Message
    reward: float
    episode_done: bool
    finish_reason: str = "stop"
    is_reward_valid: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    routing_matrices: Any | None = None


@dataclass
class Trajectory:
    """Ordered list of :class:`Transition` making up one episode."""

    transitions: list[Transition] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        """Sum of per-turn rewards across the trajectory."""
        return float(sum(t.reward for t in self.transitions))

    @property
    def is_complete(self) -> bool:
        """Whether the trajectory ended with an ``episode_done=True`` turn."""
        return bool(self.transitions) and self.transitions[-1].episode_done

    @property
    def any_truncated(self) -> bool:
        """Whether any turn hit the token budget (``finish_reason == "length"``)."""
        return any(t.finish_reason == "length" for t in self.transitions)

    @property
    def all_rewards_valid(self) -> bool:
        """Whether every turn had ``is_reward_valid=True``."""
        return all(t.is_reward_valid for t in self.transitions)

    def add_turn_reward(self, delta: float, *, turn_index: int = -1) -> None:
        """Add ``delta`` to the reward of one turn.

        Used by ``group_reward_fn`` to fold group-level rewards (pairwise
        reward models, etc.) back into the trajectory.  Default target is
        the final turn.
        """
        if not self.transitions:
            raise ValueError("Cannot add reward to an empty trajectory")
        self.transitions[turn_index].reward += float(delta)
