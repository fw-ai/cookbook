"""Sampler affinity and token ancestry for one deployment-backed agent run."""

from __future__ import annotations

import secrets
from typing import Any

from training.utils.rl.agent.trajectory import TrainingSessionTree


class DeploymentTrainingSession:
    """One rollout attempt's sampler affinity and token-level trajectory tree.

    The opaque affinity key is independent from agent credentials and logical
    rollout IDs. Every model call made through this object carries the same
    serving ``user`` value, which serving uses as session affinity when no
    explicit prompt-cache key is present. Examples remain responsible for
    rendering requests and resolving history parents; serving remains
    responsible for cache namespaces, active-request KV, and hotload policy.
    """

    def __init__(
        self,
        *,
        affinity_key: str | None = None,
    ) -> None:
        self.affinity_key = affinity_key or f"rl-session-{secrets.token_hex(16)}"
        self.tree = TrainingSessionTree()

    async def sample_with_prompt_tokens(
        self,
        sampler: Any,
        prompt_ids: list[int],
        **kwargs: Any,
    ) -> Any:
        supplied_user = kwargs.pop("user", None)
        if supplied_user is not None and supplied_user != self.affinity_key:
            raise ValueError("sampling user does not match training-session affinity")
        return await sampler.sample_with_prompt_tokens(
            prompt_ids,
            user=self.affinity_key,
            **kwargs,
        )


__all__ = ["DeploymentTrainingSession"]
