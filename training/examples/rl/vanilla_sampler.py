"""Compatibility helper for async RL rollout samplers."""

from __future__ import annotations

from typing import Any

from fireworks.training.sdk.deployment import DeploymentSampler

from training.recipes.async_rl_loop import RolloutSetup


def build_deployment_sampler(setup: RolloutSetup) -> Any:
    """Return the borrowed setup sampler or reconstruct one for legacy callers.

    Current dedicated and serverless recipes inject one recipe-owned sampler
    into ``RolloutSetup``. Rollout factories share that object across all
    trajectories and evaluation calls and must not close it. Constructing from
    the endpoint fields is retained only for older or manually assembled
    ``RolloutSetup`` values; those callers own the constructed sampler.

    Concurrency is enforced by the async runner in sample (LLM-call) units via
    ``cfg.max_concurrency_rollout_sample`` -- the same unit the
    deployment's ``max_batch_size`` gates on.  No HTTP-layer gate.
    """
    if setup.sampler is not None:
        return setup.sampler
    return DeploymentSampler(
        inference_url=setup.inference_base_url,
        model=setup.model,
        api_key=setup.api_key,
        tokenizer=setup.tokenizer,
    )
