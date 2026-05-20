"""Example-owned sampler helper for async RL rollouts."""

from __future__ import annotations

from fireworks.training.sdk.deployment import DeploymentSampler

from training.recipes.async_rl_loop import RolloutSetup


def build_deployment_sampler(setup: RolloutSetup) -> DeploymentSampler:
    """Construct a :class:`DeploymentSampler` from a :class:`RolloutSetup`.

    The training recipe assembles the setup once at startup and hands it
    to the rollout factory; the factory uses this helper to materialize
    a sampler bound to the inference deployment.  Concurrency is enforced
    by the async runner in sample (LLM-call) units via
    ``cfg.max_concurrency_rollout_sample`` -- the same unit the
    deployment's ``max_batch_size`` gates on.  No HTTP-layer gate.
    """
    return DeploymentSampler(
        inference_url=setup.inference_base_url,
        model=setup.model,
        api_key=setup.api_key,
        tokenizer=setup.tokenizer,
    )
