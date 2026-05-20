"""Single-turn token-in rollout (per-sample).

Input dataset rows carry ``prompt_token_ids``.  No messages or chat
templates.  The framework fans each dataset row out to
``completions_per_prompt`` parallel calls; each call returns one
:class:`RolloutSample`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import RolloutSample

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)

    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        prompt_token_ids = list(sample_prompt.get("prompt_token_ids") or [])
        if not prompt_token_ids:
            return None

        completions = await sampler.sample_with_prompt_tokens(
            prompt_token_ids, n=1, **sample_kwargs,
        )
        if not completions:
            return None
        completion = completions[0]
        prompt_len = int(completion.prompt_len)
        tokens = list(completion.full_tokens)
        output_tokens = tokens[prompt_len:]
        output_logprobs = list(completion.inference_logprobs or [])
        if getattr(completion, "logprobs_echoed", False) and len(output_logprobs) == len(tokens):
            output_logprobs = output_logprobs[prompt_len:]
        if not output_tokens or len(output_logprobs) != len(output_tokens):
            return None
        return RolloutSample(
            tokens=tokens,
            logprobs=[0.0] * prompt_len + output_logprobs,
            loss_mask=[0] * prompt_len + [1] * len(output_tokens),
            reward=float(sample_prompt.get("reward", 0.0)),
            finish_reason=getattr(completion, "finish_reason", "stop"),
            text=getattr(completion, "text", ""),
        )

    return rollout_fn
