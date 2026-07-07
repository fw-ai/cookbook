"""Single-turn token-in rollout (per-run).

Input dataset rows carry ``prompt_token_ids``.  No messages or chat
templates.  The framework fans each dataset row out to
``completions_per_prompt`` parallel calls; each call returns one
:class:`RolloutRun` with one :class:`RolloutSample` segment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import RolloutRun, RolloutSample

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup


def _completion_logprobs(completion, *, attr: str) -> list[float] | None:
    values = getattr(completion, attr, None)
    if values is None:
        return None
    values = list(values)
    prompt_len = int(completion.prompt_len)
    output_len = len(completion.full_tokens) - prompt_len
    if getattr(completion, "logprobs_echoed", False):
        full_len = len(completion.full_tokens)
        if len(values) == full_len:
            values = values[prompt_len:]
        elif len(values) == max(0, full_len - 1):
            values = values[max(0, prompt_len - 1):]
    if len(values) != output_len or any(v is None for v in values):
        return None
    return [float(v) for v in values]


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)

    async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
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
        output_logprobs = _completion_logprobs(
            completion, attr="sampling_logprobs",
        )
        if not output_tokens or output_logprobs is None:
            return None
        sample = RolloutSample(
            tokens=tokens,
            logprobs=[0.0] * prompt_len + output_logprobs,
            loss_mask=[0] * prompt_len + [1] * len(output_tokens),
            reward=float(sample_prompt.get("reward", 0.0)),
            finish_reason=getattr(completion, "finish_reason", "stop"),
            text=getattr(completion, "text", ""),
        )
        return RolloutRun(segments=[sample])

    return rollout_fn
