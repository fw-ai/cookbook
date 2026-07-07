"""Shared async-GRPO test helpers."""

from __future__ import annotations

import re
from collections.abc import Callable

from fireworks.training.sdk.deployment import DeploymentSampler

from training.recipes.async_rl_loop import RolloutFn, RolloutSetup
from training.utils.data import prepare_sampling_messages
from training.utils.rl.rollout import RolloutRun, RolloutSample


MAX_REALISTIC_COMPLETION_TOKENS = 32768


def gsm8k_numeric_reward(completion: str, row: dict) -> float:
    """Reward exact match on the final numeric answer; fall back to diversity."""
    ground_truth = row.get("ground_truth", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
    if numbers and ground_truth:
        truth_numbers = re.findall(r"-?\d+(?:\.\d+)?", ground_truth)
        if truth_numbers and numbers[-1] == truth_numbers[-1]:
            return 1.0
    return float(sum(ord(ch) for ch in completion) % 10) / 10.0


def completion_hash_reward(completion: str, _row: dict) -> float:
    """Small deterministic reward for smoke datasets without answer parsing."""
    return float(sum(ord(ch) for ch in completion) % 10)


def make_message_rollout_fn_factory(
    reward_fn: Callable[[str, dict], float],
) -> Callable[[RolloutSetup], RolloutFn]:
    """Build an async-RL rollout factory for message-in GSM8K-style rows."""

    def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
        sampler = DeploymentSampler(
            inference_url=setup.inference_base_url,
            model=setup.model,
            api_key=setup.api_key,
            tokenizer=setup.tokenizer,
        )
        sample_kwargs = dict(setup.sample_kwargs)

        async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
            messages = prepare_sampling_messages(list(sample_prompt.get("messages") or []))
            if not messages:
                return None

            completions = await sampler.sample_with_tokens(
                messages=messages,
                n=1,
                **sample_kwargs,
            )
            if not completions:
                return None

            completion = completions[0]
            tokens = list(completion.full_tokens)
            prompt_len = int(completion.prompt_len)
            output_tokens = tokens[prompt_len:]
            if not output_tokens:
                return None

            full_logprobs = _full_aligned_logprobs(
                completion_logprobs=list(completion.sampling_logprobs or []),
                token_count=len(tokens),
                prompt_len=prompt_len,
                is_echoed=bool(getattr(completion, "logprobs_echoed", False)),
            )
            if full_logprobs is None:
                return None

            sample = RolloutSample(
                tokens=tokens,
                logprobs=full_logprobs,
                loss_mask=[0] * prompt_len + [1] * len(output_tokens),
                reward=reward_fn(getattr(completion, "text", ""), sample_prompt),
                finish_reason=getattr(completion, "finish_reason", "stop"),
                text=getattr(completion, "text", ""),
            )
            return RolloutRun(segments=[sample])

        return rollout_fn

    return make_rollout_fn


def _full_aligned_logprobs(
    *,
    completion_logprobs: list[float | None],
    token_count: int,
    prompt_len: int,
    is_echoed: bool,
) -> list[float] | None:
    def _coerce(values: list[float | None], *, active_start: int = 0) -> list[float] | None:
        out: list[float] = []
        for idx, value in enumerate(values):
            if value is None:
                if idx >= active_start:
                    return None
                out.append(0.0)
            else:
                out.append(float(value))
        return out

    if is_echoed:
        if len(completion_logprobs) == token_count:
            return _coerce(completion_logprobs, active_start=prompt_len)
        if len(completion_logprobs) == token_count - 1:
            coerced = _coerce(completion_logprobs, active_start=max(0, prompt_len - 1))
            return [0.0] + coerced if coerced is not None else None
        return None

    output_len = token_count - prompt_len
    if len(completion_logprobs) != output_len:
        return None
    coerced = _coerce(completion_logprobs)
    return [0.0] * prompt_len + coerced if coerced is not None else None
