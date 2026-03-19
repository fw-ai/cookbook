"""Datum construction utilities for RL rollouts.

Extracts the datum-building logic from ``sample_one_prompt`` so the
sampling closure in ``rl_loop.py`` stays small and readable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List

import tinker

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup
from training.utils.rl.router_replay import build_r3_routing_matrices

logger = logging.getLogger(__name__)

RewardFn = Callable[[str, dict], float]
"""Signature: (completion_text, dataset_row) -> reward_float."""


def build_prompt_group(
    sampled: list,
    row: dict,
    *,
    reward_fn: RewardFn,
    completions_per_prompt: int,
    use_reference: bool = False,
    router_replay: bool = False,
    router_replay_completion_only: bool = True,
    trajectory_dir: str | None = None,
    input_messages: list[dict] | None = None,
) -> PromptGroup | None:
    """Build a :class:`PromptGroup` from sampled completions.

    This is the pure data-assembly step: reward computation, advantage
    normalisation, datum construction, and logprob alignment.  No I/O.

    Parameters
    ----------
    sampled
        List of ``SampleWithTokens`` returned by
        ``DeploymentSampler.sample_with_tokens``.
    row
        Original dataset row (passed through to ``reward_fn``).
    reward_fn
        ``(completion_text, dataset_row) -> float``.
    completions_per_prompt
        Expected number of completions.  If ``sampled`` has fewer,
        returns ``None``.
    """
    if not sampled or len(sampled) < completions_per_prompt:
        return None

    rewards = [reward_fn(s.text, row) for s in sampled]
    advantages = compute_advantages(rewards)

    prompt_len = sampled[0].prompt_len
    policy_data: List[tinker.Datum] = []
    reference_data: List[tinker.Datum] = []
    adv_filtered: List[float] = []
    inf_logprobs_aligned: List[List[float]] = []

    for idx, s in enumerate(sampled):
        tokens = s.full_tokens
        if len(tokens) < 2:
            continue
        model_input_len = len(tokens) - 1

        rm = None
        if router_replay:
            rm = build_r3_routing_matrices(
                s.routing_matrices,
                s.prompt_len,
                model_input_len,
                completion_only=router_replay_completion_only,
            )

        policy_datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=tokens[1:], dtype="int64", shape=[model_input_len]
                ),
            },
        )
        policy_data.append(policy_datum)

        if use_reference:
            reference_datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=tokens[1:], dtype="int64", shape=[model_input_len]
                    ),
                },
            )
            reference_data.append(reference_datum)

        adv_filtered.append(advantages[idx])

        if not s.inference_logprobs:
            raise RuntimeError(
                f"Inference logprobs required but sample {idx} has none. "
                f"Ensure the deployment returns logprobs."
            )
        response_start = max(0, prompt_len - 1)
        echoed = getattr(s, "logprobs_echoed", False)
        aligned = (
            list(s.inference_logprobs)
            if echoed
            else [0.0] * response_start + list(s.inference_logprobs)
        )
        inf_logprobs_aligned.append(aligned)

    if not policy_data:
        return None

    comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled]
    trunc = [s.finish_reason == "length" for s in sampled]

    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=adv_filtered,
        ref_logprobs=None,
        prompt_len=prompt_len,
        rewards=rewards,
        inf_logprobs=inf_logprobs_aligned,
        completion_lens=comp_lens,
        truncated=trunc,
        prompt=input_messages if trajectory_dir else None,
        completions=[s.text for s in sampled] if trajectory_dir else None,
        row_meta={"ground_truth": row.get("ground_truth", "")} if trajectory_dir else None,
    )
