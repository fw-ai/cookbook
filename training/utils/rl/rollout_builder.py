"""Build :class:`PromptGroup` objects from trajectories.

The rollout runner produces :class:`Trajectory` objects; this module packages
them into the ``(data, ref_data, advantages, inf_logprobs, ...)`` shape the
training step expects.  It also auto-recovers missing inference logprobs via
an ``echo=True`` prefill call when the rollout source only returned text.
"""

from __future__ import annotations

import logging
from typing import Any, List

import tinker

from training.utils.data import compute_advantages
from training.utils.rl.env import Trajectory
from training.utils.rl.losses import PromptGroup
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tokenize import get_prefill_logprobs

logger = logging.getLogger(__name__)

__all__ = ["trajectories_to_prompt_group"]


def _ensure_inference_logprobs(
    transition,
    *,
    inference_url: str | None,
    api_key: str | None,
    model: str | None,
) -> List[float]:
    """Return per-token logprobs for ``transition.completion_tokens``.

    If the rollout source already provided them use them as-is; otherwise call
    the deployment with ``echo=True, max_tokens=1`` to score the known token
    sequence under the current policy.
    """
    if transition.inference_logprobs is not None and transition.inference_logprobs:
        return list(transition.inference_logprobs)

    if not transition.completion_tokens:
        return []

    if not (inference_url and api_key and model):
        raise RuntimeError(
            "rollout_source returned text-only transitions but inference_url / "
            "api_key / model were not threaded through -- cannot recover "
            "logprobs via prefill."
        )

    full_tokens = list(transition.prompt_tokens) + list(transition.completion_tokens)
    full_lp = get_prefill_logprobs(
        url=inference_url, tokens=full_tokens, api_key=api_key, model=model,
    )
    n_comp = len(transition.completion_tokens)
    if len(full_lp) < n_comp:
        full_lp = full_lp + [0.0] * (n_comp - len(full_lp))
    return full_lp[-n_comp:]


def trajectories_to_prompt_group(
    trajectories: List[Trajectory],
    *,
    need_reference: bool,
    tokenizer: Any,
    inference_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    router_replay: bool = False,
    router_replay_completion_only: bool = False,
    keep_trajectory_logs: bool = False,
    row_meta: dict | None = None,
) -> PromptGroup | None:
    """Pack N trajectories (one row's group) into a :class:`PromptGroup`.

    Computes group-centered advantages over ``trajectory.total_reward``, builds
    policy and (optional) reference datums per trajectory, and fills inference
    logprobs -- auto-prefilling via ``echo=True`` when the rollout source did
    not return them.

    Returns ``None`` if no trajectory survived tokenization (e.g. every
    rollout was a single-token completion).
    """
    if not trajectories:
        return None

    rewards = [t.total_reward for t in trajectories]
    advantages = compute_advantages(rewards)

    policy_data: List[tinker.Datum] = []
    reference_data: List[tinker.Datum] = []
    adv_filtered: List[float] = []
    inf_logprobs_aligned: List[List[float]] = []
    completion_lens: List[int] = []
    truncated: List[bool] = []
    prompt_len_ref: int | None = None
    completion_texts: List[str] = []

    for idx, traj in enumerate(trajectories):
        if not traj.transitions:
            continue
        # Multi-turn envs: concatenate turn-wise prompts/completions onto the
        # initial prompt.  The initial prompt is the first transition's
        # prompt; later prompts are supersets that already include all prior
        # turns, so we only need to keep the tail of each later prompt plus
        # its completion.
        first = traj.transitions[0]
        tokens: List[int] = list(first.prompt_tokens) + list(first.completion_tokens)
        inf_lp: List[float] = list(
            _ensure_inference_logprobs(
                first, inference_url=inference_url, api_key=api_key, model=model,
            )
        )
        prompt_end = len(first.prompt_tokens)

        for transition in traj.transitions[1:]:
            prev_len = len(tokens)
            new_prompt = list(transition.prompt_tokens)
            if not (len(new_prompt) >= prev_len and new_prompt[:prev_len] == tokens):
                raise RuntimeError(
                    "Multi-turn transition prompt is not an extension of the "
                    "prior turn's tokens. This indicates a tokenizer/chat "
                    "template mismatch between turns."
                )
            tail = new_prompt[prev_len:]
            tokens.extend(tail)
            tokens.extend(transition.completion_tokens)
            # Extend inf_lp: prior turns' prompt-extension tokens contribute
            # zeros (not sampled), new turn's completion contributes its lp.
            inf_lp.extend([0.0] * len(tail))
            inf_lp.extend(
                _ensure_inference_logprobs(
                    transition,
                    inference_url=inference_url,
                    api_key=api_key,
                    model=model,
                )
            )

        if len(tokens) < 2:
            continue

        model_input_len = len(tokens) - 1

        rm = None
        if router_replay and first.routing_matrices is not None:
            rm = build_r3_routing_matrices(
                first.routing_matrices,
                prompt_end,
                model_input_len,
                completion_only=router_replay_completion_only,
            )

        policy_datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=tokens[1:], dtype="int64", shape=[model_input_len],
                ),
            },
        )
        policy_data.append(policy_datum)

        if need_reference:
            reference_datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=tokens[1:], dtype="int64", shape=[model_input_len],
                    ),
                },
            )
            reference_data.append(reference_datum)

        adv_filtered.append(advantages[idx])

        # Align inf_lp to len(model_input_len) by prepending zeros for the
        # prompt positions that did not correspond to generated tokens.
        n_comp = sum(len(t.completion_tokens) for t in traj.transitions)
        # Our inf_lp currently has zeros for prompt-extensions between turns
        # plus generated-token logprobs.  For the "first turn only" case
        # len(inf_lp) == n_comp.
        response_start = model_input_len - n_comp
        aligned = [0.0] * max(0, response_start) + inf_lp
        if len(aligned) > model_input_len:
            aligned = aligned[-model_input_len:]
        elif len(aligned) < model_input_len:
            aligned = aligned + [0.0] * (model_input_len - len(aligned))
        inf_logprobs_aligned.append(aligned)

        if prompt_len_ref is None:
            prompt_len_ref = prompt_end
        completion_lens.append(n_comp)
        truncated.append(traj.any_truncated)
        completion_texts.append(" ".join(t.completion_text for t in traj.transitions))

    if not policy_data or prompt_len_ref is None:
        return None

    prompt_messages = None
    completions_log = None
    meta_log = None
    if keep_trajectory_logs:
        prompt_messages = []
        completions_log = completion_texts
        meta_log = dict(row_meta or {})

    return PromptGroup(
        data=policy_data,
        ref_data=reference_data,
        advantages=adv_filtered,
        ref_logprobs=None,
        prompt_len=prompt_len_ref,
        rewards=rewards[: len(adv_filtered)] if len(adv_filtered) < len(rewards) else rewards,
        inf_logprobs=inf_logprobs_aligned,
        completion_lens=completion_lens,
        truncated=truncated,
        prompt=prompt_messages,
        completions=completions_log,
        row_meta=meta_log,
    )
