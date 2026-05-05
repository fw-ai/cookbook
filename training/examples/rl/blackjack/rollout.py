"""Blackjack per-sample rollout for the async RL recipe.

Exposes ``make_rollout_fn(setup) -> rollout_fn`` following the
async_rl_loop contract: one episode per call, one RolloutSample returned.
The recipe fans each seed out to ``completions_per_prompt`` parallel calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from training.examples.rl.blackjack.blackjack_env import (
    build_blackjack_tool_env,
    build_blackjack_user_prompt,
)
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import (
    MessageTrajectoryAssembler,
    RolloutSample,
    TITOTokenizer,
)

if TYPE_CHECKING:
    from fireworks.training.sdk.deployment import DeploymentSampler, SampledCompletion
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup


async def _sample_with_prompt_ids(
    sampler: "DeploymentSampler",
    prompt_ids: list[int],
    sample_kwargs: dict[str, Any],
) -> "list[SampledCompletion]":
    """Call the sampler with pre-tokenized prompt IDs.

    Sends token IDs directly to the engine so that TITO token IDs from
    prepare_next_input are preserved verbatim, maintaining the prefix-continuity
    invariant required by TrajectoryAssembler.
    """
    kw = dict(sample_kwargs)
    max_tokens = kw.pop("max_tokens", 1024)
    temperature = kw.pop("temperature", 1.0)
    max_seq_len = kw.pop("max_seq_len", None)
    user_requested_logprobs = bool(kw.get("logprobs", False))
    routing_requested = bool(kw.get("include_routing_matrix", False))
    echo_mode = bool(kw.get("echo", False))
    return await sampler._do_one_completion(
        prompt_ids, max_tokens, temperature, max_seq_len,
        user_requested_logprobs, routing_requested, echo_mode, **kw,
    )

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are playing Blackjack against a dealer.\n"
    "Rules:\n"
    "- Number cards are worth their face value. Face cards (J, Q, K) are worth 10. Aces are worth 11 unless that would bust you, in which case they are worth 1.\n"
    "- You lose if your hand exceeds 21 (bust).\n"
    "- After you stick, the dealer draws until reaching 17 or higher.\n"
    "- You win if the dealer busts, or if your sum is higher than the dealer's without busting.\n"
    "Your response MUST start with 'hit' or 'stick' as the very first word. No other text before it."
)

_ACTION_RE = re.compile(
    r"\b(hit(?:ting)?|st(?:ick(?:ing)?|and(?:ing)?))\b", re.IGNORECASE
)


def _parse_action(text: str) -> str | None:
    m = _ACTION_RE.search(text)
    if not m:
        return None
    word = m.group(1).lower()
    return "hit" if word.startswith("hit") else "stick"


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)
    tokenizer = setup.tokenizer

    system_prompt: str = setup.extras.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_steps: int = int(setup.extras.get("max_steps", 20))
    natural: bool = bool(setup.extras.get("natural", False))
    sab: bool = bool(setup.extras.get("sab", False))
    no_thinking: bool = bool(setup.extras.get("no_thinking", False))

    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        try:
            return await _rollout_fn_inner(sample_prompt)
        except Exception:
            logger.exception("rollout_fn raised for seed=%s", sample_prompt.get("seed"))
            return None

    async def _rollout_fn_inner(sample_prompt: dict) -> RolloutSample | None:
        env = build_blackjack_tool_env(
            environment_context={
                "player_cards": sample_prompt.get("player_cards"),
                "dealer_cards": sample_prompt.get("dealer_cards"),
                "seed": sample_prompt.get("seed", 0),
            },
            natural=natural,
            sab=sab,
            max_steps=max_steps,
        )
        state = env.reset()
        observation = str(state.get("observation") or "")
        done = bool(state.get("terminated") or state.get("truncated"))

        chat_template_kwargs = {"enable_thinking": False} if no_thinking else {}
        assembler = MessageTrajectoryAssembler(TITOTokenizer(tokenizer, chat_template_kwargs=chat_template_kwargs))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_blackjack_user_prompt(None, observation)},
        ]

        step_rewards: list[float] = []
        tool_call_traces: list[dict] = []
        last_text = ""
        last_finish_reason = "stop"

        for step in range(max_steps):
            if done:
                break

            prompt_tokens = assembler.prepare_next_input(messages)
            completions = await _sample_with_prompt_ids(sampler, prompt_tokens, sample_kwargs)
            if not completions:
                logger.warning("seed=%s step=%d: _do_one_completion returned empty list", sample_prompt.get("seed"), step)
                return None

            completion = completions[0]
            prompt_len = int(completion.prompt_len)
            output_tokens = list(completion.full_tokens[prompt_len:])
            output_logprobs = list(completion.inference_logprobs or [])
            if (
                getattr(completion, "logprobs_echoed", False)
                and len(output_logprobs) == len(completion.full_tokens)
            ):
                output_logprobs = output_logprobs[prompt_len:]
            if not output_tokens or len(output_logprobs) != len(output_tokens):
                logger.warning(
                    "seed=%s step=%d: token/logprob mismatch — output_tokens=%d output_logprobs=%d inference_logprobs=%s",
                    sample_prompt.get("seed"), step, len(output_tokens), len(output_logprobs),
                    "None" if completion.inference_logprobs is None else len(completion.inference_logprobs),
                )
                return None

            assistant_text = (
                getattr(completion, "text", "") or tokenizer.decode(output_tokens)
            )
            last_text = assistant_text
            last_finish_reason = getattr(completion, "finish_reason", "stop")

            action = _parse_action(assistant_text)
            if action is None:
                logger.warning("seed=%s step=%d: no action in %r (finish=%s)", sample_prompt.get("seed"), step, assistant_text[:120], last_finish_reason)
                return None

            assistant_message = {"role": "assistant", "content": assistant_text}
            assembler.add_assistant_response(
                request_messages=messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_tokens,
                completion_token_ids=output_tokens,
                completion_logprobs=output_logprobs,
                finish_reason=last_finish_reason,
            )

            state = env.step(action)
            reward = float(state.get("reward", 0.0))
            done = bool(state.get("terminated") or state.get("truncated"))
            observation = str(state.get("observation") or "")

            step_rewards.append(reward)
            tool_call_traces.append({
                "step": step + 1,
                "action": action,
                "reward": reward,
                "terminated": bool(state.get("terminated", False)),
                "player_sum": state.get("player_sum"),
                "dealer_card": state.get("dealer_card"),
            })

            if not done:
                messages = messages + [
                    assistant_message,
                    {"role": "user", "content": build_blackjack_user_prompt(None, observation)},
                ]

        if not step_rewards:
            return None

        tokens, logprobs, loss_mask = assembler.trajectory.to_flat()
        return RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=step_rewards[-1],
            finish_reason=last_finish_reason,
            text=json.dumps({
                "last_output": last_text,
                "step_rewards": step_rewards,
                "tool_call_traces": tool_call_traces,
            }),
        )

    return rollout_fn
