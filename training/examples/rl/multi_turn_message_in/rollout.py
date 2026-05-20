"""Multi-turn GSM8K rollout (per-sample) with retry-on-wrong feedback.

Mirrors AReaL's ``examples/multi_turn_math/gsm8k_rl_mt.py``: ask GSM8K, score
the boxed answer, and -- if the model is wrong on the first try -- append a
fixed user-feedback message and let it try once more (configurable via
``setup.extras["max_turns"]``, default ``2``).

The whole trajectory (prompt + assistant turn 1 + feedback + assistant turn 2)
is packed into a single ``RolloutSample``.  ``MessageTrajectoryAssembler``
keeps the per-token loss mask aligned: ``1`` on every assistant-generated
token (across all turns), ``0`` everywhere else (original prompt, the
user-feedback bridge between turns).  The scalar ``reward`` is the
last-turn verification result (``0.0`` or ``1.0``); rolling-up across turns
isn't useful in concat mode -- the GRPO advantage compares trajectories, so
"wrong then right" naturally beats "wrong then wrong" without an explicit
per-turn discount.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from training.examples.rl.multi_turn_message_in.reward import gsm8k_reward
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import (
    MessageTrajectoryAssembler,
    RolloutSample,
    TITOTokenizer,
)

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)

RETRY_PROMPT = (
    "Your answer is either wrong or not parsable to the reward function. "
    "You may misunderstand the original question. Please carefully read the "
    "original question, check the previous errors, and try to answer it again."
)


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)
    tokenizer = setup.tokenizer
    max_turns = int(setup.extras.get("max_turns", 2))
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")

    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        messages = list(sample_prompt.get("messages") or [])
        answer = sample_prompt.get("answer")
        if not messages or answer is None:
            return None

        assembler = MessageTrajectoryAssembler(TITOTokenizer(tokenizer))
        current_messages = messages
        last_reward = 0.0

        for turn in range(max_turns):
            prompt_tokens = assembler.prepare_next_input(current_messages)
            completions = await sampler.sample_with_prompt_tokens(
                prompt_tokens, n=1, **sample_kwargs,
            )
            if not completions:
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
                return None

            assistant_text = getattr(completion, "text", "") or tokenizer.decode(output_tokens)
            assistant_message = {"role": "assistant", "content": assistant_text}
            assembler.add_assistant_response(
                request_messages=current_messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_tokens,
                completion_token_ids=output_tokens,
                completion_logprobs=output_logprobs,
                finish_reason=getattr(completion, "finish_reason", "stop"),
            )

            last_reward = gsm8k_reward(assistant_text, str(answer))
            if last_reward >= 1.0:
                break

            if turn + 1 < max_turns:
                current_messages = current_messages + [
                    assistant_message,
                    {"role": "user", "content": RETRY_PROMPT},
                ]

        tokens, logprobs, loss_mask = assembler.trajectory.to_flat()
        return RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=last_reward,
        )

    return rollout_fn
