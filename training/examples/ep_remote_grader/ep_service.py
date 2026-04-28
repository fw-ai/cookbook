"""Eval-Protocol adapter that implements :class:`RolloutService`.

This is the **only** file in this example that imports ``eval_protocol``.
Everything upstream (the trainer, the generic packer) stays EP-unaware.
That boundary matches the EP integration ask -- integrating EP into RL
training should require a small, stable surface, not the full pytest /
grader / dataset-adapter stack.

Wiring:

  * ``remote_agent_complete`` -- stand-in for whatever completion source
    the user plugs in (agent framework, RAG pipeline, multi-turn tool
    loop, another LLM).  Returns token-native completions: token IDs +
    per-token assistant logprobs straight from the inference call.
  * ``test_math_answer_eval`` -- an EP ``@evaluation_test`` coroutine.
    We call it directly with an :class:`EvaluationRow` to get a scalar
    reward.
  * ``EPService.rollout`` -- fans out ``n`` completions, grades them
    concurrently, and packs each into a token-native
    :class:`RolloutPayload` with ``total_reward`` pre-filled
    (server-wins convention).

EP today emits text-only traces; the cookbook trainer is token-native
only.  This example uses the mock agent's synthetic token IDs to
demonstrate the target shape.  See
``https://github.com/fw-ai/fireworks/issues/23756`` for the EP-side
work that wires the trace pipeline to emit per-call ``token_ids`` +
per-token assistant ``logprobs`` so ``RemoteRolloutProcessor`` can
drive RL training without a re-tokenization step.
"""

import asyncio
import logging
from typing import Any, List

from eval_protocol import EvaluationRow, Message

from training.examples.ep_remote_grader.grader import test_math_answer_eval
from training.examples.ep_remote_grader.mock_agent import (
    MockCompletion,
    remote_agent_complete,
)
from training.utils.rl.rollout_service import (
    RolloutPayload,
    RolloutService,
    TurnRecord,
)


logger = logging.getLogger(__name__)


async def _grade(
    prompt_messages: List[dict],
    completion_text: str,
    ground_truth: str,
) -> float:
    row = EvaluationRow(
        messages=[Message(**m) for m in prompt_messages]
        + [Message(role="assistant", content=completion_text)],
        ground_truth=ground_truth,
    )
    graded = await test_math_answer_eval(row)
    if graded.evaluation_result is None:
        return 0.0
    return float(graded.evaluation_result.score)


def _payload_from_completion(
    completion: MockCompletion,
    *,
    prompt_token_ids: List[int],
    reward: float | None,
) -> RolloutPayload:
    """Pack a token-native completion into a :class:`RolloutPayload`.

    Two turns: the prompt (mask 0) and the assistant span (mask 1).
    Both ``token_ids`` and ``logprobs`` come straight from the
    inference call -- never re-tokenized.
    """
    return RolloutPayload(
        turns=[
            TurnRecord(role="user", token_ids=list(prompt_token_ids)),
            TurnRecord(
                role="assistant",
                text=completion.text,
                token_ids=list(completion.token_ids),
                logprobs=list(completion.logprobs),
                finish_reason=completion.finish_reason,
            ),
        ],
        total_reward=(None if reward is None else float(reward)),
        finish_reason=completion.finish_reason,
    )


class EPService(RolloutService):
    """Remote completion + EP grader exposed as a :class:`RolloutService`.

    ``grade=True`` (default) runs the EP grader inside the service and
    fills :attr:`RolloutPayload.total_reward` -- the trainer trusts it
    ("server-wins" convention).  ``grade=False`` leaves ``total_reward``
    as ``None`` and the caller supplies ``reward_fn`` to
    :func:`make_text_rollout_fn` to grade trainer-side.  Use
    ``grade=False`` when grading needs trainer-side state (reference
    model logprobs, a local reward model, metric joining against a
    separate dataset).
    """

    def __init__(self, *, grade: bool = True) -> None:
        self.grade = grade

    async def rollout(
        self,
        messages: List[dict],
        *,
        n: int,
        sample_kwargs: dict[str, Any],
        row: dict,
    ) -> List[RolloutPayload]:
        completions = await remote_agent_complete(
            messages,
            n=n,
            completion_params={
                "temperature": sample_kwargs.get("temperature", 1.0),
                "max_tokens": sample_kwargs.get("max_tokens", 1024),
            },
        )

        if self.grade:
            ground_truth = str(row.get("ground_truth", ""))
            rewards: List[float | None] = list(await asyncio.gather(
                *(_grade(messages, c.text, ground_truth) for c in completions),
            ))
        else:
            rewards = [None] * len(completions)

        # The prompt's token IDs would come from the inference server's
        # request trace in production.  In this mock we synthesize them
        # deterministically so the example is self-contained.
        prompt_token_ids = [2000 + (ord(c) % 100) for m in messages for c in m["content"]]

        return [
            _payload_from_completion(c, prompt_token_ids=prompt_token_ids, reward=r)
            for c, r in zip(completions, rewards)
        ]
