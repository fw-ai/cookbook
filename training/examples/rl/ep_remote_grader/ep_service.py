"""Eval-Protocol adapter that implements :class:`RolloutService`.

NOT TRAINING-GRADE.  This example demonstrates the EP-as-RolloutService
boundary; the response side is still mocked and EP itself currently emits
text-only traces.  Real training requires per-call ``token_ids`` + per-token
assistant ``logprobs`` from the engine that produced the completion (see
``https://github.com/fw-ai/fireworks/issues/23756`` for the EP-side work
that wires the trace pipeline to emit them).  Use this example to learn the
wiring shape only — do not rely on it for production reward signal.

This is the **only** file in this example that imports ``eval_protocol``.
Everything upstream (the trainer, the generic packer) stays EP-unaware.
That boundary matches the EP integration ask — integrating EP into RL
training should require a small, stable surface, not the full pytest /
grader / dataset-adapter stack.

Wiring:

  * ``remote_agent_complete`` — stand-in for whatever completion source
    the user plugs in (agent framework, RAG pipeline, multi-turn tool
    loop, another LLM).  Returns token-native completions: token IDs +
    per-token assistant logprobs straight from the inference call.
  * ``test_math_answer_eval`` — an EP ``@evaluation_test`` coroutine.
    We call it directly with an :class:`EvaluationRow` to get a scalar
    reward.
  * ``EPService.rollout`` — fans out ``n`` completions, grades them
    concurrently, and packs each into a token-native
    :class:`RolloutPayload` with ``total_reward`` pre-filled
    (server-wins convention).

Request-side prompt tokens come from a renderer's ``build_generation_prompt``
flattened via :func:`model_input_to_token_ids`.  The renderer is required
at construction time and is typically built via
:func:`training.utils.supervised.build_renderer`.  No client-side prompt-
token synthesis path exists in this module.
"""

import asyncio
import logging
from typing import Any, List

from eval_protocol import EvaluationRow, Message

from training.examples.rl.ep_remote_grader.grader import test_math_answer_eval
from training.examples.rl.ep_remote_grader.mock_agent import (
    MockCompletion,
    remote_agent_complete,
)
from training.utils.rl.rollout import model_input_to_token_ids
from training.utils.rl.rollout import (
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
    :func:`make_remote_rollout_fn` to grade trainer-side.  Use
    ``grade=False`` when grading needs trainer-side state (reference
    model logprobs, a local reward model, metric joining against a
    separate dataset).
    """

    def __init__(self, *, renderer: Any, grade: bool = True) -> None:
        if renderer is None:
            raise ValueError(
                "EPService requires a renderer.  Build one via "
                "training.utils.supervised.build_renderer(tokenizer, tokenizer_model) "
                "and pass it as renderer=...; the request-side prompt token IDs "
                "must come from the renderer's build_generation_prompt(...)."
            )
        self.renderer = renderer
        self.grade = grade

    def _request_prompt_token_ids(self, messages: List[dict]) -> List[int]:
        """Build the prompt's token IDs from the renderer.

        Calls ``renderer.build_generation_prompt(messages)`` and flattens the
        resulting ``ModelInput`` via the cookbook's ``model_input_to_token_ids``
        adapter — the same path used by the in-process renderer-backed
        examples.  No fallback synthesis path exists here.
        """
        model_input = self.renderer.build_generation_prompt(messages)
        return model_input_to_token_ids(model_input)

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

        prompt_token_ids = self._request_prompt_token_ids(messages)

        return [
            _payload_from_completion(c, prompt_token_ids=prompt_token_ids, reward=r)
            for c, r in zip(completions, rewards)
        ]
