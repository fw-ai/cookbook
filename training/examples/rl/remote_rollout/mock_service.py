"""Mock ``RolloutService`` implementing the token-native contract.

Demonstrates the wiring users follow when their rollout backend lives
outside of the cookbook process: a service returns
``list[RolloutPayload]`` with per-turn ``token_ids`` and per-token
assistant ``logprobs`` already attached.  The cookbook's
``make_remote_rollout_fn`` (re-exported from
``training.utils.rl.renderer_rollout``) packs those payloads into
``RolloutSample`` via ``pack_payload_to_sample`` — token-native validation
is enforced, no fallback re-tokenization.
"""

from __future__ import annotations

import math
from typing import Any, List

from training.utils.rl.rollout_service import RolloutPayload, TurnRecord


def _logprobs_for(tokens: List[int]) -> List[float]:
    """Stand-in deterministic per-token logprob (production code uses the
    real engine response)."""
    return [-math.log(2 + (t % 7)) for t in tokens]


class MockRolloutService:
    """Returns deterministic single-turn payloads.  No network calls.

    Real services wrap their inference engine to produce token-native
    traces with the same shape.  The schema here matches the cookbook's
    ``TurnRecord`` / ``RolloutPayload`` exactly; provenance fields
    (renderer name, stop condition, model id) are intentionally NOT
    added because no in-scope consumer needs them yet.
    """

    def __init__(self, *, tokenizer_id: str = "mock-tokenizer") -> None:
        self.tokenizer_id = tokenizer_id

    async def rollout(
        self,
        messages: List[dict],
        *,
        n: int,
        sample_kwargs: dict[str, Any],
        row: dict,
    ) -> List[RolloutPayload]:
        # Toy: pretend the user prompt rendered to [1, 2, 3] and each
        # completion is the same prompt continued by [4, 5] with a small
        # reward dependent on the row id (so GRPO sees variance).
        prompt_tokens = [1, 2, 3]
        completion_tokens = [4, 5]
        payloads: List[RolloutPayload] = []
        for i in range(n):
            user_turn = TurnRecord(
                role="user",
                text=str(messages[-1].get("content", "") if messages else ""),
                token_ids=prompt_tokens,
            )
            assistant_turn = TurnRecord(
                role="assistant",
                text=f"completion-{i}",
                token_ids=completion_tokens,
                logprobs=_logprobs_for(completion_tokens),
                finish_reason="stop",
            )
            reward = 1.0 if i == 0 else 0.0
            payloads.append(RolloutPayload(
                turns=[user_turn, assistant_turn],
                total_reward=reward,
                tokenizer_id=self.tokenizer_id,
            ))
        return payloads
