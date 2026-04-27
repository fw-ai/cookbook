"""Mock remote agent service.

Stands in for any production agent framework: a chain-of-tools pipeline,
a RAG-with-verifier stack, an LLM-as-judge loop.  Returns ``n``
*token-native* completions concurrently -- the same shape a production
service must emit to drive RL training.  See
``https://github.com/fw-ai/fireworks/issues/23512`` for the EP-side
work that lets ``RemoteRolloutProcessor`` produce this shape from real
inference traces.

A real integration replaces this with an HTTP / gRPC / SDK call that
returns per-completion token IDs and per-token assistant logprobs
straight from the same inference call that generated them
(slime/AReaL convention).  Re-tokenizing text post-hoc is not
supported because it silently misaligns the loss mask and inference
logprobs.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


@dataclass
class MockCompletion:
    """Per-completion result from a token-native rollout service.

    A real service returns these fields straight from the inference
    call (no re-tokenization).  Multi-turn rollouts return per-turn
    spans; this single-turn mock collapses to one assistant span.
    """

    text: str
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str = "stop"


async def remote_agent_complete(
    messages: list[dict],
    *,
    n: int,
    completion_params: dict,
) -> list[MockCompletion]:
    """Return ``n`` token-native completions for ``messages``.

    Simulated latency is ~10ms per completion, so a group of 4 completes
    concurrently in ~10ms rather than ~40ms.  Token IDs / logprobs are
    synthetic (this is a mock); the shape matches what a real
    inference-serving trace will emit.
    """
    seed = hash(str(messages)) & 0xFFFFFFFF
    rng = random.Random(seed)

    async def _one(idx: int) -> MockCompletion:
        await asyncio.sleep(0.01)
        text = (
            f"Let me work through this step by step.\n"
            f"(variant {idx + 1})\n"
            f"<answer>{rng.randint(0, 100)}</answer>"
        )
        # Synthetic token-native payload: in production these come from
        # the inference response, never from a local tokenizer.
        token_ids = [1000 + (ord(c) % 100) for c in text]
        logprobs = [-0.1 * (i % 5 + 1) for i in range(len(token_ids))]
        return MockCompletion(text=text, token_ids=token_ids, logprobs=logprobs)

    return await asyncio.gather(*[_one(i) for i in range(n)])
