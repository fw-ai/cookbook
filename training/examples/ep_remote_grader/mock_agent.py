"""Mock remote agent service.

Stands in for any production agent framework: a chain-of-tools pipeline,
a RAG-with-verifier stack, an LLM-as-judge loop.  Returns ``n`` plausible
completion strings concurrently so the example behaves like a real
networked service -- which is what a typical remote-grader integration
has to deal with (backpressure, concurrent grading across the group,
transient failures).

A real integration replaces this with an HTTP / gRPC / SDK call.  The
shape stays identical: ``messages`` in, ``list[str]`` out of length
``n``.
"""

from __future__ import annotations

import asyncio
import random


async def remote_agent_complete(
    messages: list[dict],
    *,
    n: int,
    completion_params: dict,
) -> list[str]:
    """Return ``n`` completion strings for ``messages``.

    Simulated latency is ~10ms per completion, so a group of 4 completes
    concurrently in ~10ms rather than ~40ms.  That matches what a real
    remote agent will look like: each slot fires in parallel.
    """
    seed = hash(str(messages)) & 0xFFFFFFFF
    rng = random.Random(seed)

    async def _one(idx: int) -> str:
        await asyncio.sleep(0.01)
        # Produce a plausible <answer>NNN</answer> payload.  A real agent
        # would return multi-paragraph reasoning followed by the answer.
        return (
            f"Let me work through this step by step.\n"
            f"(variant {idx + 1})\n"
            f"<answer>{rng.randint(0, 100)}</answer>"
        )

    return await asyncio.gather(*[_one(i) for i in range(n)])
