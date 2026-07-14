"""Thin wrapper for running a single-turn eval-protocol eval inside a notebook.

The training notebooks and benchmarks use this so their evals go through
eval-protocol (`SingleTurnRolloutProcessor` + `execute_pytest`) instead of ad-hoc
scoring. `eval_protocol` is installed from PyPI (`pip install eval-protocol`);
see requirements.txt.

Key gotcha baked in: reasoning models (e.g. gpt-oss) need a generous `max_tokens`
so the analysis channel doesn't starve the final answer — default is 1024.

Async execution: `single_turn_eval` is a *sync* helper called from notebook cells,
so it has to drive an event loop itself. Rather than `nest_asyncio` + the notebook's
own loop (which breaks on Python 3.14 with anyio's "Timeout should be used inside a
task"), we run the coroutine on a persistent background event-loop thread. The loop
lives for the kernel session, so litellm's cached async HTTP client stays bound to a
live loop across repeated calls (e.g. eval-before then eval-after).
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Callable, List, Tuple

# Re-export the models so notebooks can `from ep_eval import EvaluationRow, ...`.
from eval_protocol.models import EvaluationRow, Message, EvaluateResult  # noqa: F401

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """A single long-lived event loop running on a daemon thread for the whole session.

    Reusing one loop (instead of asyncio.run per call, which closes the loop) keeps
    litellm's cached httpx client valid across successive evals.
    """
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, daemon=True,
                             name="ep-eval-loop").start()
        return _LOOP


def _run_coro(coro):
    """Run a coroutine to completion on the background loop from a sync caller."""
    return asyncio.run_coroutine_threadsafe(coro, _background_loop()).result()


def single_turn_eval(
    rows: List["EvaluationRow"],
    model: str,
    reward_fn: Callable,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    concurrency: int = 8,
    api_base: str = FIREWORKS_BASE_URL,
    extra_completion_params: dict | None = None,
) -> Tuple[float, List["EvaluationRow"]]:
    """Run one-turn rollouts of `model` over `rows`, score each with `reward_fn`.

    - `rows`: EvaluationRow list whose last message is the USER turn (no assistant).
    - `model`: a Fireworks model/deployment id, e.g. "accounts/fireworks/models/gpt-oss-20b"
      or "accounts/<acct>/deployments/<id>".
    - `reward_fn`: async `def f(row: EvaluationRow) -> EvaluationRow` that sets
      `row.evaluation_result` (an EvaluateResult). The rolled-out model answer is the
      last message on `row`.
    - `extra_completion_params`: extra keys merged into the OpenAI/litellm completion
      request. Use for provider-specific controls, e.g. `{"reasoning_effort": "none"}`
      to disable a hybrid-thinking model's reasoning channel (Qwen3.5 ignores the
      `/no_think` tag, so without this it emits a long "Thinking Process:" preamble that
      runs past `max_tokens` and never emits the final answer).
    Returns (mean_score, evaluated_rows).
    """
    os.environ.setdefault("EP_LLM_API_BASE", api_base)
    # litellm's fireworks provider reads FIREWORKS_AI_API_KEY; mirror our key into it.
    if os.getenv("FIREWORKS_API_KEY") and not os.getenv("FIREWORKS_AI_API_KEY"):
        os.environ["FIREWORKS_AI_API_KEY"] = os.environ["FIREWORKS_API_KEY"]

    from eval_protocol.pytest import SingleTurnRolloutProcessor
    from eval_protocol.pytest.types import RolloutProcessorConfig
    from eval_protocol.pytest.execution import execute_pytest

    async def _run():
        # Build the config (and its semaphore) inside the loop that will run it.
        cfg = RolloutProcessorConfig(
            completion_params={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                               **(extra_completion_params or {})},
            mcp_config_path="",
            semaphore=asyncio.Semaphore(concurrency),
        )
        proc = SingleTurnRolloutProcessor()
        rolled = await asyncio.gather(*proc(rows, cfg))
        return [await execute_pytest(reward_fn, processed_row=r) for r in rolled]

    evaluated = _run_coro(_run())
    scores = [r.evaluation_result.score for r in evaluated if r.evaluation_result]
    return (sum(scores) / len(scores) if scores else 0.0), evaluated


def final_text(row: "EvaluationRow") -> str:
    """The rolled-out model's answer (last assistant message content)."""
    if not row.messages:
        return ""
    return row.messages[-1].content or ""
