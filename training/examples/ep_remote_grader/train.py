#!/usr/bin/env python3
"""Remote-grader async RL via Eval Protocol.

End-to-end example of the cookbook's :func:`training.recipes.async_rl_loop.main`
wired to:

  * An :mod:`eval_protocol`-decorated grader (see :mod:`grader`).
  * A mock remote agent service (see :mod:`mock_agent`) that stands in
    for any external completion source -- agent framework, RAG pipeline,
    LLM judge, multi-turn tool-use loop.

The rollout_fn shape generalises beyond GSM8K: any task whose rollouts
come from a text-only external service follows the same three steps --
call the service, grade the completions concurrently, and package into
:class:`~training.utils.rl.rollout.Rollout` with explicit tokens +
``loss_mask`` + per-token logprobs.

``eval_protocol`` is a user-side dependency (``pip install eval-protocol``);
the cookbook does not import it anywhere else.

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.ep_remote_grader.train
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests
from eval_protocol import EvaluationRow, Message

from training.examples.ep_remote_grader.grader import test_math_answer_eval
from training.examples.ep_remote_grader.mock_agent import remote_agent_complete
from training.recipes.async_rl_loop import Config, RolloutContext, main
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import Rollout, RolloutSample

logger = logging.getLogger(__name__)


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Grading via EP (calls the @evaluation_test coroutine directly)
# ---------------------------------------------------------------------------


async def _grade(
    prompt_messages: list[dict],
    completion_text: str,
    ground_truth: str,
) -> float:
    """Wrap ``(prompt + assistant)`` into an :class:`EvaluationRow`, run
    the EP-decorated grader, return its scalar score."""
    row = EvaluationRow(
        messages=[Message(**m) for m in prompt_messages]
        + [Message(role="assistant", content=completion_text)],
        ground_truth=ground_truth,
    )
    graded = await test_math_answer_eval(row)
    if graded.evaluation_result is None:
        return 0.0
    return float(graded.evaluation_result.score)


# ---------------------------------------------------------------------------
# Logprob recovery for text-only rollouts (echo=True prefill)
# ---------------------------------------------------------------------------


def _normalize_completions_url(inference_url: str) -> str:
    url = inference_url.rstrip("/")
    for suffix in ("/inference/v1", "/inference"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return f"{url}/inference/v1/completions"


def _recover_logprobs(
    tokens: list[int],
    *,
    inference_url: str,
    api_key: str,
    model: str,
) -> list[float]:
    """Score ``tokens`` under the current policy via ``echo=True, max_tokens=1``.

    Returns a list of length ``len(tokens)`` with ``0.0`` for the first
    position (no prior context to score it against) and per-token
    logprobs for positions 1..N.
    """
    resp = requests.post(
        _normalize_completions_url(inference_url),
        json={
            "model": model,
            "prompt": tokens,
            "echo": True,
            "max_tokens": 1,
            "logprobs": 0,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )
    resp.raise_for_status()
    token_lp = resp.json()["choices"][0]["logprobs"]["token_logprobs"]
    out: list[float] = [0.0 if x is None else float(x) for x in token_lp]
    if len(out) < len(tokens):
        out = out + [0.0] * (len(tokens) - len(out))
    return out[: len(tokens)]


# ---------------------------------------------------------------------------
# The one extension point
# ---------------------------------------------------------------------------


async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
    prompt_messages = row.get("messages") or []
    if not prompt_messages:
        return None

    # 1. Remote agent produces N completion texts (no tokens, no logprobs).
    try:
        completions = await remote_agent_complete(
            prompt_messages,
            n=ctx.completions_per_prompt,
            completion_params={
                "temperature": ctx.sample_kwargs.get("temperature", 1.0),
                "max_tokens": ctx.sample_kwargs.get("max_tokens", 1024),
            },
        )
    except Exception as exc:
        logger.warning("remote agent failed: %s", exc)
        return None

    # 2. Grade concurrently -- don't serialize graders across the group.
    ground_truth = str(row.get("ground_truth", ""))
    try:
        rewards = await asyncio.gather(
            *(_grade(prompt_messages, c, ground_truth) for c in completions),
        )
    except Exception as exc:
        logger.warning("grader failed: %s", exc)
        return None

    # 3. Tokenize against the current policy; recover logprobs via echo.
    version = ctx.current_version()
    prompt_ids: list[int] = ctx.tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
    )

    samples: list[RolloutSample] = []
    for text, reward in zip(completions, rewards):
        full_ids: list[int] = ctx.tokenizer.apply_chat_template(
            [*prompt_messages, {"role": "assistant", "content": text}],
            tokenize=True, add_generation_prompt=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            logger.warning("chat template not prefix-preserving; dropping")
            return None

        prompt_len = len(prompt_ids)
        comp_len = len(full_ids) - prompt_len
        if comp_len <= 0:
            return None

        try:
            logprobs = _recover_logprobs(
                full_ids,
                inference_url=ctx.inference_url,
                api_key=ctx.api_key,
                model=ctx.model,
            )
        except Exception as exc:
            logger.warning("logprob recovery failed: %s", exc)
            return None

        loss_mask = [0] * prompt_len + [1] * comp_len

        samples.append(
            RolloutSample(
                tokens=full_ids,
                logprobs=logprobs,
                loss_mask=loss_mask,
                reward=float(reward),
                versions=[version] * len(full_ids),
                finish_reason="stop",
                text=text,
            )
        )

    return Rollout(
        samples=samples,
        row_meta={"ground_truth": ground_truth},
    )


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-ep",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=(
            "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
            "main/development/gsm8k_sample.jsonl"
        ),
        prompt_groups_per_step=2,
        max_head_offpolicy_versions=1,
        completions_per_prompt=4,
    )
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept)
