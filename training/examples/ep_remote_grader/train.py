#!/usr/bin/env python3
"""Remote-grader async RL example via Eval Protocol.

Shows how to plug any async reward source -- an LLM judge, a verifier
service, an @evaluation_test grader -- into the cookbook's async RL
recipe as a single ``rollout_fn``.  This example targets
`eval_protocol <https://github.com/eval-protocol/python-sdk>`_ as the
grader; the same shape works for any framework whose grading is an
``async def score(text, row) -> float`` call.

The key multi-turn pattern: the flat :class:`Rollout` contract expresses
tool use / env feedback via interleaved ``loss_mask`` runs.  Prompt
tokens, user replies, and tool responses all get ``loss_mask = 0``;
assistant-generated tokens get ``loss_mask = 1``.  The cookbook's loss
kernels already respect per-token ``loss_mask`` (see
``training/utils/rl/common.py::_get_loss_mask``), so multi-turn
"just works" with no trainer-side changes.

Dependencies that stay user-side (the cookbook never imports them)::

    pip install eval-protocol

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.ep_remote_grader.train
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from training.recipes.async_rl_loop import Config, RolloutContext, main
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import Rollout, RolloutSample

# User-side imports (uncomment once eval_protocol is installed):
# from eval_protocol.models import EvaluationRow
# from eval_protocol.pytest.rollout_processor import RolloutProcessorConfig

logger = logging.getLogger(__name__)


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Grader integration -- user-owned
# ---------------------------------------------------------------------------


async def _ep_grade(completion_text: str, row: dict) -> float:
    """Stub: call your @evaluation_test grader and return a scalar reward.

    Typical body::

        from your_project.evaluators.math_grader import grade_math
        r = EvaluationRow(messages=row["messages"] + [
            {"role": "assistant", "content": completion_text},
        ])
        result = await grade_math(r)
        return float(result.evaluation_result.score)
    """
    raise NotImplementedError(
        "Wire this to your grader.  The body should async-call your "
        "verifier / LLM judge / evaluation_test and return a float reward."
    )


# ---------------------------------------------------------------------------
# Rollout -- the one extension point
# ---------------------------------------------------------------------------


async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
    messages = row.get("messages") or []
    if not messages:
        return None

    try:
        sampled = await ctx.sampler.sample_with_tokens(
            messages=messages, n=ctx.completions_per_prompt, **ctx.sample_kwargs,
        )
    except Exception as exc:
        logger.warning("sample_with_tokens failed: %s", exc)
        return None

    if not sampled or len(sampled) < ctx.completions_per_prompt:
        return None

    # Grade N completions concurrently -- don't serialize graders.
    try:
        rewards = await asyncio.gather(*(
            _ep_grade(s.text, row) for s in sampled
        ))
    except Exception as exc:
        logger.warning("grader failed: %s", exc)
        return None

    version = ctx.current_version()
    samples: list[RolloutSample] = []
    for s, reward in zip(sampled, rewards):
        tokens = list(s.full_tokens)
        prompt_len = s.prompt_len
        comp_len = len(tokens) - prompt_len
        if comp_len <= 0:
            return None

        inf_lp = list(s.inference_logprobs or [])
        if len(inf_lp) != comp_len:
            logger.warning(
                "logprob length %d != completion length %d",
                len(inf_lp), comp_len,
            )
            return None

        # Flat contract: prompt -> loss_mask=0, assistant -> loss_mask=1.
        # For multi-turn with tool results interleaved, interleave 0-runs
        # for the tool/user tokens; the adapter passes the mask straight
        # to the loss kernel.
        logprobs = [0.0] * prompt_len + inf_lp
        loss_mask = [0] * prompt_len + [1] * comp_len

        samples.append(RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=float(reward),
            versions=[version] * len(tokens),
            finish_reason=s.finish_reason,
            text=s.text,
        ))

    return Rollout(
        samples=samples,
        row_meta={"ground_truth": row.get("ground_truth", "")},
    )


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-ep",
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset="/path/to/your_dataset.jsonl",
        prompt_groups_per_step=4,
        max_head_offpolicy_versions=2,
    )
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept)
