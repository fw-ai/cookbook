#!/usr/bin/env python3
"""rollr_cispo -> cookbook async RL example.

Shows how a remote-grader / agent-framework integration maps onto the
cookbook's flat :class:`Rollout` contract.  The multi-turn shape matters
here: a typical Rollr episode interleaves assistant turns with tool
results or grader feedback.  The flat contract handles this naturally --
``loss_mask`` is ``1`` on the tokens the assistant generated and ``0``
everywhere else (prompt, tool output, grader feedback).  The cookbook's
existing loss kernels already respect per-token ``loss_mask`` (see
``training/utils/rl/common.py::_get_loss_mask``), so no trainer-side
changes are needed.

Dependencies that stay user-side (the cookbook never imports them):
    pip install eval-protocol rollr

Run::

    export FIREWORKS_API_KEY=...
    python -m training.examples.rollr_cispo.train
"""

from __future__ import annotations

import logging
from typing import Any

from training.recipes.async_rl_loop import Config, RolloutContext, main
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import Rollout, RolloutSample

# --- User-side imports (uncomment once eval-protocol + rollr are installed) ---
# from eval_protocol.models import EvaluationRow
# from eval_protocol.training.utils import build_ep_parameters_from_test
# from training.examples.rollr_cispo.evaluators import cispo_eval  # user's grader

logger = logging.getLogger(__name__)


def should_accept(pg: PromptGroup) -> bool:
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Rollr-specific helpers (stubs -- wire to your evaluator module)
# ---------------------------------------------------------------------------


async def _run_remote_episode(
    row: dict, *, n: int, completion_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Call the Rollr / EP ``rollout_processor`` and return per-completion
    records.  Each record must provide the flat arrays required for training:

        {
            "tokens":     list[int],   # prompt + turns + env feedback, flat
            "logprobs":   list[float], # aligned with tokens; 0.0 on non-gen
            "loss_mask":  list[int],   # 1 on assistant-gen, 0 elsewhere
            "reward":     float,
            "text":       str,         # for logging
            "finish_reason": str,
        }

    Replace this stub with the real EP / Rollr integration -- the
    customer's original code is at
    ``rollr_cispo/train_cispo_cookbook.py:657-783`` and
    ``rollr_cispo/cispo_lib.py:351``.  The flat contract means the helper
    that used to live there (``get_prefill_logprobs`` for text-only
    rollouts) belongs here, not in the cookbook.
    """
    raise NotImplementedError(
        "Wire this to your @evaluation_test grader.  The body should:\n"
        "  1. Build EvaluationRows from (row, n) with "
        "rollout_id/run_id/invocation_id/experiment_id metadata\n"
        "  2. Await rollout_processor(rows, RolloutProcessorConfig(...))\n"
        "  3. Await grade_fn(row) for each completed row\n"
        "  4. Tokenize the full multi-turn conversation and build a "
        "per-token loss_mask (1 on assistant-generated turns, 0 on "
        "prompt / user / tool messages).\n"
        "  5. If the remote agent returned text only, call the inference "
        "endpoint with echo=True to recover per-token logprobs.\n"
        "  6. Return per-completion dicts as documented above."
    )


# ---------------------------------------------------------------------------
# The one extension point
# ---------------------------------------------------------------------------


async def rollout_fn(row: dict, ctx: RolloutContext) -> Rollout | None:
    completion_params = {
        "temperature": ctx.sample_kwargs.get("temperature", 1.0),
        "max_tokens": ctx.sample_kwargs.get("max_tokens", 1024),
    }

    try:
        records = await _run_remote_episode(
            row, n=ctx.completions_per_prompt, completion_params=completion_params,
        )
    except Exception as exc:
        logger.warning("remote episode failed: %s", exc)
        return None

    if len(records) < ctx.completions_per_prompt:
        return None

    version = ctx.current_version()
    samples: list[RolloutSample] = []
    for rec in records[: ctx.completions_per_prompt]:
        tokens = list(rec["tokens"])
        logprobs = list(rec["logprobs"])
        loss_mask = list(rec["loss_mask"])
        if not (len(tokens) == len(logprobs) == len(loss_mask)):
            logger.warning("record length mismatch; dropping row")
            return None
        samples.append(RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=float(rec["reward"]),
            versions=[version] * len(tokens),
            finish_reason=rec.get("finish_reason", "stop"),
            text=rec.get("text", ""),
        ))

    return Rollout(
        samples=samples,
        row_meta={"ground_truth": row.get("ground_truth", "")},
    )


if __name__ == "__main__":
    cfg = Config(
        log_path="/tmp/rl-async-cispo",
        base_model="accounts/fireworks/models/kimi-k2p5-text-only-256k-lora",
        dataset="/path/to/cispo_dataset.jsonl",
        policy_loss="cispo",
        prompt_groups_per_step=4,
        max_head_offpolicy_versions=2,
    )
    main(cfg, rollout_fn=rollout_fn, dynamic_filter_fn=should_accept)
