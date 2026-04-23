#!/usr/bin/env python3
"""rollr_cispo -> cookbook async RL example.

Shows how a customer-fork like ``rollr_cispo/train_cispo_cookbook.py``
(~1,000 LOC today) collapses onto the cookbook's one-function extension
point.  This file contains the *entire* rollr-specific integration; the
recipe / trainer / loss / gate / weight-sync all stay in the cookbook
unchanged.

The original fork lives at ``/Users/chengxili/workspace_batching/rollr_cispo/``
for reference.  The pieces that used to be entangled with training code are
now each isolated inside ``rollout_fn``:

  1. Evaluator auto-discovery           (rollr_cispo/cispo_lib.py:39-77)
  2. EvaluationRow construction         (train_cispo_cookbook.py:659-668)
  3. Rollout + remote grader call       (train_cispo_cookbook.py:673, 702)
  4. Tokenization against the policy    (train_cispo_cookbook.py:713-757)
  5. Prefill-logprob recovery via echo  (rollr_cispo/cispo_lib.py:351-396)
  6. Trajectory packaging               (below; the cookbook handles the
                                         advantage-centering + Datum-packing)

Dependencies that stay *user-side* (never imported by the cookbook):
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
from training.utils.rl.trajectory import CompletionSegment, Trajectory

# --- User-side imports (uncomment once eval-protocol + rollr are installed) ---
# from eval_protocol.models import EvaluationRow
# from eval_protocol.training.utils import build_ep_parameters_from_test
# from training.examples.rollr_cispo.evaluators import cispo_eval  # user's grader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter -- user-owned
# ---------------------------------------------------------------------------

def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (GRPO assigns zero advantage on ties)."""
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Rollr-specific helpers (stubs -- wire to your evaluator module)
# ---------------------------------------------------------------------------


async def _run_remote_agent(
    prompt_messages: list[dict],
    *,
    n: int,
    completion_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Call the Rollr / EP ``rollout_processor`` and return per-completion dicts.

    Each dict must carry at minimum ``text`` and ``reward``; returning the
    inference endpoint's logprobs here saves a prefill round-trip.

    Replace this stub with the real EP / Rollr integration -- the customer's
    original code is at train_cispo_cookbook.py:673 and :702.
    """
    raise NotImplementedError(
        "Wire this to your @evaluation_test grader.  The body should:\n"
        "  1. Build EvaluationRows from (prompt_messages, n) with "
        "rollout_id/run_id/invocation_id/experiment_id metadata\n"
        "  2. Await rollout_processor(rows, RolloutProcessorConfig(...))\n"
        "  3. Await grade_fn(row) for each completed row\n"
        "  4. Return [{'text': ..., 'reward': ..., 'logprobs': [...]?} ...]"
    )


def _tokenize_completion(
    prompt_messages: list[dict],
    completion_text: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Tokenize (prompt_messages, assistant(completion_text)) and return
    ``(prompt_tokens, completion_tokens)``.

    Equivalent to rollr's inline path at train_cispo_cookbook.py:713-757.
    The strict-prefix invariant (prompt render must be a prefix of the full
    render) is the caller's responsibility -- see ``tokenize_chat_turn``
    helper if you want a pre-built check.
    """
    prompt_ids: list[int] = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
    )
    full_ids: list[int] = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": completion_text}],
        tokenize=True, add_generation_prompt=False,
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Chat template did not produce a prefix-preserving render; "
            "the completion tokenization is unsafe."
        )
    return prompt_ids, full_ids[len(prompt_ids):]


async def _recover_logprobs(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    *,
    inference_url: str,
    api_key: str,
    model: str,
) -> list[float]:
    """Call the inference endpoint with ``echo=True, max_tokens=1`` to score
    the concatenation under the current policy and return per-completion-token
    logprobs.

    Equivalent to rollr's ``get_prefill_logprobs`` at cispo_lib.py:351-396.
    Implement with ``requests.post`` or an httpx client -- the cookbook
    intentionally does not ship this helper, since users who have native
    logprobs from their rollout source don't need it.
    """
    raise NotImplementedError(
        "Port rollr_cispo/cispo_lib.py:351 get_prefill_logprobs here.  "
        "The cookbook does not provide a default."
    )


# ---------------------------------------------------------------------------
# The one extension point
# ---------------------------------------------------------------------------


async def rollout_fn(row: dict, ctx: RolloutContext) -> Trajectory | None:
    prompt_messages = row.get("messages") or []
    if not prompt_messages:
        return None

    completion_params = {
        "temperature": ctx.sample_kwargs.get("temperature", 1.0),
        "max_tokens": ctx.sample_kwargs.get("max_tokens", 1024),
    }

    # 1-3. Remote agent produces text + reward (+ maybe native logprobs).
    try:
        completions = await _run_remote_agent(
            prompt_messages, n=ctx.completions_per_prompt,
            completion_params=completion_params,
        )
    except Exception as exc:
        logger.warning("remote agent failed: %s", exc)
        return None

    if len(completions) < ctx.completions_per_prompt:
        return None

    # 4. Tokenize against the current policy so the datums align with training.
    prompt_tokens: list[int] | None = None
    segments: list[list[CompletionSegment]] = []
    rewards: list[float] = []
    version = ctx.current_version()

    for comp in completions[: ctx.completions_per_prompt]:
        p_tokens, c_tokens = _tokenize_completion(
            prompt_messages, comp["text"], ctx.tokenizer,
        )
        if prompt_tokens is None:
            prompt_tokens = p_tokens

        # 5. Logprobs: native if the agent surfaced them, otherwise recover.
        lp = comp.get("logprobs")
        if lp is None:
            lp = await _recover_logprobs(
                p_tokens, c_tokens,
                inference_url=ctx.inference_url, api_key=ctx.api_key,
                model=ctx.model,
            )
        if len(lp) != len(c_tokens):
            logger.warning("skipping completion: logprob length mismatch")
            continue

        segments.append([CompletionSegment(
            tokens=c_tokens,
            inference_logprobs=list(lp),
            version=version,
            text=comp["text"],
        )])
        rewards.append(float(comp["reward"]))

    if not segments or prompt_tokens is None:
        return None

    # 6. Trajectory -- the cookbook packages it into a PromptGroup
    #    (advantage centering, Datum construction, reference datums).
    return Trajectory(
        prompt_tokens=prompt_tokens,
        completions=segments,
        rewards=rewards,
        prompt_messages=prompt_messages,
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
