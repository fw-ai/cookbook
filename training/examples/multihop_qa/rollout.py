"""Multi-hop QA rollout factory for the async RL recipe.

Wraps the existing :class:`MultiHopQARolloutProcessor` (eval-protocol
based, multi-turn search + submit_answer tool loop) into the
``rollout_fn(sample_prompt) -> RolloutSample | None`` contract that
:func:`training.recipes.async_rl_loop.main` expects.

IGPO turn-level Information-Gain scoring is preserved: the rollout fires
per-turn IG futures via :class:`IGPOTurnScorer.on_turn_complete` (running
in parallel with sampling), then folds them into the trajectory's scalar
reward as ``outcome + ig_weight * sum(ig_per_turn)``.

Trade-off vs. the legacy ``train_multihop_qa_igpo.py`` recipe: per-turn
credit assignment is collapsed into a single trajectory-level scalar so
the GRPO advantage is computed on the IG-augmented reward.  This is
strictly compatible with :class:`RolloutSample`'s flat contract; the
async loop's z-score normalisation across the prompt group still
propagates the IG signal, just without per-token weighting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.multihop_qa.multihop_qa_rollout import (
    MultiHopQARolloutProcessor,
)
from training.examples.rl.frozen_lake.masking import (
    build_ui_token_mask,
    compute_model_output_spans,
)
from training.utils.rl.igpo import IGPOTurnScorer
from training.utils.rl.rollout import RolloutSample

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question by searching for "
    "relevant information. You have access to two tools:\n"
    "- search(query): Search for information about a topic. Returns "
    "relevant paragraphs.\n"
    "- submit_answer(answer): Submit your final answer.\n\n"
    "Search as many times as needed, then submit your answer. "
    "Always respond with exactly one tool call and no additional text."
)


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    """Build the per-call rollout function for async_rl_loop.

    Reads IGPO knobs from ``setup.extras`` so the train script stays in
    charge of all tunables:

    * ``ig_weight`` (float, default 1.0) -- folds turn-level IG into the
      scalar reward.  ``0`` disables IG scoring (pure GRPO baseline).
    * ``skip_ig_last_turn`` (bool, default True) -- zero out IG on the
      ``submit_answer`` turn (already covered by the env reward).
    * ``max_steps`` (int, default 8) -- max search/submit turns per
      trajectory.
    * ``search_top_k`` (int, default 2) -- paragraphs per search.
    * ``scoring_workers`` (int, default 4) -- IG scoring threadpool size.
    * ``enable_thinking`` (bool, default False) -- pass through to the
      Qwen3 chat template.
    * ``system_prompt`` (str, optional) -- override the default tool-use
      system prompt.
    """

    extras = setup.extras
    ig_weight = float(extras.get("ig_weight", 1.0))
    skip_ig_last_turn = bool(extras.get("skip_ig_last_turn", True))
    max_steps = int(extras.get("max_steps", 8))
    search_top_k = int(extras.get("search_top_k", 2))
    scoring_workers = int(extras.get("scoring_workers", 4))
    enable_thinking = bool(extras.get("enable_thinking", False))
    system_prompt = str(extras.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)

    api_key = setup.api_key
    inference_base_url = setup.inference_base_url.rstrip("/") + "/inference"
    inference_model = setup.model
    tokenizer = setup.tokenizer
    tokenizer_id = setup.tokenizer_id

    sample_kwargs = setup.sample_kwargs
    temperature = float(sample_kwargs.get("temperature", 1.0))
    max_tokens = int(sample_kwargs.get("max_tokens", 1024))

    scoring_executor = ThreadPoolExecutor(
        max_workers=scoring_workers,
        thread_name_prefix="igpo-scorer",
    )

    processor = MultiHopQARolloutProcessor(
        model_id=inference_model,
        tokenizer_name_or_path=tokenizer_id,
        api_key=api_key,
        base_url=inference_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        logprobs=True,
        enable_thinking=enable_thinking,
        search_top_k=search_top_k,
    )

    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        ground_truth = str(sample_prompt.get("ground_truth", ""))
        if not ground_truth:
            return None
        answer_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
        if not answer_tokens:
            return None

        context = sample_prompt.get("context") or {}
        messages_raw = sample_prompt.get("messages") or []
        question = ""
        for m in messages_raw:
            if m.get("role") == "user":
                question = m.get("content", "")
                break

        row_id = (
            f"q_{abs(hash(question)) % 100000}_"
            f"{int(time.time() * 1000) & 0xffff}_{id(sample_prompt) & 0xffff}"
        )

        scorer = IGPOTurnScorer(
            answer_tokens=answer_tokens,
            executor=scoring_executor,
            ig_weight=ig_weight,
            skip_ig_last_turn=skip_ig_last_turn,
            inference_url=inference_base_url,
            model_id=inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
        )
        scorer._turn_futs[row_id] = []

        eval_row = EvaluationRow(
            input_metadata=InputMetadata(
                row_id=row_id,
                dataset_info={
                    "context": context,
                    "ground_truth": ground_truth,
                    "question": question,
                },
            ),
            messages=[
                Message(role=m["role"], content=m["content"])
                for m in messages_raw
            ],
        )

        rollout_config = RolloutProcessorConfig(
            completion_params={"model": inference_model},
            mcp_config_path="",
            steps=max_steps,
            semaphore=asyncio.Semaphore(1),
            kwargs={"turn_callback": scorer.on_turn_complete},
        )

        tasks = processor([eval_row], rollout_config)
        try:
            result = await tasks[0]
        except Exception as e:
            logger.warning(
                "Multi-hop QA rollout failed (q=%r): %s", question[:60], e,
            )
            return None

        extra = result.execution_metadata.extra or {}
        if extra.get("rollout_error"):
            logger.warning("Rollout error: %s", extra["rollout_error"])
            return None

        token_turn_traces = extra.get("token_turn_traces") or []
        model_request_traces = extra.get("model_request_traces") or []
        step_rewards = extra.get("step_rewards") or []
        if not token_turn_traces or not step_rewards:
            return None

        last_trace = token_turn_traces[-1]
        last_prompt_ids = [int(x) for x in (last_trace.get("prompt_ids") or [])]
        last_completion_ids = [
            int(x) for x in (last_trace.get("completion_ids") or [])
        ]
        full_tokens = last_prompt_ids + last_completion_ids
        if len(full_tokens) < 2:
            return None

        if ig_weight > 0.0:
            first_prompt_ids = [
                int(x) for x in (token_turn_traces[0].get("prompt_ids") or [])
            ]
            if first_prompt_ids:
                scorer.on_rollout_start(row_id, first_prompt_ids)
            try:
                ig_rewards, _outcome = await asyncio.to_thread(
                    scorer.collect_rewards, row_id, step_rewards,
                )
            except Exception as e:
                logger.warning("IG scoring failed: %s", e)
                ig_rewards = [0.0] * len(step_rewards)
        else:
            ig_rewards = [0.0] * len(step_rewards)

        spans = compute_model_output_spans(
            token_turn_traces, model_request_traces,
        )
        token_mask = build_ui_token_mask(spans, len(full_tokens))
        loss_mask = [1 if m > 0 else 0 for m in token_mask]
        if not any(loss_mask):
            return None

        logprobs: List[float] = [0.0] * len(full_tokens)
        for trace in token_turn_traces:
            turn_prompt_len = len(trace.get("prompt_ids") or [])
            turn_completion_logprobs = trace.get("completion_logprobs") or []
            for i, lp in enumerate(turn_completion_logprobs):
                pos = turn_prompt_len + i
                if pos < len(full_tokens):
                    logprobs[pos] = float(lp)

        outcome_reward = float(step_rewards[-1]) if step_rewards else 0.0
        ig_sum = float(sum(ig_rewards))
        final_reward = outcome_reward + ig_weight * ig_sum

        return RolloutSample(
            tokens=full_tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=final_reward,
            finish_reason=str(result.execution_metadata.finish_reason or "stop"),
        )

    return rollout_fn
