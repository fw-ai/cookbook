#!/usr/bin/env python3
"""Run a few rollouts on qwen3-8b serverless and write eval-protocol JSONL for visualization.

Usage:
    python verify_rollout.py [--num-rollouts 4] [--port 8765]

After it finishes, the vite dashboard opens automatically at http://localhost:<port>/table
with token-level debug data in execution_metadata.extra.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FROZEN_LAKE_DIR = Path(__file__).resolve().parent
COOKBOOK_DIR = FROZEN_LAKE_DIR.parent.parent
sys.path.insert(0, str(COOKBOOK_DIR))

from eval_protocol.integrations.frozen_lake_tool_rollout_processor import FrozenLakeToolRolloutProcessor
from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Status
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.dataset_logger import default_logger

DEFAULT_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are an RL policy for FrozenLake.\n"
    "Pick the action that moves toward G while avoiding H.\n"
    "Always respond with exactly one tool call, no text."
)


def load_seed_contexts(path: str, max_seeds: int = 4):
    contexts = []
    with open(path) as f:
        for line in f:
            ctx = json.loads(line)
            contexts.append(ctx)
            if len(contexts) >= max_seeds:
                break
    return contexts


async def run_rollouts(num_rollouts: int = 4):
    api_key = os.environ["FIREWORKS_API_KEY"]
    model_id = "accounts/fireworks/models/qwen3-8b"
    tokenizer = "Qwen/Qwen3-8B"

    seed_path = str(FROZEN_LAKE_DIR / "seeds.jsonl")
    seed_contexts = load_seed_contexts(seed_path, max_seeds=num_rollouts)
    logger.info("Loaded %d seeds", len(seed_contexts))

    processor = FrozenLakeToolRolloutProcessor(
        model_id=model_id,
        tokenizer_name_or_path=tokenizer,
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference",
        temperature=1.0,
        max_tokens=128,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        logprobs=True,
    )

    config = RolloutProcessorConfig(
        completion_params={"model": model_id},
        mcp_config_path="",
        steps=30,
        semaphore=asyncio.Semaphore(8),
    )

    all_results = []
    for ctx in seed_contexts:
        rows = [
            EvaluationRow(
                input_metadata=InputMetadata(
                    row_id=f"verify_seed_{ctx.get('seed', 0)}_{i}",
                    dataset_info={"environment_context": dict(ctx)},
                ),
            )
            for i in range(2)
        ]

        tasks = processor(rows, config)
        for task in tasks:
            try:
                result = await task
                extra = result.execution_metadata.extra or {}
                if extra.get("rollout_error"):
                    logger.warning("Rollout error for seed %s: %s", ctx.get("seed"), extra["rollout_error"])
                all_results.append(result)
            except Exception as e:
                logger.warning("Rollout failed for seed %s: %s", ctx.get("seed"), e)

    logger.info("Completed %d rollouts", len(all_results))
    return all_results


def enrich_rows(results: list[EvaluationRow], tokenizer_name: str):
    """Enrich rows with detokenized tokens, full-episode view, status, and score."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    for row in results:
        extra = row.execution_metadata.extra
        if not isinstance(extra, dict):
            continue
        token_turn_traces = extra.get("token_turn_traces") or []
        step_rewards = extra.get("step_rewards", [])
        episode_reward = 1.0 if step_rewards and float(step_rewards[-1]) > 0 else 0.0
        extra["episode_reward"] = episode_reward

        # Per-turn detokenization
        for trace in token_turn_traces:
            prompt_ids = trace.get("prompt_ids") or []
            completion_ids = trace.get("completion_ids") or []
            all_ids = [int(x) for x in prompt_ids] + [int(x) for x in completion_ids]
            detok_tokens = [_detok(tokenizer, tid) for tid in all_ids]
            trace["detokenized_tokens"] = detok_tokens
            trace["prompt_len"] = len(prompt_ids)
            trace["completion_len"] = len(completion_ids)

        # Build full-episode view matching the trainer's corrected masking.
        #
        # Only model-generated tokens (assistant completions) get gradients.
        # Environment tokens (tool responses, chat template separators) are
        # masked even though they appear after first_prompt_len.
        #
        # For each turn k, model output in full_tokens spans:
        #   [prompt_start[k], prompt_start[k] + assistant_turn_len[k])  (intermediate)
        #   [prompt_start[-1], prompt_start[-1] + len(completion_ids[-1]))  (last turn)
        if token_turn_traces:
            last = token_turn_traces[-1]
            full_ids = [int(x) for x in last["prompt_ids"]] + [int(x) for x in last["completion_ids"]]
            full_len = len(full_ids)

            first_prompt_len = len(token_turn_traces[0].get("prompt_ids") or [])
            mrt = extra.get("model_request_traces") or []
            num_turns = len(token_turn_traces)

            # mask: 0 = masked (prompt/env), >0 = turn index (model output)
            mask = [0] * full_len
            logprobs_arr: list[float | None] = [None] * full_len

            for k in range(num_turns):
                turn_prompt_len = len(token_turn_traces[k].get("prompt_ids") or [])
                if k < num_turns - 1:
                    mrt_k = mrt[k] if k < len(mrt) else {}
                    model_output_len = int(mrt_k.get("assistant_turn_len") or 0)
                else:
                    model_output_len = len(token_turn_traces[k].get("completion_ids") or [])
                if model_output_len == 0:
                    model_output_len = len(token_turn_traces[k].get("completion_ids") or [])

                turn_idx = k + 1
                for j in range(model_output_len):
                    pos = turn_prompt_len + j
                    if 0 <= pos < full_len:
                        mask[pos] = turn_idx

                turn_lp = (token_turn_traces[k].get("completion_logprobs") or []) if k < num_turns else []
                for j, lp in enumerate(turn_lp):
                    pos = turn_prompt_len + j
                    if pos < full_len:
                        logprobs_arr[pos] = lp

            full_detok = [_detok(tokenizer, tid) for tid in full_ids]

            extra["full_episode"] = {
                "token_ids": full_ids,
                "mask": mask,
                "logprobs": logprobs_arr,
                "detokenized_tokens": full_detok,
                "num_turns": num_turns,
                "first_prompt_len": first_prompt_len,
            }

        # Set rollout_status and evaluation_result
        has_error = bool(extra.get("rollout_error"))
        if has_error:
            row.rollout_status = Status.rollout_internal_error(extra["rollout_error"])
        else:
            row.rollout_status = Status.rollout_finished()

        row.evaluation_result = EvaluateResult(
            score=episode_reward,
            is_score_valid=not has_error,
            reason=extra.get("rollout_error") or f"episode_reward={episode_reward}",
        )


def _detok(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)])
    except Exception:
        return f"<{tid}>"


def write_to_default_logger(results: list[EvaluationRow]):
    """Write results to the default eval-protocol logger so serve_logs picks them up."""
    for row in results:
        default_logger.log(row)
    logger.info("Wrote %d rows to default logger (sqlite)", len(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-serve", action="store_true")
    args = parser.parse_args()

    results = asyncio.run(run_rollouts(num_rollouts=args.num_rollouts))
    if not results:
        logger.error("No rollouts completed")
        return

    logger.info("Enriching rows...")
    enrich_rows(results, "Qwen/Qwen3-8B")

    write_to_default_logger(results)

    if args.no_serve:
        logger.info("Done (--no-serve). Data written to default logger.")
        return

    logger.info("Starting eval-protocol dashboard on port %d...", args.port)
    from eval_protocol.utils.logs_server import serve_logs
    serve_logs(port=args.port)


if __name__ == "__main__":
    main()
