#!/usr/bin/env python3
"""Eval-only agentic coding benchmark using the coding_agent rollout stack.

This runs full multi-step coding-agent rollouts (read/edit/test/iterate in a
SWE-Gym runtime), then grades each produced patch in a fresh runtime with the
SWE-bench harness. Unlike ``examples/rl/coding_agent/train.py``, this script
does not run policy optimization or weight updates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logging.basicConfig(
    level=os.environ.get("EVAL_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("agentic_coding_eval")

from training.examples.rl.coding_agent.rollout import make_rollout_fn
from training.examples.rl.coding_agent.swegym_data import (
    fetch_dataset_instances,
    row_for_instance,
)
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.recipes.async_rl_loop import RolloutSetup
from training.utils import load_tokenizer


DEFAULT_BASE_MODEL = "accounts/fireworks/models/glm-5p2"
DEFAULT_TOKENIZER = "zai-org/GLM-5.1"
# Host root only -- the SDK's DeploymentSampler appends "/inference/v1/completions".
# Passing the full completions path here doubles it and 404s (misreported as
# "Deployment not ready").
DEFAULT_INFERENCE_URL = "https://api.fireworks.ai"


@dataclass
class EvalResult:
    row_id: str
    solved: bool
    applied: bool
    reward: float
    run_id: str | None
    dropped: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER)
    p.add_argument("--inference-url", default=os.environ.get("FIREWORKS_INFERENCE_URL", DEFAULT_INFERENCE_URL))
    p.add_argument("--dataset-path", default=None, help="Optional JSONL dataset path made by make_swegym_data.py")
    p.add_argument("--split", default="train", help="SWE-Gym split when --dataset-path is not provided")
    p.add_argument("--max-rows", type=int, default=5, help="Smoke-test rows to run")
    p.add_argument("--concurrency", type=int, default=1, help="Concurrent rollout runs")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=16000)
    p.add_argument("--max-seq-len", type=int, default=64000)
    p.add_argument("--timeout-seconds", type=int, default=1800, help="Per rollout timeout budget")
    p.add_argument("--output-json", default=None, help="Optional output summary JSON path")
    return p.parse_args()


def _read_jsonl(path: str, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_rows:
                break
    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_path:
        return _read_jsonl(args.dataset_path, args.max_rows)
    instances = fetch_dataset_instances(args.split)
    instances = instances[: args.max_rows]
    return [row_for_instance(instance, args.split) for instance in instances]


def build_rollout_setup(args: argparse.Namespace) -> RolloutSetup:
    api_key = os.environ["FIREWORKS_API_KEY"]
    tokenizer = load_tokenizer(args.tokenizer_model)
    seed_setup = RolloutSetup(
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer_model,
        sample_kwargs={
            "temperature": args.temperature,
            "max_tokens": args.max_completion_tokens,
            "max_seq_len": args.max_seq_len,
            "http_timeout": args.timeout_seconds,
            "logprobs": True,
            "top_p": 1.0,
            "top_k": 0,
        },
        inference_base_url=args.inference_url,
        api_key=api_key,
        model=args.base_model,
        completions_per_prompt=1,
        extras={},
    )
    sampler = build_deployment_sampler(seed_setup)
    return RolloutSetup(
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer_model,
        sample_kwargs=seed_setup.sample_kwargs,
        inference_base_url=sampler.base_url,
        api_key=api_key,
        model=sampler.model,
        completions_per_prompt=1,
        extras={},
    )


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args)
    setup = build_rollout_setup(args)
    rollout_fn = make_rollout_fn(setup)
    sem = asyncio.Semaphore(args.concurrency)

    total_rows = len(rows)

    async def run_one(i: int, row: dict[str, Any]) -> EvalResult:
        row_id = str(row.get("id") or f"row-{i}")
        image = (row.get("metadata") or {}).get("image") or row.get("image") or "?"
        try:
            async with sem:
                logger.info("[%d/%d] start row=%s image=%s", i + 1, total_rows, row_id, image)
                t0 = time.time()
                run = await asyncio.wait_for(
                    rollout_fn(row, row_index=i, sample_index=0),
                    timeout=args.timeout_seconds + 120,
                )
        except Exception as e:  # noqa: BLE001 - benchmark should continue across failures
            logger.warning("[%d/%d] row=%s FAILED: %s: %s", i + 1, total_rows, row_id, type(e).__name__, str(e)[:200])
            return EvalResult(
                row_id=row_id,
                solved=False,
                applied=False,
                reward=0.0,
                run_id=None,
                dropped=True,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        if run is None:
            logger.warning("[%d/%d] row=%s dropped (no trainable run) after %.0fs", i + 1, total_rows, row_id, time.time() - t0)
            return EvalResult(row_id=row_id, solved=False, applied=False, reward=0.0, run_id=None, dropped=True)
        md = run.metadata or {}
        reward = run.segments[0].reward if run.segments else 0.0
        logger.info(
            "[%d/%d] done row=%s solved=%s applied=%s reward=%.3f in %.0fs",
            i + 1, total_rows, row_id, bool(md.get("solved", False)), bool(md.get("applied", False)),
            float(reward), time.time() - t0,
        )
        return EvalResult(
            row_id=row_id,
            solved=bool(md.get("solved", False)),
            applied=bool(md.get("applied", False)),
            reward=float(reward),
            run_id=run.run_id,
            dropped=False,
        )

    started = time.time()
    results = await asyncio.gather(*(run_one(i, r) for i, r in enumerate(rows)))
    elapsed = time.time() - started

    kept = [r for r in results if not r.dropped]
    solved = sum(1 for r in kept if r.solved)
    applied = sum(1 for r in kept if r.applied)
    summary = {
        "model": args.base_model,
        "tokenizer_model": args.tokenizer_model,
        "inference_url": args.inference_url,
        "rows_requested": len(rows),
        "rows_completed": len(kept),
        "rows_dropped": len(results) - len(kept),
        "solve_rate": (solved / len(kept)) if kept else 0.0,
        "apply_rate": (applied / len(kept)) if kept else 0.0,
        "avg_reward": (sum(r.reward for r in kept) / len(kept)) if kept else 0.0,
        "elapsed_seconds": elapsed,
        "results": [r.__dict__ for r in results],
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.max_rows < 1:
        raise ValueError("--max-rows must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise EnvironmentError("Set FIREWORKS_API_KEY before running this benchmark.")

    summary = asyncio.run(run_eval(args))

    print("\n=== Agentic Coding Eval Summary ===")
    print(f"model:         {summary['model']}")
    print(f"rows:          {summary['rows_completed']}/{summary['rows_requested']} completed")
    print(f"dropped:       {summary['rows_dropped']}")
    print(f"solve rate:    {summary['solve_rate']:.1%}")
    print(f"apply rate:    {summary['apply_rate']:.1%}")
    print(f"avg reward:    {summary['avg_reward']:.3f}")
    print(f"elapsed:       {summary['elapsed_seconds']:.1f}s")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        print(f"wrote:         {args.output_json}")


if __name__ == "__main__":
    main()
