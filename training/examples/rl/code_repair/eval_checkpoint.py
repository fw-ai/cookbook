#!/usr/bin/env python3
"""Score the base model or a saved sampler checkpoint on the code-repair eval rows.

Samples through a serverless sampling client (no training run, no slot,
billed per sampled token), grades each reply in the local sandbox exactly like
training does (a reply cut off at ``--max-tokens`` scores 0), and writes
``<run-dir>/sampled_rows.jsonl`` with per-sample rewards and ``finished``
flags plus ``summary.json``. ``compare_runs`` reads those files.

    # tuned adapter at the training cap
    python -m training.examples.rl.code_repair.eval_checkpoint \\
        --checkpoint pyroworks/run-.../code-repair-step-9-... --run-dir .../runs/eval_tuned_8k
    # base model with room to finish its reasoning
    python -m training.examples.rl.code_repair.eval_checkpoint \\
        --max-tokens 32000 --run-dir .../runs/eval_base_32k
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from training.examples.rl.code_repair.prepare_data import DATA_DIR
from training.examples.rl.code_repair.replies import split_reply
from training.examples.rl.code_repair.rewards import extract_code
from training.examples.rl.code_repair.sandbox import grade

MAX_SEQ_LEN = 32768


async def score_sample(sampler: Any, renderer: Any, prompt: Any, params: Any, row: dict) -> dict[str, Any]:
    response = await asyncio.to_thread(
        lambda: sampler.sample(prompt, num_samples=1, sampling_params=params).result(timeout=3600)
    )
    sequence = response.sequences[0]
    tokens = list(sequence.tokens)
    finished = "length" not in str(sequence.stop_reason).lower()
    gt = row["ground_truth"]
    reward = 0.0
    if finished:
        _, answer = split_reply(renderer, tokens)
        code = extract_code(answer, gt["entry_point"])
        if code is not None:
            result = await grade(code, gt["test"], gt["entry_point"])
            reward = min(1.0, result.passed / gt["num_cases"])
    return {"reward": round(reward, 4), "finished": finished, "tokens": len(tokens)}


async def evaluate(args: argparse.Namespace) -> None:
    from fireworks.training.sdk import FiretitanServiceClient, types

    from training.recipes.experiment.async_rl_loop_serverless import load_tokenizer
    from training.utils import load_jsonl_dataset
    from training.utils.supervised import build_renderer

    rows = load_jsonl_dataset(args.rows)[: args.limit]
    tokenizer = load_tokenizer(args.tokenizer_model, None)
    renderer = build_renderer(tokenizer, args.tokenizer_model, "")
    service = FiretitanServiceClient(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/training/v1/serverless"
    )
    if args.checkpoint:
        sampler = service.create_sampling_client(model_path=args.checkpoint, tokenizer=tokenizer)
    else:
        sampler = service.create_sampling_client(base_model=args.base_model, tokenizer=tokenizer)
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0

    async def one(row: dict) -> dict[str, Any]:
        nonlocal failures
        prompt = renderer.build_generation_prompt(row["messages"])
        params = types.SamplingParams(
            temperature=args.temperature,
            max_tokens=min(args.max_tokens, MAX_SEQ_LEN - prompt.length),
            stop=renderer.get_stop_sequences(),
        )

        async def guarded():
            nonlocal failures
            async with semaphore:
                try:
                    return await score_sample(sampler, renderer, prompt, params, row)
                except Exception as err:  # noqa: BLE001 - a failed request is reported, not scored
                    failures += 1
                    print(f"{row['id']}: {type(err).__name__}: {err}", flush=True)
                    return None

        samples = [s for s in await asyncio.gather(*(guarded() for _ in range(args.samples))) if s]
        return {
            "id": row["id"],
            "buggy_pass_rate": row["ground_truth"]["buggy_pass_rate"],
            "rewards": [s["reward"] for s in samples],
            "finished": [s["finished"] for s in samples],
            "completion_tokens": [s["tokens"] for s in samples],
        }

    started = time.monotonic()
    try:
        per_row = await asyncio.gather(*(one(row) for row in rows))
    finally:
        sampler.close()
        service.close()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "sampled_rows.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in per_row)
    rewards = [x for r in per_row for x in r["rewards"]]
    finished = [x for r in per_row for x in r["finished"]]
    tokens = [x for r in per_row for x in r["completion_tokens"]]
    summary = {
        "model": args.checkpoint or args.base_model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "rows": len(per_row),
        "samples": len(rewards),
        "failed_requests": failures,
        "fully_fixed": sum(x >= 1.0 for x in rewards) / len(rewards) if rewards else 0.0,
        "pass_rate": sum(rewards) / len(rewards) if rewards else 0.0,
        "truncation_rate": 1 - sum(finished) / len(finished) if finished else 0.0,
        "completion_tokens_median": statistics.median(tokens) if tokens else 0,
        "sampled_tokens": sum(tokens),
        "wall_minutes": round((time.monotonic() - started) / 60, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="Sampler checkpoint path; omit to score the base model.")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3p8-27b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--rows", default=str(DATA_DIR / "eval.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--run-dir", required=True)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
