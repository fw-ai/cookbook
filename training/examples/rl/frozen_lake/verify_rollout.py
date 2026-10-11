#!/usr/bin/env python3
"""Play FrozenLake episodes against a serving model and check they pack for training.

Runs the rollout function ``train_frozen_lake.py`` trains on, without a
trainer, and packs every episode into its training sample: each prompt must
extend the previous prompt and completion, with one sampling logprob per
generated token. A malformed episode is reported with the reason. Episodes are
also logged to eval-protocol's local store; browse them with ``ep logs``.

Usage:
    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.frozen_lake.verify_rollout \\
        --model accounts/fireworks/models/qwen3-8b --tokenizer-model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from training.examples.rl.frozen_lake.train_frozen_lake import (
    FrozenLakeTask,
    evaluator_rows,
    make_frozen_lake_evaluator,
    make_rollout_fn_factory,
)
from training.recipes.async_rl_loop import RolloutSetup


async def verify(setup: RolloutSetup, task: FrozenLakeTask) -> list[str]:
    """Play one episode per seed row; return one summary line per episode."""
    evaluator = make_frozen_lake_evaluator(task)
    rollout_fn = make_rollout_fn_factory(evaluator)(setup)
    rows = evaluator_rows(evaluator)
    runs = await asyncio.gather(*(rollout_fn(row) for row in rows))
    lines = []
    for row, run in zip(rows, runs):
        if run is None:
            lines.append(f"{row['id']}: not trainable (reason logged above)")
            continue
        sample = run.segments[0]
        lines.append(
            f"{row['id']}: reward={sample.reward:.0f} tokens={len(sample.tokens)} "
            f"trained={sum(sample.loss_mask)} finish={sample.finish_reason}"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--base-url", default=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    )
    parser.add_argument("--num-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--observation-mode", choices=("text", "image"), default="text")
    parser.add_argument("--allow-plaintext-action-fallback", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    task = FrozenLakeTask(
        max_seeds=args.num_episodes,
        max_steps=args.max_steps,
        observation_mode=args.observation_mode,
        allow_plaintext_action_fallback=args.allow_plaintext_action_fallback,
        log_rollouts=True,
    )
    setup = RolloutSetup(
        sampler=None,  # the Eval Protocol rollout samples over HTTP
        tokenizer=None,
        tokenizer_id=args.tokenizer_model,
        sample_kwargs={
            "temperature": args.temperature,
            "max_tokens": args.max_completion_tokens,
        },
        inference_base_url=args.base_url,
        api_key=os.environ["FIREWORKS_API_KEY"],
        model=args.model,
        completions_per_prompt=1,
    )
    for line in asyncio.run(verify(setup, task)):
        print(line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
