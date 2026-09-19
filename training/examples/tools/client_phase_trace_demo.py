#!/usr/bin/env python3
"""Run one real serverless RL optimizer step and export its Perfetto trace.

This is an actual Fireworks training run, not a synthetic tracing workload. It
creates a serverless LoRA training session, publishes weights to a sampler,
rolls out Countdown completions, computes rewards and GRPO advantages, runs
forward/backward plus an optimizer step, and saves the final sampler checkpoint.

Usage:
    export FIREWORKS_API_KEY=fw_...
    export HF_TRUST_REMOTE_CODE=1
    python -m training.examples.tools.client_phase_trace_demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from training.examples.serverless_rl.countdown_rl import (
    Config,
    ServerlessCountdownRL,
)
from training.utils import configure_phase_tracing, flush_phase_trace, phase_span

SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "serverless_rl"
    / "data"
    / "countdown_train.jsonl"
)


def run_demo(
    trace_file: str | os.PathLike[str],
    *,
    run_dir: str | os.PathLike[str],
    base_model: str = Config.base_model,
    tokenizer_model: str = Config.tokenizer_model,
    renderer_name: str = Config.renderer_name,
    prompt_groups: int = 8,
    group_size: int = 4,
    max_sample_tokens: int = 256,
    max_seq_len: int = Config.max_seq_len,
    enable_otel: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run one bounded Countdown training step and return trace and metrics."""

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError("FIREWORKS_API_KEY is required for this real training demo")
    if prompt_groups < 1 or group_size < 2:
        raise ValueError("prompt_groups must be >= 1 and group_size must be >= 2")

    run_path = Path(run_dir).expanduser().resolve()
    trace_path = Path(trace_file).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    if enable_otel:
        os.environ["COOKBOOK_OTEL_ENABLED"] = "1"

    recorder = configure_phase_tracing(trace_path)
    if recorder is None:
        raise RuntimeError("client phase tracing did not initialize")

    records: list[dict[str, Any]] = []
    try:
        with phase_span(
            "serverless_countdown_training",
            category="train",
            attributes={
                "steps": 1,
                "prompt_groups": prompt_groups,
                "group_size": group_size,
            },
        ):
            runner = ServerlessCountdownRL(
                Config(
                    base_model=base_model,
                    tokenizer_model=tokenizer_model,
                    renderer_name=renderer_name,
                    dataset=str(SAMPLE_DATASET),
                    steps=1,
                    prompt_groups_per_step=prompt_groups,
                    group_size=group_size,
                    prompt_concurrency=min(prompt_groups, 8),
                    max_sample_tokens=max_sample_tokens,
                    max_seq_len=max_seq_len,
                    eval_prompt_groups=4,
                    eval_interval=0,
                    eval_at_start=False,
                    eval_at_end=False,
                    dcp_save_interval=0,
                    output_model_id="",
                    run_dir=str(run_path),
                    plot_reward_curve=False,
                )
            )
            records = runner.run()
    finally:
        written_path = flush_phase_trace()

    if written_path is None:
        raise RuntimeError("client phase trace was not written")
    if not records or not records[0]["train/trained"]:
        raise RuntimeError(
            "the sampled groups had no reward variance, so no optimizer step ran; "
            "rerun with larger --prompt-groups or --group-size"
        )
    return written_path, records[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="./client-phase-trace-demo")
    parser.add_argument(
        "--trace-file",
        default="./client-phase-trace-demo/client-phase-trace.json",
        help="Perfetto Chrome trace output path",
    )
    parser.add_argument("--base-model", default=Config.base_model)
    parser.add_argument("--tokenizer-model", default=Config.tokenizer_model)
    parser.add_argument("--renderer-name", default=Config.renderer_name)
    parser.add_argument("--prompt-groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-sample-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=Config.max_seq_len)
    parser.add_argument(
        "--otel",
        action="store_true",
        help="also use the process's configured OpenTelemetry tracer",
    )
    args = parser.parse_args(argv)

    path, metrics = run_demo(
        args.trace_file,
        run_dir=args.run_dir,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        prompt_groups=args.prompt_groups,
        group_size=args.group_size,
        max_sample_tokens=args.max_sample_tokens,
        max_seq_len=args.max_seq_len,
        enable_otel=args.otel,
    )
    print(f"Wrote {path}")
    print(
        "Completed real optimizer step: "
        f"loss={metrics['train/loss']} samples={metrics['rollout/raw_samples']}"
    )
    print("Open https://ui.perfetto.dev and select 'Open trace file'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
