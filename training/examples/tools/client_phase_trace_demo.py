#!/usr/bin/env python3
"""Run one real serverless RL optimizer step and export its Perfetto trace.

This is an actual Fireworks training run, not a synthetic tracing workload. It
runs the serverless DeepMath example (``examples/rl/deepmath/train_deepmath_serverless.py``)
for one optimizer batch: the recipe creates a serverless LoRA training session,
publishes weights to a sampler, rolls out DeepMath completions, runs
forward/backward plus an optimizer step, publishes the next sampler snapshot,
and saves the final training state.

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

from training.examples.rl.deepmath.train_deepmath import DEFAULT_DATASET
from training.examples.rl.deepmath.train_deepmath_serverless import train as train_deepmath
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.utils import configure_phase_tracing, flush_phase_trace, phase_span

SAMPLE_DATASET = DEFAULT_DATASET


def run_demo(
    trace_file: str | os.PathLike[str],
    *,
    run_dir: str | os.PathLike[str],
    base_model: str = Config.base_model,
    tokenizer_model: str = Config.tokenizer_model,
    renderer_name: str = "",
    prompt_groups: int = 8,
    group_size: int = 4,
    max_sample_tokens: int = 2048,
    max_seq_len: int = 32768,
    enable_otel: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run one bounded DeepMath training step and return trace and run result."""

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

    result: dict[str, Any] = {}
    try:
        with phase_span(
            "serverless_deepmath_training",
            category="train",
            attributes={
                "steps": 1,
                "prompt_groups": prompt_groups,
                "group_size": group_size,
            },
        ):
            result = train_deepmath(
                Config(
                    base_model=base_model,
                    tokenizer_model=tokenizer_model,
                    dataset=str(SAMPLE_DATASET),
                    max_rows=prompt_groups,
                    prompt_groups_per_step=prompt_groups,
                    completions_per_prompt=group_size,
                    min_group_size=2,
                    max_concurrency_rollout_sample=prompt_groups * group_size,
                    max_completion_tokens=max_sample_tokens,
                    max_seq_len=max_seq_len,
                    lora_rank=32,
                    snapshot_prefix="phase-trace-demo",
                    metrics_file=str(run_path / "metrics.jsonl"),
                ),
                renderer_name=renderer_name,
                eval_rows=0,
            )
    finally:
        written_path = flush_phase_trace()

    if written_path is None:
        raise RuntimeError("client phase trace was not written")
    if not result.get("steps"):
        raise RuntimeError(
            "the sampled groups had no reward variance, so no optimizer step ran; "
            "rerun with larger --prompt-groups, --group-size or --max-sample-tokens"
        )
    return written_path, result


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
    parser.add_argument("--renderer-name", default="")
    parser.add_argument("--prompt-groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-sample-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument(
        "--otel",
        action="store_true",
        help="also use the process's configured OpenTelemetry tracer",
    )
    args = parser.parse_args(argv)

    path, result = run_demo(
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
        f"steps={result['steps']} accepted_groups={result['accepted_groups']}"
    )
    print("Open https://ui.perfetto.dev and select 'Open trace file'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
