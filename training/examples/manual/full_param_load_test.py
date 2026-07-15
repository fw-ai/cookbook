#!/usr/bin/env python3
# ruff: noqa: E402
"""Deterministic full-parameter trainer load test.

This is a manual service-mode RLOR driver for an already-created trainer job.
It sends a fixed ladder of 20 synthetic next-token datums, interleaves optimizer
steps at a fixed cadence, and loops until measured token throughput stabilizes.

Usage:
    export FIREWORKS_API_KEY=...
    python training/examples/manual/full_param_load_test.py \
        --job-id <rlor-trainer-job-id> \
        --base-model glm-5p2-fp8
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

import tinker
from fireworks.training.sdk import FiretitanServiceClient, TrainerJobManager

try:
    from fireworks.training.sdk.client import GradAccNormalization
except ImportError:

    class GradAccNormalization(str, enum.Enum):
        NUM_SEQUENCES = "num_sequences"
        NUM_LOSS_TOKENS = "num_loss_tokens"

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 3600
DEFAULT_ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0)
_FIREWORKS_API_EXTRA_HEADERS_ENV = "FIREWORKS_API_EXTRA_HEADERS"

DEFAULT_REQUEST_LENGTHS = (
    512,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    81920,
    98304,
    114688,
    131072,
    147456,
    163840,
    180224,
    196608,
    204785,
)


@dataclass(frozen=True)
class RequestSpec:
    index: int
    sequence_length: int
    target_tokens: int


@dataclass(frozen=True)
class CycleMetrics:
    cycle: int
    measured: bool
    requests: int
    optimizer_steps: int
    target_tokens: int
    elapsed_s: float
    fwd_bwd_s: float
    optim_step_s: float
    tokens_per_sec: float
    requests_per_sec: float


def _parse_positive_int_list(value: str, *, label: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be a comma-separated int list") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{label} entries must be positive")
    return parsed


def _parse_request_lengths(value: str) -> tuple[int, ...]:
    lengths = _parse_positive_int_list(value, label="request lengths")
    if len(lengths) != 20:
        raise argparse.ArgumentTypeError(f"expected exactly 20 request lengths, got {len(lengths)}")
    if any(length < 2 for length in lengths):
        raise argparse.ArgumentTypeError("request lengths must be at least 2 tokens")
    if sorted(lengths) != lengths:
        raise argparse.ArgumentTypeError("request lengths must be monotonically increasing")
    return tuple(lengths)


def _even_optimizer_schedule(num_requests: int, optimizer_steps: int) -> tuple[int, ...]:
    if optimizer_steps <= 0:
        raise ValueError("--optimizer-steps-per-cycle must be positive")
    if optimizer_steps > num_requests:
        raise ValueError("--optimizer-steps-per-cycle cannot exceed request count")

    points: list[int] = []
    for i in range(1, optimizer_steps + 1):
        point = round(i * num_requests / optimizer_steps)
        point = max(1, min(num_requests, point))
        if points and point <= points[-1]:
            point = points[-1] + 1
        points.append(min(point, num_requests))
    points[-1] = num_requests
    return tuple(points)


def _resolve_optimizer_schedule(
    *,
    optimizer_step_after: str | None,
    optimizer_steps_per_cycle: int,
    num_requests: int,
) -> tuple[int, ...]:
    if optimizer_step_after:
        points = _parse_positive_int_list(
            optimizer_step_after,
            label="optimizer step positions",
        )
        if sorted(points) != points:
            raise ValueError("--optimizer-step-after must be monotonically increasing")
        if len(set(points)) != len(points):
            raise ValueError("--optimizer-step-after must not contain duplicates")
        if points[-1] != num_requests:
            raise ValueError(
                f"--optimizer-step-after must end with {num_requests} so gradients are flushed"
            )
        if any(point > num_requests for point in points):
            raise ValueError("--optimizer-step-after cannot exceed request count")
        return tuple(points)
    return _even_optimizer_schedule(num_requests, optimizer_steps_per_cycle)


def _relative_range(values: list[float]) -> float:
    mean = sum(values) / len(values)
    if mean <= 0:
        return float("inf")
    return (max(values) - min(values)) / mean


def _summarize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    keys = ("loss:sum", "loss:mean", "response_tokens", "num_loss_tokens", "tokens")
    return {key: metrics[key] for key in keys if key in metrics}


def _read_api_extra_headers_env() -> dict[str, str] | None:
    raw = os.environ.get(_FIREWORKS_API_EXTRA_HEADERS_ENV, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Invalid %s (not valid JSON); ignoring: %s",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
            exc,
        )
        return None
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        logger.warning(
            "%s must be a JSON object of string->string; ignoring",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
        )
        return None
    return parsed or None


def _normalize_grad_accumulation_normalization(
    value: str | None,
) -> GradAccNormalization | None:
    if value is None:
        return None
    return GradAccNormalization(value)


def _deterministic_token(
    position: int,
    *,
    request_index: int,
    min_token_id: int,
    vocab_modulo: int,
) -> int:
    # Fixed LCG-style pattern. This avoids Python's randomized hash seed and
    # keeps every run byte-for-byte deterministic without depending on a tokenizer.
    return min_token_id + ((request_index * 1_000_003 + position * 97 + position // 17) % vocab_modulo)


def _build_datum(
    spec: RequestSpec,
    *,
    min_token_id: int,
    vocab_modulo: int,
) -> tinker.Datum:
    tokens = [
        _deterministic_token(
            pos,
            request_index=spec.index,
            min_token_id=min_token_id,
            vocab_modulo=vocab_modulo,
        )
        for pos in range(spec.sequence_length)
    ]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens[1:],
                dtype="int64",
                shape=[spec.target_tokens],
            )
        },
    )


def _maybe_log_nvidia_smi(skip: bool) -> None:
    if skip:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        logger.warning("nvidia-smi not found locally; run it on any GPU launch host before starting GPU work.")
        return
    try:
        completed = subprocess.run(
            [nvidia_smi],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort preflight logging
        logger.warning("nvidia-smi check failed: %s", exc)
        return
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        logger.warning("nvidia-smi exited with %d: %s", completed.returncode, output)
        return
    logger.info("nvidia-smi preflight output:\n%s", output)


def _run_cycle(
    *,
    cycle: int,
    measured: bool,
    policy: Any,
    datums: list[tinker.Datum],
    specs: list[RequestSpec],
    optimizer_after: set[int],
    adam_params: tinker.AdamParams,
    grad_accumulation_normalization: str | None,
    step_timeout_s: int,
) -> CycleMetrics:
    start = time.perf_counter()
    fwd_bwd_s = 0.0
    optim_step_s = 0.0
    optim_steps = 0
    target_tokens = sum(spec.target_tokens for spec in specs)

    for one_based_idx, (spec, datum) in enumerate(zip(specs, datums, strict=True), start=1):
        request_start = time.perf_counter()
        result = policy.forward_backward([datum], "cross_entropy").result(timeout=step_timeout_s)
        request_elapsed = time.perf_counter() - request_start
        fwd_bwd_s += request_elapsed
        logger.info(
            "cycle=%d phase=%s request=%02d sequence_length=%d target_tokens=%d fwd_bwd_s=%.3f metrics=%s",
            cycle,
            "measure" if measured else "warmup",
            one_based_idx,
            spec.sequence_length,
            spec.target_tokens,
            request_elapsed,
            _summarize_metrics(getattr(result, "metrics", None)),
        )

        if one_based_idx in optimizer_after:
            optim_start = time.perf_counter()
            kwargs: dict[str, Any] = {}
            normalized_grad_acc = _normalize_grad_accumulation_normalization(
                grad_accumulation_normalization
            )
            if normalized_grad_acc is not None:
                kwargs["grad_accumulation_normalization"] = normalized_grad_acc
            optim_result = policy.optim_step(adam_params, **kwargs).result(timeout=step_timeout_s)
            optim_elapsed = time.perf_counter() - optim_start
            optim_step_s += optim_elapsed
            optim_steps += 1
            logger.info(
                "cycle=%d phase=%s request=%02d optimizer_step=%d optim_step_s=%.3f metrics=%s",
                cycle,
                "measure" if measured else "warmup",
                one_based_idx,
                optim_steps,
                optim_elapsed,
                _summarize_metrics(getattr(optim_result, "metrics", None)),
            )

    elapsed = time.perf_counter() - start
    return CycleMetrics(
        cycle=cycle,
        measured=measured,
        requests=len(specs),
        optimizer_steps=optim_steps,
        target_tokens=target_tokens,
        elapsed_s=elapsed,
        fwd_bwd_s=fwd_bwd_s,
        optim_step_s=optim_step_s,
        tokens_per_sec=target_tokens / elapsed,
        requests_per_sec=len(specs) / elapsed,
    )


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive a running full-param service-mode RLOR trainer with deterministic synthetic load."
    )
    parser.add_argument("--job-id", required=True, help="Existing RLOR trainer job ID")
    parser.add_argument("--base-model", default="glm-5p2-fp8", help="Base model passed to create_training_client")
    parser.add_argument("--lora-rank", type=int, default=0, help="Use 0 for full-parameter training")
    parser.add_argument("--base-url", default=None, help="Fireworks API base URL")
    parser.add_argument("--connect-timeout-s", type=int, default=1800)
    parser.add_argument("--step-timeout-s", type=int, default=7200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--request-lengths",
        type=_parse_request_lengths,
        default=DEFAULT_REQUEST_LENGTHS,
        help="Comma-separated 20 sequence lengths. Default spans 512..204785.",
    )
    parser.add_argument("--min-token-id", type=int, default=100)
    parser.add_argument(
        "--vocab-modulo",
        type=int,
        default=32000,
        help="Synthetic token IDs are in [min_token_id, min_token_id + vocab_modulo).",
    )
    parser.add_argument("--optimizer-steps-per-cycle", type=int, default=8)
    parser.add_argument(
        "--optimizer-step-after",
        default=None,
        help="Optional comma-separated 1-based request positions for optimizer steps. Must end at 20.",
    )
    parser.add_argument(
        "--grad-accumulation-normalization",
        default=None,
        choices=("num_sequences", "num_loss_tokens"),
        help="Optional server-side normalization passed to optim_step.",
    )
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--min-measured-cycles", type=int, default=3)
    parser.add_argument("--max-measured-cycles", type=int, default=12)
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument(
        "--stable-rel-range",
        type=float,
        default=0.05,
        help="Stability threshold: (max TPS - min TPS) / mean TPS over the stable window.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path for final JSON summary")
    parser.add_argument(
        "--skip-nvidia-smi-check",
        action="store_true",
        help="Skip the local nvidia-smi preflight log.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.lora_rank != 0:
        logger.warning("--lora-rank=%d was provided; this is no longer a full-param load test.", args.lora_rank)
    if args.vocab_modulo <= 0:
        parser.error("--vocab-modulo must be positive")
    if args.min_token_id < 0:
        parser.error("--min-token-id must be non-negative")
    if args.warmup_cycles < 0:
        parser.error("--warmup-cycles must be non-negative")
    if args.min_measured_cycles <= 0:
        parser.error("--min-measured-cycles must be positive")
    if args.max_measured_cycles < args.min_measured_cycles:
        parser.error("--max-measured-cycles must be >= --min-measured-cycles")
    if args.stable_window <= 0:
        parser.error("--stable-window must be positive")
    if args.stable_window > args.max_measured_cycles:
        parser.error("--stable-window must be <= --max-measured-cycles")
    if args.stable_rel_range <= 0:
        parser.error("--stable-rel-range must be positive")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _maybe_log_nvidia_smi(args.skip_nvidia_smi_check)

    request_lengths = tuple(args.request_lengths)
    specs = [
        RequestSpec(index=i, sequence_length=length, target_tokens=length - 1)
        for i, length in enumerate(request_lengths)
    ]
    optimizer_schedule = _resolve_optimizer_schedule(
        optimizer_step_after=args.optimizer_step_after,
        optimizer_steps_per_cycle=args.optimizer_steps_per_cycle,
        num_requests=len(specs),
    )
    logger.info("request_lengths=%s", ",".join(str(x) for x in request_lengths))
    logger.info("optimizer_step_after=%s", ",".join(str(x) for x in optimizer_schedule))

    logger.info("building deterministic datums")
    datums = [
        _build_datum(spec, min_token_id=args.min_token_id, vocab_modulo=args.vocab_modulo)
        for spec in specs
    ]
    total_target_tokens = sum(spec.target_tokens for spec in specs)
    logger.info(
        "built %d datums; target_tokens_per_cycle=%d",
        len(datums),
        total_target_tokens,
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = args.base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = _read_api_extra_headers_env()

    service: FiretitanServiceClient | None = None
    policy: Any | None = None
    cycles: list[CycleMetrics] = []
    measured_tps: list[float] = []
    stable = False
    final_tps: float | None = None

    try:
        mgr = TrainerJobManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )
        logger.info("reconnecting to trainer job %s", args.job_id)
        endpoint = mgr.reconnect_and_wait(args.job_id, timeout_s=args.connect_timeout_s)
        service = FiretitanServiceClient(
            base_url=endpoint.base_url,
            api_key=api_key,
            additional_headers=additional_headers,
        )
        policy = service.create_training_client(args.base_model, lora_rank=args.lora_rank)
        adam_params = tinker.AdamParams(learning_rate=args.learning_rate, **DEFAULT_ADAM)

        for cycle in range(1, args.warmup_cycles + args.max_measured_cycles + 1):
            measured = cycle > args.warmup_cycles
            metrics = _run_cycle(
                cycle=cycle,
                measured=measured,
                policy=policy,
                datums=datums,
                specs=specs,
                optimizer_after=set(optimizer_schedule),
                adam_params=adam_params,
                grad_accumulation_normalization=args.grad_accumulation_normalization,
                step_timeout_s=args.step_timeout_s or DEFAULT_TIMEOUT_S,
            )
            cycles.append(metrics)
            logger.info(
                "cycle=%d phase=%s elapsed_s=%.3f fwd_bwd_s=%.3f optim_step_s=%.3f tokens_per_sec=%.3f requests_per_sec=%.4f",
                metrics.cycle,
                "measure" if metrics.measured else "warmup",
                metrics.elapsed_s,
                metrics.fwd_bwd_s,
                metrics.optim_step_s,
                metrics.tokens_per_sec,
                metrics.requests_per_sec,
            )

            if measured:
                measured_tps.append(metrics.tokens_per_sec)
                if (
                    len(measured_tps) >= args.min_measured_cycles
                    and len(measured_tps) >= args.stable_window
                ):
                    window = measured_tps[-args.stable_window :]
                    rel_range = _relative_range(window)
                    logger.info(
                        "stability window=%s rel_range=%.4f threshold=%.4f",
                        [round(v, 3) for v in window],
                        rel_range,
                        args.stable_rel_range,
                    )
                    if rel_range <= args.stable_rel_range:
                        stable = True
                        final_tps = sum(window) / len(window)
                        break

        if final_tps is None and measured_tps:
            final_tps = sum(measured_tps[-args.stable_window :]) / min(args.stable_window, len(measured_tps))

    finally:
        if service is not None:
            service.close()

    summary = {
        "job_id": args.job_id,
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "request_lengths": list(request_lengths),
        "optimizer_step_after": list(optimizer_schedule),
        "target_tokens_per_cycle": total_target_tokens,
        "warmup_cycles": args.warmup_cycles,
        "stable": stable,
        "stable_window": args.stable_window,
        "stable_rel_range": args.stable_rel_range,
        "final_tokens_per_sec": final_tps,
        "cycles": [asdict(cycle) for cycle in cycles],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    _write_json(args.output_json, summary)

    if not stable:
        logger.error("throughput did not stabilize within %d measured cycles", args.max_measured_cycles)
        return 2
    logger.info("final stable tokens_per_sec=%.3f", final_tps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
