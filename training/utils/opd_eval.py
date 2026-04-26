"""Evaluation and validation helpers for privileged-context OPD."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.data import prepare_sampling_messages
from training.utils.opd_sampling import (
    _score_with_teacher,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)


FINAL_ANSWER_LINE_RE = re.compile(r"^\s*Final:\s*([^\n\r]+?)\s*$", re.IGNORECASE)
PROMPT_FINAL_ANSWER_RE = re.compile(r"Final:\s*([^\n\r]+)", re.IGNORECASE)
DEFAULT_PRIVILEGED_CONTEXT_MARKERS = (
    "privileged",
    "private",
    "worked solution",
    "gold solution",
    "golden solution",
    "teacher-only",
    "hidden",
)


def normalize_final_answer(answer: Any) -> str:
    """Normalize a final answer for lightweight exact-match evaluation."""
    normalized = re.sub(r"\s+", " ", str(answer).strip())
    return normalized.rstrip(".")


def expected_final_answer(row: dict[str, Any]) -> str | None:
    """Read the expected answer from common JSONL row fields."""
    for key in ("expected_answer", "expected", "answer"):
        if row.get(key) is not None:
            normalized = normalize_final_answer(row[key])
            return normalized or None
    return None


def extract_final_answer(text: str) -> str | None:
    """Extract ``Final: ...`` only when it is the completion's last non-empty line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = FINAL_ANSWER_LINE_RE.match(lines[-1])
    if not match:
        return None
    return normalize_final_answer(match.group(1))


def _extract_prompt_final_answer(text: str) -> str | None:
    """Extract a pinned ``Final: ...`` answer from prompt text."""
    matches = list(PROMPT_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return normalize_final_answer(matches[-1].group(1))


def _load_opd_rows(dataset: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(dataset, list):
        return list(dataset)
    path = Path(dataset)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_privileged_opd_dataset(
    dataset: str | Path | list[dict[str, Any]],
    *,
    min_rows: int = 1,
    privileged_markers: tuple[str, ...] = DEFAULT_PRIVILEGED_CONTEXT_MARKERS,
    require_teacher_final_answer: bool = True,
    require_expected_answer: bool = True,
) -> None:
    """Validate JSONL rows for privileged-context OPD before launching a job."""
    errors: list[str] = []
    try:
        rows = _load_opd_rows(dataset)
    except Exception as exc:
        raise ValueError(f"Failed to load OPD dataset {dataset!r}: {exc}") from exc

    if len(rows) < min_rows:
        errors.append(f"expected at least {min_rows} rows, got {len(rows)}")
    lower_markers = tuple(marker.lower() for marker in privileged_markers)
    for idx, row in enumerate(rows, start=1):
        messages = row.get("messages")
        teacher_messages = (
            row.get("teacher_messages")
            or row.get("privileged_messages")
            or row.get("teacher_prompt_messages")
        )
        if not messages:
            errors.append(f"row {idx} must include student `messages`")
            continue
        if not teacher_messages:
            errors.append(f"row {idx} must include privileged `teacher_messages`")
            continue
        if messages == teacher_messages:
            errors.append(f"row {idx} teacher_messages must differ from student messages")

        student_text = "\n".join(str(message.get("content", "")) for message in messages)
        teacher_text = "\n".join(str(message.get("content", "")) for message in teacher_messages)
        lower_student_text = student_text.lower()
        lower_teacher_text = teacher_text.lower()
        if any(marker in lower_student_text for marker in lower_markers):
            errors.append(f"row {idx} leaks privileged context into student messages")
        if not any(marker in lower_teacher_text for marker in lower_markers):
            errors.append(f"row {idx} teacher_messages missing privileged context")
        if "Final:" not in student_text or "Final:" not in teacher_text:
            errors.append(f"row {idx} prompts must ask for a `Final:` answer")
        if "Final: <" in teacher_text:
            errors.append(f"row {idx} teacher prompt must not use a placeholder final-answer format")

        expected = expected_final_answer(row)
        if require_expected_answer and expected is None:
            errors.append(f"row {idx} is missing expected_answer")
        if require_teacher_final_answer and expected is not None:
            teacher_final = _extract_prompt_final_answer(teacher_text)
            if teacher_final != expected:
                errors.append(
                    f"row {idx} teacher prompt must pin exact final answer "
                    f"`Final: {expected}`; parsed {teacher_final!r}"
                )

    if errors:
        raise ValueError("Invalid privileged OPD dataset:\n" + "\n".join(f"- {error}" for error in errors))


def _find_last_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def _final_answer_span(
    tokenizer: Any,
    text: str,
    completion_tokens: list[int],
) -> tuple[int, int, str] | None:
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return None
    final_line = non_empty_lines[-1]
    line_match = FINAL_ANSWER_LINE_RE.match(final_line)
    if not line_match:
        return None
    line_start = text.rfind(final_line)
    answer_start = line_start + line_match.start(1)
    answer_text = line_match.group(1)
    answer = normalize_final_answer(answer_text)
    prefix_tokens = tokenizer.encode(text[:answer_start], add_special_tokens=False)
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
    start = len(prefix_tokens)
    end = start + len(answer_tokens)
    if not answer_tokens:
        return None
    if end > len(completion_tokens) or completion_tokens[start:end] != list(answer_tokens):
        found_start = _find_last_subsequence(completion_tokens, list(answer_tokens))
        if found_start is None:
            return None
        start = found_start
        end = start + len(answer_tokens)
    return start, end, answer


def _span_nll(logprobs: list[float], start: int = 0, end: int | None = None) -> float:
    span = logprobs[start:end]
    if not span:
        return 0.0
    return -sum(float(lp) for lp in span) / len(span)


async def _sample_eval_completion(
    sampler: DeploymentSampler,
    messages: list[dict[str, Any]],
    config: Any,
    *,
    max_seq_len: int | None,
    temperature: float,
) -> Any:
    samples = await sampler.sample_with_tokens(
        messages=prepare_sampling_messages(messages),
        n=1,
        max_tokens=config.max_completion_tokens,
        temperature=temperature,
        max_seq_len=max_seq_len,
        http_timeout=config.deployment.sample_timeout,
        logprobs=True,
    )
    if not samples:
        raise RuntimeError("deployment returned no samples during OPD trace eval")
    sample = samples[0]
    if len(sample.full_tokens) <= sample.prompt_len:
        raise RuntimeError("deployment returned an empty completion during OPD trace eval")
    return sample


async def _score_eval_response_tokens(
    sampler: DeploymentSampler,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    completion_tokens: list[int],
    http_timeout: int,
) -> list[float]:
    prompt_tokens = _tokenize_teacher_prompt(tokenizer, prepare_sampling_messages(messages))
    response_logprobs = await _score_with_teacher(
        sampler,
        list(prompt_tokens) + list(completion_tokens),
        prompt_len=len(prompt_tokens),
        response_len=len(completion_tokens),
        top_logprobs=0,
        http_timeout=http_timeout,
    )
    if response_logprobs is None:
        raise RuntimeError("failed to score OPD trace response tokens")
    if len(response_logprobs) < len(completion_tokens):
        raise RuntimeError(
            "trace scoring returned too few response logprobs: "
            f"{len(response_logprobs)} < {len(completion_tokens)}"
        )
    return [float(lp) for lp in response_logprobs]


async def _evaluate_teacher_trace_logprob_gap_async(
    context: dict[str, Any],
    *,
    trace_log_path: str | Path | None = None,
    min_pre_final_tokens: int = 0,
) -> dict[str, Any]:
    """Score teacher reasoning traces under teacher and student prompts.

    This is meant for privileged-context OPD eval: the frozen teacher generates
    a trace from ``teacher_messages``; the same trace tokens are then scored
    under both the privileged teacher prompt and the ordinary student prompt.
    """
    config: Any = context["config"]
    student_sampler = context["student_sampler"]
    teacher_sampler = context["teacher_sampler"]
    tokenizer = context["tokenizer"]
    rows = list(context["dataset"])
    step = int(context.get("global_step", 0))
    max_seq_len = context.get("max_seq_len")

    records: list[dict[str, Any]] = []
    for row in rows:
        expected = expected_final_answer(row)
        if expected is None:
            raise RuntimeError(f"row is missing an expected answer: {row!r}")

        student_messages = prepare_sampling_messages(row.get("messages", []))
        if not student_messages:
            raise RuntimeError(f"row is missing messages: {row!r}")
        teacher_messages = _teacher_messages_for_row(
            row,
            student_messages,
        )

        teacher_sample = await _sample_eval_completion(
            teacher_sampler,
            teacher_messages,
            config,
            max_seq_len=max_seq_len,
            temperature=0.0,
        )
        completion_tokens = list(teacher_sample.full_tokens[teacher_sample.prompt_len :])
        trace_text = str(getattr(teacher_sample, "text", ""))
        span = _final_answer_span(tokenizer, trace_text, completion_tokens)
        if span is None:
            raise RuntimeError(f"teacher trace did not contain a parseable final answer: {trace_text!r}")
        final_start, final_end, teacher_generated_final = span
        if final_start < min_pre_final_tokens:
            raise RuntimeError(
                "teacher trace did not include enough pre-final tokens to validate "
                f"thinking-token distillation ({final_start} < {min_pre_final_tokens})"
            )

        teacher_logprobs = await _score_eval_response_tokens(
            teacher_sampler,
            tokenizer,
            teacher_messages,
            completion_tokens,
            config.deployment.sample_timeout,
        )
        student_logprobs = await _score_eval_response_tokens(
            student_sampler,
            tokenizer,
            student_messages,
            completion_tokens,
            config.deployment.sample_timeout,
        )
        student_sample = await _sample_eval_completion(
            student_sampler,
            student_messages,
            config,
            max_seq_len=max_seq_len,
            temperature=0.0,
        )
        student_text = str(getattr(student_sample, "text", ""))
        student_generated_final = extract_final_answer(student_text)

        teacher_nll = _span_nll(teacher_logprobs)
        student_nll = _span_nll(student_logprobs)
        pre_final_teacher_nll = _span_nll(teacher_logprobs, 0, final_start)
        pre_final_student_nll = _span_nll(student_logprobs, 0, final_start)
        final_teacher_nll = _span_nll(teacher_logprobs, final_start, final_end)
        final_student_nll = _span_nll(student_logprobs, final_start, final_end)
        records.append(
            {
                "step": step,
                "expected_answer": expected,
                "teacher_generated_final": teacher_generated_final,
                "student_generated_final": student_generated_final,
                "tokens": len(completion_tokens),
                "pre_final_tokens": final_start,
                "final_tokens": final_end - final_start,
                "teacher_trace": trace_text,
                "student_generation": student_text,
                "teacher_nll": teacher_nll,
                "student_nll": student_nll,
                "student_minus_teacher_nll": student_nll - teacher_nll,
                "pre_final_teacher_nll": pre_final_teacher_nll,
                "pre_final_student_nll": pre_final_student_nll,
                "pre_final_student_minus_teacher_nll": pre_final_student_nll - pre_final_teacher_nll,
                "final_teacher_nll": final_teacher_nll,
                "final_student_nll": final_student_nll,
                "final_student_minus_teacher_nll": final_student_nll - final_teacher_nll,
                "teacher_final_correct": normalize_final_answer(teacher_generated_final) == expected,
                "student_generation_correct": (
                    student_generated_final is not None
                    and normalize_final_answer(student_generated_final) == expected
                ),
            }
        )

    total = len(records)
    total_tokens = sum(int(record["tokens"]) for record in records)
    pre_final_tokens = sum(int(record["pre_final_tokens"]) for record in records)
    final_tokens = sum(int(record["final_tokens"]) for record in records)
    denom = max(total_tokens, 1)
    pre_final_denom = max(pre_final_tokens, 1)
    final_denom = max(final_tokens, 1)
    teacher_nll = sum(float(record["teacher_nll"]) * int(record["tokens"]) for record in records) / denom
    student_nll = sum(float(record["student_nll"]) * int(record["tokens"]) for record in records) / denom
    pre_final_teacher_nll = sum(
        float(record["pre_final_teacher_nll"]) * int(record["pre_final_tokens"]) for record in records
    ) / pre_final_denom
    pre_final_student_nll = sum(
        float(record["pre_final_student_nll"]) * int(record["pre_final_tokens"]) for record in records
    ) / pre_final_denom
    final_teacher_nll = sum(
        float(record["final_teacher_nll"]) * int(record["final_tokens"]) for record in records
    ) / final_denom
    final_student_nll = sum(
        float(record["final_student_nll"]) * int(record["final_tokens"]) for record in records
    ) / final_denom
    teacher_final_accuracy = sum(
        1.0 for record in records if record["teacher_final_correct"]
    ) / max(total, 1)
    student_generation_accuracy = sum(
        1.0 for record in records if record["student_generation_correct"]
    ) / max(total, 1)

    if trace_log_path is not None:
        path = Path(trace_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n"
            )

    return {
        "eval/opd_trace_teacher_nll": teacher_nll,
        "eval/opd_trace_student_nll": student_nll,
        "eval/opd_trace_student_minus_teacher_nll": student_nll - teacher_nll,
        "eval/opd_trace_pre_final_teacher_nll": pre_final_teacher_nll,
        "eval/opd_trace_pre_final_student_nll": pre_final_student_nll,
        "eval/opd_trace_pre_final_student_minus_teacher_nll": (
            pre_final_student_nll - pre_final_teacher_nll
        ),
        "eval/opd_trace_final_teacher_nll": final_teacher_nll,
        "eval/opd_trace_final_student_nll": final_student_nll,
        "eval/opd_trace_final_student_minus_teacher_nll": final_student_nll - final_teacher_nll,
        "eval/opd_trace_teacher_final_accuracy": teacher_final_accuracy,
        "eval/opd_trace_student_generation_accuracy": student_generation_accuracy,
        "eval/opd_trace_examples": float(total),
        "eval/opd_trace_tokens": float(total_tokens),
        "eval/opd_trace_pre_final_tokens": float(pre_final_tokens),
        "eval/opd_trace_final_tokens": float(final_tokens),
    }


def evaluate_teacher_trace_logprob_gap(
    context: dict[str, Any],
    *,
    trace_log_path: str | Path | None = None,
    min_pre_final_tokens: int = 0,
) -> dict[str, Any]:
    """Synchronous callback wrapper for ``Config.step_eval`` / ``post_training_eval``."""
    return asyncio.run(
        _evaluate_teacher_trace_logprob_gap_async(
            context,
            trace_log_path=trace_log_path,
            min_pre_final_tokens=min_pre_final_tokens,
        )
    )


def make_teacher_trace_logprob_gap_eval(
    *,
    trace_log_path: str | Path | None = None,
    min_pre_final_tokens: int = 0,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a no-argument eval callback suitable for ``Config.step_eval``."""

    def _callback(context: dict[str, Any]) -> dict[str, Any]:
        return evaluate_teacher_trace_logprob_gap(
            context,
            trace_log_path=trace_log_path,
            min_pre_final_tokens=min_pre_final_tokens,
        )

    return _callback


def validate_opd_trace_result(
    config: Any,
    result: dict[str, Any],
    *,
    expected_steps: int | None = None,
    min_abs_advantage: float = 1e-4,
    min_train_gap_improvement: float | None = None,
    min_trace_gap_improvement: float | None = None,
    min_teacher_final_accuracy: float | None = None,
    min_student_generation_accuracy: float | None = None,
    min_max_seq_len: int | None = None,
) -> None:
    """Validate that an OPD run produced train and teacher-trace eval signal."""
    errors: list[str] = []
    if expected_steps is None:
        expected_steps = (
            config.max_rows * config.epochs + config.prompt_groups_per_step - 1
        ) // config.prompt_groups_per_step

    steps = int(result.get("steps", 0))
    if steps < expected_steps:
        errors.append(f"expected {expected_steps} optimizer steps, got {steps}")
    resolved_max_seq_len = result.get("max_seq_len")
    if min_max_seq_len is not None and resolved_max_seq_len is not None:
        if int(resolved_max_seq_len) < min_max_seq_len:
            errors.append(
                f"expected resolved max_seq_len >= {min_max_seq_len}, got {resolved_max_seq_len}"
            )

    metrics_path = Path(config.runner.metrics_file or "")
    records: list[dict[str, Any]] = []
    if not metrics_path.exists():
        errors.append(f"metrics file was not written: {metrics_path}")
    else:
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_records = [record for record in records if "train/opd_active_tokens" in record]
        if len(train_records) < expected_steps:
            errors.append(
                f"expected at least {expected_steps} train metric records, got {len(train_records)}"
            )
        active_tokens = sum(float(record.get("train/opd_active_tokens", 0.0)) for record in train_records)
        max_abs_advantage = max(
            (abs(float(record.get("train/opd_abs_advantage", 0.0))) for record in train_records),
            default=0.0,
        )
        if active_tokens <= 0:
            errors.append("no active OPD tokens were trained")
        if max_abs_advantage < min_abs_advantage:
            errors.append(
                "teacher/student logprob delta was effectively zero "
                f"(max train/opd_abs_advantage={max_abs_advantage:.6g})"
            )

        sampled_kl = [
            float(record.get("train/opd_sampled_reverse_kl", 0.0))
            for record in train_records
            if "train/opd_sampled_reverse_kl" in record
        ]
        if min_train_gap_improvement is not None and len(sampled_kl) >= 4:
            head = sampled_kl[1 : max(2, len(sampled_kl) // 2)]
            tail = sampled_kl[-min(5, len(sampled_kl)) :]
            head_mean = sum(head) / max(len(head), 1)
            tail_mean = sum(tail) / max(len(tail), 1)
            improvement = (head_mean - tail_mean) / max(abs(head_mean), 1e-9)
            if improvement < min_train_gap_improvement:
                errors.append(
                    "sampled teacher/student logprob gap did not improve enough "
                    f"(head_mean={head_mean:.4g}, tail_mean={tail_mean:.4g}, "
                    f"improvement={improvement:.1%})"
                )

        eval_records = [
            record for record in records if "eval/opd_trace_teacher_nll" in record
        ]
        expected_eval_records = expected_steps + (1 if config.eval_before_training else 0)
        if len(eval_records) < expected_eval_records:
            errors.append(
                f"expected {expected_eval_records} teacher-trace eval records, got {len(eval_records)}"
            )
        if config.eval_before_training and not any(record.get("step") == 0 for record in eval_records):
            errors.append("missing teacher-trace pre-training eval at step 0")
        trace_gaps = [
            float(record["eval/opd_trace_student_minus_teacher_nll"])
            for record in eval_records
            if "eval/opd_trace_student_minus_teacher_nll" in record
        ]
        if eval_records and len(trace_gaps) < len(eval_records):
            errors.append("teacher-trace eval records are missing student/teacher logprob gap")
        if min_trace_gap_improvement is not None and len(trace_gaps) >= 2 and trace_gaps[0] > 0:
            improvement = (trace_gaps[0] - trace_gaps[-1]) / max(abs(trace_gaps[0]), 1e-9)
            if improvement < min_trace_gap_improvement:
                errors.append(
                    "teacher-trace student/teacher logprob gap did not improve enough "
                    f"(initial={trace_gaps[0]:.4g}, final={trace_gaps[-1]:.4g}, "
                    f"improvement={improvement:.1%})"
                )

    eval_metrics = result.get("eval") or {}
    required_eval_metrics = [
        "eval/opd_trace_teacher_nll",
        "eval/opd_trace_student_nll",
        "eval/opd_trace_student_minus_teacher_nll",
        "eval/opd_trace_pre_final_teacher_nll",
        "eval/opd_trace_pre_final_student_nll",
        "eval/opd_trace_pre_final_student_minus_teacher_nll",
        "eval/opd_trace_final_teacher_nll",
        "eval/opd_trace_final_student_nll",
        "eval/opd_trace_final_student_minus_teacher_nll",
        "eval/opd_trace_teacher_final_accuracy",
        "eval/opd_trace_student_generation_accuracy",
        "eval/opd_trace_examples",
        "eval/opd_trace_tokens",
        "eval/opd_trace_pre_final_tokens",
        "eval/opd_trace_final_tokens",
    ]
    for key in required_eval_metrics:
        value = eval_metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            errors.append(f"missing or invalid teacher-trace eval metric: {key}")

    teacher_accuracy = float(eval_metrics.get("eval/opd_trace_teacher_final_accuracy", 0.0))
    student_accuracy = float(eval_metrics.get("eval/opd_trace_student_generation_accuracy", 0.0))
    if min_teacher_final_accuracy is not None and teacher_accuracy < min_teacher_final_accuracy:
        errors.append(
            "frozen teacher does not reliably generate the privileged final answer "
            f"(accuracy={teacher_accuracy:.3f}, required={min_teacher_final_accuracy:.3f})"
        )
    if (
        min_student_generation_accuracy is not None
        and student_accuracy < min_student_generation_accuracy
    ):
        errors.append(
            "student did not reach the privileged-answer exact-match target by final eval "
            f"(accuracy={student_accuracy:.3f}, required={min_student_generation_accuracy:.3f})"
        )
    if float(eval_metrics.get("eval/opd_trace_examples", 0.0)) <= 0:
        errors.append("teacher-trace eval scored no examples")
    if float(eval_metrics.get("eval/opd_trace_tokens", 0.0)) <= 0:
        errors.append("teacher-trace eval scored no response tokens")
    if float(eval_metrics.get("eval/opd_trace_pre_final_tokens", 0.0)) <= 0:
        errors.append("teacher-trace eval scored no pre-final thinking/reasoning tokens")
    if float(eval_metrics.get("eval/opd_trace_final_tokens", 0.0)) <= 0:
        errors.append("teacher-trace eval scored no final-answer tokens")

    if errors:
        raise RuntimeError("OPD trace validation failed:\n" + "\n".join(f"- {error}" for error in errors))


