#!/usr/bin/env python3
"""Run a few FrozenLake rollouts and write eval-protocol rows for inspection.

Usage:
    python verify_rollout.py [--num-rollouts 4] [--port 8765] [--model-id ...] [--visual]

After it finishes, the vite dashboard opens automatically at http://localhost:<port>/table
with token-level debug data in execution_metadata.extra.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import sys
import time
import logging
import threading
import webbrowser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FROZEN_LAKE_DIR = Path(__file__).resolve().parent
COOKBOOK_DIR = FROZEN_LAKE_DIR.parent.parent
sys.path.insert(0, str(COOKBOOK_DIR))

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Status
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.dataset_logger import default_logger

from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS,
    FrozenLakeToolRolloutProcessor,
)
from training.examples.rl.frozen_lake.frozen_lake_env import build_frozen_lake_tool_env
from training.examples.rl.frozen_lake.masking import compute_model_output_spans, build_ui_token_mask

DEFAULT_LOGS_DB_PATH = str(COOKBOOK_DIR / ".tmp" / "frozen_lake_verify_logs.db")
DEFAULT_REPORT_PATH = str(COOKBOOK_DIR / ".tmp" / "frozen_lake_verify_report.html")
DEFAULT_KIMI_TOKENIZER = "moonshotai/Kimi-K2.5"

DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS


def load_seed_contexts(path: str, max_seeds: int = 4, seed_values: list[int] | None = None):
    wanted_seeds = {int(seed) for seed in (seed_values or [])}
    contexts = []
    with open(path) as f:
        for line in f:
            ctx = json.loads(line)
            if wanted_seeds and int(ctx.get("seed", -1)) not in wanted_seeds:
                continue
            contexts.append(ctx)
            if len(contexts) >= max_seeds:
                break
    return contexts


def _looks_like_kimi_target(value: object) -> bool:
    lowered = str(value or "").strip().lower()
    return "kimi-k2.5" in lowered or "kimi-k2p5" in lowered


def _resolve_tokenizer_model(model_id: str, tokenizer_model: str | None) -> str:
    if tokenizer_model:
        return str(tokenizer_model)
    env_tokenizer = os.environ.get("FROZEN_LAKE_VERIFY_TOKENIZER_MODEL", "").strip()
    if env_tokenizer:
        return env_tokenizer
    if _looks_like_kimi_target(model_id):
        return DEFAULT_KIMI_TOKENIZER
    return "Qwen/Qwen3-8B"


def _validation_target(*, model_id: str, tokenizer_name: str, visual: bool) -> str:
    if visual and (_looks_like_kimi_target(model_id) or _looks_like_kimi_target(tokenizer_name)):
        return "kimi-k2.5-vl"
    if visual:
        return "generic-vlm"
    if _looks_like_kimi_target(model_id) or _looks_like_kimi_target(tokenizer_name):
        return "kimi-k2.5-text"
    return "text"


async def run_rollouts(
    *,
    num_rollouts: int = 4,
    rows_per_seed: int = 1,
    model_id: str = "accounts/fireworks/models/qwen3-8b",
    tokenizer: str | None = None,
    visual: bool = False,
    allow_plaintext_action_fallback: bool = False,
    temperature: float = 0.0,
    seed_values: list[int] | None = None,
):
    api_key = os.environ["FIREWORKS_API_KEY"]
    resolved_tokenizer = _resolve_tokenizer_model(model_id=model_id, tokenizer_model=tokenizer)

    seed_path = str(FROZEN_LAKE_DIR / "seeds.jsonl")
    seed_contexts = load_seed_contexts(seed_path, max_seeds=num_rollouts, seed_values=seed_values)
    logger.info("Loaded %d seeds", len(seed_contexts))

    processor = FrozenLakeToolRolloutProcessor(
        model_id=model_id,
        tokenizer_name_or_path=resolved_tokenizer,
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference",
        temperature=float(temperature),
        max_tokens=128,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        logprobs=True,
        observation_mode="image" if visual else "text",
        allow_plaintext_action_fallback=allow_plaintext_action_fallback,
        max_parse_retries=3 if visual and _looks_like_kimi_target(model_id) else 0,
    )

    config = RolloutProcessorConfig(
        completion_params={"model": model_id, "temperature": float(temperature)},
        mcp_config_path="",
        steps=30,
        semaphore=asyncio.Semaphore(1),
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
            for i in range(rows_per_seed)
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


def _build_row_validation_details(
    row: EvaluationRow,
    *,
    model_id: str,
    tokenizer_name: str,
    visual: bool,
) -> dict[str, object]:
    extra = row.execution_metadata.extra if isinstance(row.execution_metadata.extra, dict) else {}
    token_turn_traces = extra.get("token_turn_traces") or []
    model_request_traces = extra.get("model_request_traces") or []
    tool_call_traces = extra.get("tool_call_traces") or []
    full_episode = extra.get("full_episode") or {}
    prompt_lengths = [len(trace.get("prompt_ids") or []) for trace in token_turn_traces]
    completion_lengths = [len(trace.get("completion_ids") or []) for trace in token_turn_traces]
    image_counts = [trace.get("image_count") for trace in model_request_traces]
    token_ids = list(full_episode.get("token_ids") or [])
    mask = list(full_episode.get("mask") or [])
    logprobs = list(full_episode.get("logprobs") or [])

    validation_checks: list[dict[str, str]] = []
    failing_checks: list[str] = []
    skipped_checks: list[str] = []

    def add_check(name: str, status: str, detail: str) -> None:
        validation_checks.append({"name": name, "status": status, "detail": detail})
        if status == "fail":
            failing_checks.append(name)
        elif status == "skip":
            skipped_checks.append(name)

    rollout_error = str(extra.get("rollout_error") or "").strip()
    add_check(
        "no_rollout_error",
        "pass" if not rollout_error else "fail",
        "row completed without rollout_error" if not rollout_error else rollout_error,
    )

    if visual:
        observation_mode = str(extra.get("observation_mode") or "")
        add_check(
            "observation_mode_image",
            "pass" if observation_mode == "image" else "fail",
            f"observation_mode={observation_mode!r}",
        )

    add_check(
        "token_turn_traces_present",
        "pass" if token_turn_traces else "fail",
        f"turn_count={len(token_turn_traces)}",
    )
    add_check(
        "prompt_ids_present_each_turn",
        "pass" if token_turn_traces and all(trace.get("prompt_ids") for trace in token_turn_traces) else "fail",
        f"prompt_lengths={prompt_lengths}",
    )
    add_check(
        "completion_ids_present_each_turn",
        "pass" if token_turn_traces and all(trace.get("completion_ids") for trace in token_turn_traces) else "fail",
        f"completion_lengths={completion_lengths}",
    )

    if len(token_turn_traces) < 2:
        add_check("multi_turn_continuity", "skip", "requires at least two turns")
        add_check("prompt_prefix_continuity", "skip", "requires at least two turns")
        add_check("completion_replayed_in_next_prompt", "skip", "requires at least two turns")
    else:
        prompt_increase_ok = all(prompt_lengths[idx] > prompt_lengths[idx - 1] for idx in range(1, len(prompt_lengths)))
        add_check(
            "multi_turn_continuity",
            "pass" if prompt_increase_ok else "fail",
            f"prompt_lengths={prompt_lengths}",
        )

        prefix_ok = True
        replay_ok = True
        for idx in range(1, len(token_turn_traces)):
            prev_prompt_ids = [int(x) for x in (token_turn_traces[idx - 1].get("prompt_ids") or [])]
            prev_completion_ids = [int(x) for x in (token_turn_traces[idx - 1].get("completion_ids") or [])]
            current_prompt_ids = [int(x) for x in (token_turn_traces[idx].get("prompt_ids") or [])]
            if current_prompt_ids[: len(prev_prompt_ids)] != prev_prompt_ids:
                prefix_ok = False
            replay_start = len(prev_prompt_ids)
            replay_end = replay_start + len(prev_completion_ids)
            if current_prompt_ids[replay_start:replay_end] != prev_completion_ids:
                replay_ok = False

        add_check(
            "prompt_prefix_continuity",
            "pass" if prefix_ok else "fail",
            "next turn prompt_ids preserve the prior prompt_ids prefix",
        )
        add_check(
            "completion_replayed_in_next_prompt",
            "pass" if replay_ok else "fail",
            "next turn prompt_ids embed the prior completion_ids immediately after the prior prompt_ids",
        )

    request_count_matches = len(model_request_traces) == len(token_turn_traces)
    add_check(
        "model_request_trace_count_matches_turns",
        "pass" if request_count_matches else "fail",
        f"request_traces={len(model_request_traces)} turn_traces={len(token_turn_traces)}",
    )
    add_check(
        "tool_call_trace_count_matches_turns",
        "pass" if len(tool_call_traces) == len(token_turn_traces) else "fail",
        f"tool_call_traces={len(tool_call_traces)} turn_traces={len(token_turn_traces)}",
    )

    if visual:
        image_count_ok = (
            bool(token_turn_traces)
            and len(image_counts) == len(token_turn_traces)
            and image_counts[0] == 1
            and all(
                isinstance(image_counts[idx], int)
                and image_counts[idx] == image_counts[idx - 1] + 1
                for idx in range(1, len(image_counts))
            )
        )
        add_check(
            "image_counts_increment_per_turn",
            "pass" if image_count_ok else "fail",
            f"image_counts={image_counts}",
        )
        prompt_text_present = bool(model_request_traces) and all(
            str(trace.get("prompt_text") or "").strip() for trace in model_request_traces
        )
        add_check(
            "prompt_text_present_for_image_turns",
            "pass" if prompt_text_present else "fail",
            f"prompt_text_turns={sum(1 for trace in model_request_traces if str(trace.get('prompt_text') or '').strip())}",
        )
        assistant_turn_len_ok = bool(model_request_traces) and all(
            int(model_request_traces[idx].get("assistant_turn_len") or 0) == completion_lengths[idx]
            for idx in range(min(len(model_request_traces), len(completion_lengths)))
        )
        add_check(
            "assistant_turn_len_matches_completion_ids",
            "pass" if assistant_turn_len_ok else "fail",
            "assistant_turn_len matches completion_ids length on every turn",
        )
        if _looks_like_kimi_target(model_id) or _looks_like_kimi_target(tokenizer_name):
            kimi_instant_prompt_ok = bool(token_turn_traces) and all(
                "<think></think>" in str(trace.get("prompt_text") or "")
                for trace in token_turn_traces
            )
            add_check(
                "kimi_instant_prompt_contains_empty_think_block",
                "pass" if kimi_instant_prompt_ok else "fail",
                "each Kimi prompt_text should include <think></think> when thinking is disabled",
            )

    add_check(
        "full_episode_token_ids_present",
        "pass" if token_ids else "fail",
        f"token_count={len(token_ids)}",
    )
    add_check(
        "mask_length_matches_token_count",
        "pass" if token_ids and len(mask) == len(token_ids) else "fail",
        f"mask_count={len(mask)} token_count={len(token_ids)}",
    )
    add_check(
        "logprobs_length_matches_token_count",
        "pass" if token_ids and len(logprobs) == len(token_ids) else "fail",
        f"logprobs_count={len(logprobs)} token_count={len(token_ids)}",
    )

    if token_ids and len(logprobs) == len(token_ids):
        generated_positions: set[int] = set()
        for idx, trace in enumerate(token_turn_traces):
            prompt_len = len(trace.get("prompt_ids") or [])
            prefill_len = int((model_request_traces[idx] if idx < len(model_request_traces) else {}).get("assistant_prefill_len") or 0)
            completion_logprobs = trace.get("completion_logprobs") or []
            for offset in range(len(completion_logprobs)):
                generated_positions.add(prompt_len + prefill_len + offset)

        generated_present = all(
            pos < len(logprobs) and logprobs[pos] is not None
            for pos in generated_positions
        )
        prompt_positions_empty = all(
            logprobs[pos] is None
            for pos in range(len(logprobs))
            if pos not in generated_positions
        )
        add_check(
            "generated_logprobs_present",
            "pass" if generated_present else "fail",
            f"generated_positions={sorted(generated_positions)}",
        )
        add_check(
            "prompt_positions_have_no_logprobs",
            "pass" if prompt_positions_empty else "fail",
            "only generated-token positions should carry logprobs",
        )
    else:
        add_check("generated_logprobs_present", "skip", "full_episode.logprobs unavailable")
        add_check("prompt_positions_have_no_logprobs", "skip", "full_episode.logprobs unavailable")

    return {
        "target": _validation_target(model_id=model_id, tokenizer_name=tokenizer_name, visual=visual),
        "passed": not failing_checks,
        "failing_checks": failing_checks,
        "skipped_checks": skipped_checks,
        "turn_count": len(token_turn_traces),
        "prompt_lengths": prompt_lengths,
        "completion_lengths": completion_lengths,
        "image_counts": image_counts,
        "token_count": len(token_ids),
        "mask_count": len(mask),
        "validation_checks": validation_checks,
    }


def enrich_rows(results: list[EvaluationRow], tokenizer_name: str, *, model_id: str, visual: bool):
    """Enrich rows with detokenized tokens, full-episode view, status, and score."""
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Tokenizer load failed for %s: %s", tokenizer_name, exc)
        tokenizer = None

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
            if tokenizer is not None:
                trace["detokenized_tokens"] = [_detok(tokenizer, tid) for tid in all_ids]
                trace["prompt_detokenized_tokens"] = [_detok(tokenizer, tid) for tid in prompt_ids]
                trace["completion_detokenized_tokens"] = [_detok(tokenizer, tid) for tid in completion_ids]
            trace["prompt_len"] = len(prompt_ids)
            trace["completion_len"] = len(completion_ids)

        if token_turn_traces:
            last = token_turn_traces[-1]
            full_ids = [int(x) for x in last["prompt_ids"]] + [int(x) for x in last["completion_ids"]]
            full_len = len(full_ids)

            first_prompt_len = len(token_turn_traces[0].get("prompt_ids") or [])
            mrt = extra.get("model_request_traces") or []
            num_turns = len(token_turn_traces)

            spans = compute_model_output_spans(token_turn_traces, mrt)
            mask = build_ui_token_mask(spans, full_len)

            logprobs_arr: list[float | None] = [None] * full_len
            for k in range(num_turns):
                turn_prompt_len = len(token_turn_traces[k].get("prompt_ids") or [])
                prefill_len = int((mrt[k] if k < len(mrt) else {}).get("assistant_prefill_len") or 0)
                turn_lp = token_turn_traces[k].get("completion_logprobs") or []
                for j, lp in enumerate(turn_lp):
                    pos = turn_prompt_len + prefill_len + j
                    if pos < full_len:
                        logprobs_arr[pos] = lp

            extra["full_episode"] = {
                "token_ids": full_ids,
                "mask": mask,
                "logprobs": logprobs_arr,
                "num_turns": num_turns,
                "first_prompt_len": first_prompt_len,
            }
            if tokenizer is not None:
                extra["full_episode"]["detokenized_tokens"] = [_detok(tokenizer, tid) for tid in full_ids]

        validation_summary = _build_row_validation_details(
            row,
            model_id=model_id,
            tokenizer_name=tokenizer_name,
            visual=visual,
        )
        extra["validation_summary"] = {
            key: value for key, value in validation_summary.items() if key != "validation_checks"
        }
        extra["validation_checks"] = validation_summary["validation_checks"]

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


def configure_default_logger(*, logs_db_path: str, reset_logs: bool = False):
    """Point eval-protocol's lazy default logger at a specific sqlite DB."""
    from eval_protocol.dataset_logger.sqlite_dataset_logger_adapter import SqliteDatasetLoggerAdapter

    adapter = SqliteDatasetLoggerAdapter(db_path=logs_db_path)
    if reset_logs:
        adapter._store.delete_all_rows()
        logger.info("Cleared existing eval-protocol rows from %s", logs_db_path)

    default_logger._logger = adapter
    logger.info("Using eval-protocol sqlite db at %s", logs_db_path)


def _reconstruct_turn_image_urls(row: EvaluationRow) -> list[str]:
    dataset_info = row.input_metadata.dataset_info if isinstance(row.input_metadata.dataset_info, dict) else {}
    env_context = dataset_info.get("environment_context") if isinstance(dataset_info, dict) else {}
    if not isinstance(env_context, dict):
        return []

    extra = row.execution_metadata.extra if isinstance(row.execution_metadata.extra, dict) else {}
    tool_call_traces = extra.get("tool_call_traces") or []
    token_turn_traces = extra.get("token_turn_traces") or []
    if not token_turn_traces:
        return []

    env = build_frozen_lake_tool_env(environment_context=env_context, max_steps=max(1, len(tool_call_traces) + 1))
    env.reset()
    images = [env.render_image_data_url()]
    for trace in tool_call_traces[: max(0, len(token_turn_traces) - 1)]:
        action = trace.get("action")
        if not isinstance(action, str):
            break
        env.step(action)
        images.append(env.render_image_data_url())
    return images[: len(token_turn_traces)]


def _format_token_piece(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.encode("unicode_escape").decode("ascii")


def _mask_class(mask_value: int) -> str:
    if mask_value <= 0:
        return "tok tok-prompt"
    palette_index = ((mask_value - 1) % 6) + 1
    return f"tok tok-turn-{palette_index}"


def _status_chip(status: str) -> str:
    normalized = str(status or "skip").lower()
    if normalized == "pass":
        return "chip chip-pass"
    if normalized == "fail":
        return "chip chip-fail"
    return "chip chip-skip"


def build_debug_report_html(results: list[EvaluationRow]) -> str:
    sections: list[str] = []
    for row in results:
        extra = row.execution_metadata.extra if isinstance(row.execution_metadata.extra, dict) else {}
        full_episode = extra.get("full_episode") or {}
        token_ids = [int(x) for x in (full_episode.get("token_ids") or [])]
        mask = [int(x) for x in (full_episode.get("mask") or [])]
        logprobs = list(full_episode.get("logprobs") or [])
        detok = list(full_episode.get("detokenized_tokens") or [])
        token_turn_traces = extra.get("token_turn_traces") or []
        tool_call_traces = extra.get("tool_call_traces") or []
        model_request_traces = extra.get("model_request_traces") or []
        validation_summary = extra.get("validation_summary") or {}
        validation_checks = extra.get("validation_checks") or []
        image_urls = _reconstruct_turn_image_urls(row)
        rollout_id = ""
        if row.execution_metadata and row.execution_metadata.rollout_id:
            rollout_id = str(row.execution_metadata.rollout_id)
        score = row.evaluation_result.score if row.evaluation_result else None

        token_spans: list[str] = []
        token_rows: list[str] = []
        for idx, token_id in enumerate(token_ids):
            token_text = detok[idx] if idx < len(detok) else ""
            mask_value = mask[idx] if idx < len(mask) else 0
            logprob_value = logprobs[idx] if idx < len(logprobs) else None
            tooltip = html.escape(
                f"idx={idx} id={token_id} mask={mask_value} logprob={logprob_value} text={_format_token_piece(token_text)}"
            )
            token_spans.append(
                f'<span class="{_mask_class(mask_value)}" title="{tooltip}">{html.escape(_format_token_piece(token_text) or "<empty>")}</span>'
            )
            token_rows.append(
                html.escape(
                    f"{idx:05d}  id={token_id:<8} mask={mask_value:<3} logprob={logprob_value!s:<12} text={_format_token_piece(token_text)}"
                )
            )

        turn_rows: list[str] = []
        for turn_index, trace in enumerate(token_turn_traces):
            request_trace = model_request_traces[turn_index] if turn_index < len(model_request_traces) else {}
            tool_trace = tool_call_traces[turn_index] if turn_index < len(tool_call_traces) else {}
            image_html = ""
            if turn_index < len(image_urls):
                image_html = f'<img class="turn-image" src="{image_urls[turn_index]}" alt="turn {turn_index + 1} input image" />'
            turn_meta = html.escape(
                json.dumps(
                    {
                        "image_count": request_trace.get("image_count"),
                        "assistant_turn_len": request_trace.get("assistant_turn_len"),
                        "tool_suffix_len": request_trace.get("tool_suffix_len"),
                        "prompt_token_count": request_trace.get("prompt_token_count"),
                    },
                    indent=2,
                )
            )
            turn_rows.append(
                "\n".join(
                    [
                        "<details class=\"turn\">",
                        f"<summary>Turn {turn_index + 1}: prompt={len(trace.get('prompt_ids') or [])} completion={len(trace.get('completion_ids') or [])} action={html.escape(str(tool_trace.get('action') or ''))} reward={html.escape(str(tool_trace.get('reward') or ''))}</summary>",
                        image_html,
                        f"<pre>{turn_meta}</pre>",
                        f"<pre>{html.escape(str(request_trace.get('prompt_text') or ''))}</pre>",
                        f"<pre>{html.escape(json.dumps(trace, indent=2))}</pre>",
                        "</details>",
                    ]
                )
            )

        validation_rows = [
            (
                f'<li><span class="{_status_chip(check.get("status") or "skip")}">'
                f'{html.escape(str(check.get("status") or "skip").upper())}</span> '
                f'<strong>{html.escape(str(check.get("name") or ""))}</strong>: '
                f'{html.escape(str(check.get("detail") or ""))}</li>'
            )
            for check in validation_checks
        ]
        validation_passed = bool(validation_summary.get("passed"))
        validation_badge = (
            '<span class="chip chip-pass">PASS</span>'
            if validation_passed else '<span class="chip chip-fail">FAIL</span>'
        )

        sections.append(
            "\n".join(
                [
                    "<section class=\"row-block\">",
                    f"<h2>{html.escape(str(row.input_metadata.row_id))}</h2>",
                    f"<p><strong>rollout_id</strong>: {html.escape(rollout_id)} | <strong>mode</strong>: {html.escape(str(extra.get('observation_mode') or ''))} | <strong>score</strong>: {html.escape(str(score))}</p>",
                    f"<p><strong>turns</strong>: {len(token_turn_traces)} | <strong>tokens</strong>: {len(token_ids)} | <strong>error</strong>: {html.escape(str(extra.get('rollout_error') or ''))}</p>",
                    f"<p><strong>validation</strong>: {validation_badge} <strong>target</strong>: {html.escape(str(validation_summary.get('target') or ''))} <strong>image_counts</strong>: {html.escape(str(validation_summary.get('image_counts') or []))}</p>",
                    "<h3>Validation Checks</h3>",
                    "<ul class=\"validation-list\">" + "".join(validation_rows) + "</ul>",
                    "<h3>Masked Token Stream</h3>",
                    "<div class=\"token-stream\">" + "".join(token_spans) + "</div>",
                    "<details><summary>Token Table</summary><pre>" + "\n".join(token_rows) + "</pre></details>",
                    "<h3>Per-Turn Trace</h3>",
                    "\n".join(turn_rows),
                    "</section>",
                ]
            )
        )

    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<title>FrozenLake Visual Rollout Report</title>",
            "<style>",
            "body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; background: #f5f5f0; color: #111827; }",
            "h1, h2, h3 { font-family: ui-sans-serif, system-ui, sans-serif; }",
            ".row-block { margin-bottom: 48px; padding: 20px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 12px; }",
            ".token-stream { line-height: 1.8; word-break: break-word; }",
            ".tok { display: inline-block; margin: 1px; padding: 2px 4px; border-radius: 4px; border: 1px solid transparent; }",
            ".tok-prompt { background: #f3f4f6; color: #374151; }",
            ".tok-turn-1 { background: #dbeafe; }",
            ".tok-turn-2 { background: #dcfce7; }",
            ".tok-turn-3 { background: #fef3c7; }",
            ".tok-turn-4 { background: #fee2e2; }",
            ".tok-turn-5 { background: #ede9fe; }",
            ".tok-turn-6 { background: #fce7f3; }",
            ".chip { display: inline-block; padding: 2px 8px; border-radius: 999px; font: 600 12px/1.8 ui-sans-serif, system-ui, sans-serif; }",
            ".chip-pass { background: #dcfce7; color: #166534; }",
            ".chip-fail { background: #fee2e2; color: #991b1b; }",
            ".chip-skip { background: #e5e7eb; color: #374151; }",
            ".validation-list { padding-left: 20px; }",
            "pre { white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e5e7eb; padding: 12px; border-radius: 8px; overflow-x: auto; }",
            ".turn { margin: 12px 0; }",
            ".turn-image { width: min(420px, 100%); display: block; margin: 12px 0; border: 1px solid #cbd5e1; border-radius: 8px; background: white; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>FrozenLake Visual Rollout Report</h1>",
            "<p>This report is generated from eval-protocol rows after mask enrichment. Prompt tokens are gray; model output tokens are color-coded by turn index.</p>",
            "\n".join(sections),
            "</body>",
            "</html>",
        ]
    )


def write_debug_report(results: list[EvaluationRow], report_path: str):
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(build_debug_report_html(results), encoding="utf-8")
    logger.info("Wrote debug report to %s", report_file)


def _dashboard_urls(port: int) -> tuple[str, str]:
    base = f"http://localhost:{port}"
    return f"{base}/table", f"{base}/frozen-lake-report"


def _schedule_browser_open(port: int, *, report_path: str) -> None:
    table_url, report_url = _dashboard_urls(port)

    def _open() -> None:
        logger.info("Opening logs UI at %s", table_url)
        try:
            webbrowser.open(table_url)
        except Exception as exc:
            logger.warning("Failed to open browser for %s: %s", table_url, exc)
        if Path(report_path).exists():
            logger.info("FrozenLake report available at %s", report_url)

    opener = threading.Timer(1.0, _open)
    opener.daemon = True
    opener.start()


def serve_logs_dashboard(port: int, *, logs_db_path: str, report_path: str, open_browser: bool = False):
    """Start the eval-protocol dashboard against the local sqlite logger.

    We bypass ``serve_logs()`` because the installed ``ep`` path and the default
    uvicorn lifespan startup can both hang in this local environment.
    """
    import uvicorn
    from fastapi.responses import HTMLResponse
    from eval_protocol.utils.browser_utils import write_pid_file
    from eval_protocol.utils.logs_server import LogsServer

    async def _run():
        configure_default_logger(logs_db_path=logs_db_path)
        server = LogsServer(port=port)

        if Path(report_path).exists():
            @server.app.get("/frozen-lake-report")
            async def frozen_lake_report():
                return HTMLResponse(Path(report_path).read_text(encoding="utf-8"))

            report_route = server.app.router.routes.pop()
            server.app.router.routes.insert(0, report_route)

        try:
            server.start_loops()
            config = uvicorn.Config(
                server.app,
                host=server.host,
                port=server.port,
                log_level="info",
                lifespan="off",
            )
            uvicorn_server = uvicorn.Server(config)
            write_pid_file(os.getpid(), server.port)
            await uvicorn_server.serve()
        finally:
            server.evaluation_watcher.stop()
            server.websocket_manager.stop_broadcast_loop()

    table_url, report_url = _dashboard_urls(port)
    logger.info("Starting eval-protocol dashboard on port %d...", port)
    logger.info("Logs table: %s", table_url)
    logger.info("FrozenLake report: %s", report_url)
    if open_browser:
        _schedule_browser_open(port, report_path=report_path)
    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--rows-per-seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--reset-logs", action="store_true")
    parser.add_argument("--logs-db", default=DEFAULT_LOGS_DB_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("FROZEN_LAKE_VERIFY_MODEL_ID", "accounts/fireworks/models/qwen3-8b"),
    )
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--allow-plaintext-action-fallback", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    resolved_tokenizer_model = _resolve_tokenizer_model(
        model_id=args.model_id,
        tokenizer_model=args.tokenizer_model,
    )

    if args.serve_only:
        serve_logs_dashboard(
            args.port,
            logs_db_path=args.logs_db,
            report_path=args.report_path,
            open_browser=args.open_browser,
        )
        return

    configure_default_logger(logs_db_path=args.logs_db, reset_logs=args.reset_logs)

    results = asyncio.run(
        run_rollouts(
            num_rollouts=args.num_rollouts,
            rows_per_seed=args.rows_per_seed,
            model_id=args.model_id,
            tokenizer=resolved_tokenizer_model,
            visual=args.visual,
            allow_plaintext_action_fallback=args.allow_plaintext_action_fallback,
            temperature=args.temperature,
            seed_values=list(args.seed or []),
        )
    )
    if not results:
        logger.error("No rollouts completed")
        return

    logger.info("Enriching rows...")
    enrich_rows(results, resolved_tokenizer_model, model_id=args.model_id, visual=args.visual)
    write_debug_report(results, args.report_path)

    rewards = []
    for row in results:
        extra = row.execution_metadata.extra or {}
        step_rewards = extra.get("step_rewards") or []
        rewards.append(1.0 if step_rewards and float(step_rewards[-1]) > 0 else 0.0)
    if rewards:
        validation_passes = 0
        for row in results:
            extra = row.execution_metadata.extra or {}
            validation_summary = extra.get("validation_summary") or {}
            if validation_summary.get("passed"):
                validation_passes += 1
        logger.info(
            "Rollout summary: n=%d success_rate=%.3f avg_reward=%.3f visual=%s model=%s tokenizer=%s temperature=%.3f validation_pass_rows=%d",
            len(rewards),
            sum(rewards) / len(rewards),
            sum(rewards) / len(rewards),
            args.visual,
            args.model_id,
            resolved_tokenizer_model,
            float(args.temperature),
            validation_passes,
        )

    write_to_default_logger(results)

    if args.no_serve:
        logger.info("Done (--no-serve). Data written to default logger.")
        return

    serve_logs_dashboard(
        args.port,
        logs_db_path=args.logs_db,
        report_path=args.report_path,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()
