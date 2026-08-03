"""Wordle LLM eval harness (single .py, run via pytest).

Mirrors training/examples/benchmarks/tau2_airline_multiturn_eval.py:
@evaluation_test + MCPGymRolloutProcessor, one HTTP MCP server
(wordle_mcp_server.py) for the whole run, one MCP session per row.
Three models x six word lengths x four games = 72 rollouts.

Run:
    cd /Users/sinan/cookbook
    pytest -q training/examples/benchmarks/wordle_eval.py -s

Or shrink to a smoke test via env:
    EP_MAX_ROWS=2 pytest -q training/examples/benchmarks/wordle_eval.py -s

Requires FIREWORKS_API_KEY and OPENROUTER_API_KEY (see training/.env).
"""

import hashlib
import json
import os
import sys
import csv
import threading
from pathlib import Path
from typing import Any, Dict, List

from eval_protocol import evaluation_test
from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata

try:
    from eval_protocol.pytest import MCPGymRolloutProcessor as _RolloutProcessor
except Exception:
    from eval_protocol.pytest import default_mcp_gym_rollout_processor as _RolloutProcessor

# Ensure eval_protocol's "python <server_script>" subprocess resolves to this
# same interpreter (where mcp / eval_protocol are installed).
_shim_dir = Path(__file__).resolve().parent / ".python_shim"
_shim_dir.mkdir(parents=True, exist_ok=True)
_shim_path = _shim_dir / "python"
_shim_path.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
os.chmod(_shim_path, 0o755)
os.environ["PATH"] = f"{_shim_dir}:{os.environ.get('PATH', '')}"

# ---- Config ----
MODELS: Dict[str, str] = {
    "GLM-5.2 (Fireworks)": "fireworks_ai/accounts/fireworks/models/glm-5p2",
    "Opus 4.8 (OpenRouter)": "openrouter/anthropic/claude-opus-4.8",
    "GPT-5 (OpenRouter)": "openrouter/openai/gpt-5",
}
# Smoke knob: set WORDLE_MODEL="<label substring>" to run only that model
# (e.g. WORDLE_MODEL=GLM).
_ACTIVE_MODELS = {
    label: model
    for label, model in MODELS.items()
    if (not os.environ.get("WORDLE_MODEL")) or (os.environ["WORDLE_MODEL"].lower() in label.lower())
}
if not _ACTIVE_MODELS:
    raise SystemExit(
        f"WORDLE_MODEL={os.environ.get('WORDLE_MODEL')!r} matched no model in {list(MODELS)}"
    )

WORD_LENGTHS = [5, 6, 7, 8, 9, 10]
GAMES_PER_COMBINATION = 4
MAX_GUESSES = 6
ROLLOUT_STEPS = MAX_GUESSES + 1  # 6 guesses * 2 turns + slack; hard backstop (enforced by MCPGymRolloutProcessor)
MAX_CONCURRENT_ROLLOUTS = 8
MAX_TOKENS = None
REQUEST_TIMEOUT_S = 120

BENCH_DIR = Path(__file__).resolve().parent
WORDS_PATH = BENCH_DIR / "wordle_words.json"
SERVER_SCRIPT_PATH = BENCH_DIR / "wordle_mcp_server.py"
DATASET_PATH = BENCH_DIR / "wordle_eval_dataset.jsonl"
METRICS_PATH = BENCH_DIR / "wordle_game_metrics.csv"

assert WORDS_PATH.exists(), f"Missing word list: {WORDS_PATH}"
assert SERVER_SCRIPT_PATH.exists(), f"Missing server: {SERVER_SCRIPT_PATH}"

_metrics_lock = threading.Lock()
_wins_by_key: Dict[str, int] = {}
_games_by_key: Dict[str, int] = {}

with METRICS_PATH.open("w", encoding="utf-8", newline="") as _f:
    _w = csv.writer(_f)
    _w.writerow(
        [
            "row_id",
            "model",
            "word_length",
            "seed",
            "tokens_out_including_reasoning",
            "latency_seconds",
            "win_rate",
            "submitted_words_csv",
            "result",
        ]
    )

SYSTEM_PROMPT = """\
You are playing Wordle. The secret word has a known number of letters (given below).
You have 6 guesses. After each guess you receive per-letter feedback:
  🟩 green  = that letter is in the secret word AND in the correct position
  🟨 yellow = that letter is in the secret word but in a different position
  ⬜ gray   = that letter is not in the secret word

Guesses must be real words of the exact required length, drawn from common English words.
Use the submit_guess tool to submit each guess. Reason briefly (up to 500 words in your reasoning every time) about your guess, then call
submit_guess. Keep calling the submit_guess tool until you see "GAME OVER".
"""

USER_PROMPT_TEMPLATE = (
    "The secret word has {length} letters. You have {max_guesses} guesses. "
    "Make your first guess by calling the submit_guess tool."
)


def _row_seed(model_label: str, length: int, game_idx: int) -> int:
    h = hashlib.sha256(f"{model_label}|{length}|{game_idx}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _build_dataset() -> None:
    """Emit one row per (active model, length, game).

    Defaults to 3*6*4 = 72 rows. Set WORDLE_MODEL="<substring>" to emit rows
    for a single model only (e.g. WORDLE_MODEL=GLM -> 24 rows).
    """
    rows: List[Dict[str, Any]] = []
    for model_label in _ACTIVE_MODELS:
        for length in WORD_LENGTHS:
            for game_idx in range(GAMES_PER_COMBINATION):
                seed = _row_seed(model_label, length, game_idx)
                rows.append(
                    {
                        "id": f"wordle-{model_label}-{length}-{game_idx}".lower().replace(" ", "-").replace("(", "").replace(")", ""),
                        "model_label": model_label,
                        "length": length,
                        "game_idx": game_idx,
                        "seed": seed,
                        "max_guesses": MAX_GUESSES,
                        "valid_words_path": str(WORDS_PATH),
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt_template": USER_PROMPT_TEMPLATE,
                    }
                )
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"[wordle] wrote {len(rows)} rows -> {DATASET_PATH}")


_build_dataset()


def wordle_to_evaluation_row(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows: List[EvaluationRow] = []
    for item in data:
        length = int(item["length"])
        seed = int(item["seed"])
        environment_context = {
            "seed": seed,
            "length": length,
            "max_guesses": int(item.get("max_guesses", MAX_GUESSES)),
            "valid_words_path": item["valid_words_path"],
        }
        rows.append(
            EvaluationRow(
                messages=[{"role": "system", "content": item["system_prompt"]}],
                input_metadata=InputMetadata(
                    row_id=item["id"],
                    dataset_info={
                        "environment_context": environment_context,
                        "user_prompt_template": item["user_prompt_template"],
                        "model_label": item["model_label"],
                        "length": length,
                        "seed": seed,
                    },
                ),
            )
        )
    return rows


# completion_params: a list with one entry per model. eval-protocol parametrizes
# the test into one experiment per entry and runs every dataset row under each.
# Let all models use provider-default reasoning behavior for fairness.
# Keep drop_params=True for provider compatibility.
# Opus 4.8 (via OpenRouter) rejects an explicit temperature, so we keep
# temperature=None there.
#
def _completion_params() -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    for model_label, model in _ACTIVE_MODELS.items():
        cp: Dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "timeout": REQUEST_TIMEOUT_S,
            "extra_body": {"drop_params": True},
        }
        if model_label.startswith("Opus"):
            cp["temperature"] = None
        else:
            cp["temperature"] = 0.0
        params.append(cp)
    return params


def grade_game(row: EvaluationRow) -> EvaluateResult:
    """Substring-grade the rollout. Win if any submit_guess tool result contains
    "You won"; loss if "Out of guesses"; 0 if the model never called submit_guess."""
    dataset_info = row.input_metadata.dataset_info if row.input_metadata else {}
    length = dataset_info.get("length", "?")
    seed = dataset_info.get("seed")

    tool_results: List[str] = []
    submit_guess_calls = 0
    for msg in row.messages:
        if msg.role == "assistant" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if getattr(tc.function, "name", None) == "submit_guess":
                    submit_guess_calls += 1
        if msg.role == "tool":
            content = msg.content or ""
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except Exception:
                    content = str(content)
            tool_results.append(content)

    blob = "\n".join(tool_results)
    secret = None
    # parse 'Out of guesses. The word was "secret". GAME OVER.'
    import re

    m = re.search(r'The word was "([^"]+)"', blob)
    if m:
        secret = m.group(1)

    if "You won" in blob:
        # count guesses used = number of submit_guess calls up to and including the winning one
        guesses = submit_guess_calls
        return EvaluateResult(
            score=1.0,
            reason=f"won in {guesses} guesses (length={length}, seed={seed}, secret={secret})",
        )
    if "Out of guesses" in blob:
        guesses = submit_guess_calls
        return EvaluateResult(
            score=0.0,
            reason=f"lost after {guesses} guesses (length={length}, seed={seed}, secret={secret})",
        )
    if submit_guess_calls == 0:
        return EvaluateResult(
            score=0.0,
            reason=f"no tool use: model never called submit_guess (length={length}, seed={seed})",
        )
    return EvaluateResult(
        score=0.0,
        reason=(
            f"incomplete: {submit_guess_calls} submit_guess call(s) but no GAME OVER "
            f"(length={length}, seed={seed})"
        ),
    )


def _extract_submitted_words(row: EvaluationRow) -> List[str]:
    words: List[str] = []
    for msg in row.messages:
        if msg.role != "assistant":
            continue
        for tc in (getattr(msg, "tool_calls", None) or []):
            if getattr(getattr(tc, "function", None), "name", None) != "submit_guess":
                continue
            raw_args = getattr(getattr(tc, "function", None), "arguments", None)
            if not raw_args:
                continue
            try:
                parsed = json.loads(raw_args)
                word = parsed.get("word")
                if word:
                    words.append(str(word))
            except Exception:
                continue
    return words


def _extract_tokens_out_including_reasoning(row: EvaluationRow) -> int | None:
    usage = getattr(getattr(row, "execution_metadata", None), "usage", None)
    if usage is None:
        return None
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)
    if completion_tokens is not None:
        # completion/output tokens are already provider-reported output tokens;
        # when providers expose reasoning separately it is typically a subset.
        return int(completion_tokens)
    reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        return int(reasoning_tokens)
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        detail_reasoning = (
            details.get("reasoning_tokens")
            if isinstance(details, dict)
            else getattr(details, "reasoning_tokens", None)
        )
        if detail_reasoning is not None:
            return int(detail_reasoning)
    return None


def _extract_latency_seconds(row: EvaluationRow) -> float | None:
    exec_meta = getattr(row, "execution_metadata", None)
    if exec_meta is None:
        return None
    for field in ("rollout_duration_seconds", "duration_seconds", "total_duration_seconds"):
        value = getattr(exec_meta, field, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return None


def _result_label(result: EvaluateResult) -> str:
    reason = (result.reason or "").lower()
    if "won in" in reason:
        return "won"
    if "lost after" in reason:
        return "lost"
    if "no tool use" in reason:
        return "no_tool_use"
    return "incomplete"


def _record_game_metrics(row: EvaluationRow) -> None:
    dataset_info = row.input_metadata.dataset_info if row.input_metadata else {}
    completion_params = row.input_metadata.completion_params if row.input_metadata else {}
    row_id = row.input_metadata.row_id if row.input_metadata and row.input_metadata.row_id else "unknown"
    model = str((completion_params or {}).get("model", "unknown"))
    word_length = dataset_info.get("length")
    seed = dataset_info.get("seed")

    words_submitted = _extract_submitted_words(row)
    tokens_out = _extract_tokens_out_including_reasoning(row)
    latency_s = _extract_latency_seconds(row)
    result = row.evaluation_result
    assert result is not None
    result_label = _result_label(result)
    win = 1 if result_label == "won" else 0

    key = f"{model}|{word_length}"
    with _metrics_lock:
        _wins_by_key[key] = _wins_by_key.get(key, 0) + win
        _games_by_key[key] = _games_by_key.get(key, 0) + 1
        win_rate = _wins_by_key[key] / _games_by_key[key]

        with METRICS_PATH.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    row_id,
                    model,
                    word_length,
                    seed,
                    tokens_out,
                    latency_s,
                    f"{win_rate:.4f}",
                    ",".join(words_submitted),
                    result_label,
                ]
            )


@evaluation_test(
    input_dataset=[str(DATASET_PATH)],
    dataset_adapter=wordle_to_evaluation_row,
    completion_params=_completion_params(),
    rollout_processor=_RolloutProcessor() if callable(_RolloutProcessor) else _RolloutProcessor,
    num_runs=1,
    mode="pointwise",
    max_concurrent_rollouts=MAX_CONCURRENT_ROLLOUTS,
    steps=ROLLOUT_STEPS,
    server_script_path=str(SERVER_SCRIPT_PATH),
)
def test_wordle(row: EvaluationRow) -> EvaluationRow:
    result = grade_game(row)
    row.evaluation_result = result
    _record_game_metrics(row)
    dataset_info = row.input_metadata.dataset_info if row.input_metadata else {}
    row_id = row.input_metadata.row_id if row.input_metadata and row.input_metadata.row_id else "?"
    if os.environ.get("WORDLE_PRINT_RAW_MESSAGES") == "1":
        lines: List[str] = [f"[wordle][{row_id}] messages_after_system:"]
        for msg in row.messages:
            if msg.role == "system":
                continue
            role = msg.role or "unknown"
            lines.append(f"{role}:")

            if role == "assistant":
                provider_fields = getattr(msg, "provider_specific_fields", None)
                reasoning = None
                if isinstance(provider_fields, dict):
                    reasoning = provider_fields.get("reasoning") or provider_fields.get("thinking")
                if reasoning is not None:
                    lines.append(f"    reasoning: {json.dumps(reasoning, ensure_ascii=False)}")
                else:
                    lines.append("    reasoning: <none>")

                content = msg.content if msg.content is not None else ""
                lines.append(f"    content: {content!r}")

                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    for tc in tool_calls:
                        fn_name = getattr(getattr(tc, "function", None), "name", None)
                        fn_args = getattr(getattr(tc, "function", None), "arguments", None)
                        lines.append(f"    tool_call: {fn_name}({fn_args})")
                else:
                    lines.append("    tool_call: <none>")
            elif role == "tool":
                lines.append(f"    tool_call_id: {getattr(msg, 'tool_call_id', None)!r}")
                lines.append(f"    content: {msg.content!r}")
            else:
                lines.append(f"    content: {msg.content!r}")

        print("\n".join(lines))
    print(
        f"[wordle][{row_id}] score={result.score:.1f} reason={result.reason}"
    )
    return row


if __name__ == "__main__":
    # Convenience: run pytest on this file.
    import subprocess

    rc = subprocess.call(
        [sys.executable, "-m", "pytest", "-q", str(Path(__file__).resolve()), "-s"],
        cwd=str(BENCH_DIR),
    )
    sys.exit(rc)
