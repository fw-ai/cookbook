#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnose GLM-5.1 SFT datasets that produce thinking-loop behaviour at inference.

Renders every row through the registered ``glm5`` renderer (same path the
SFT loop uses) and flags shapes that train the model to enter ``<think>``
without closing it, drop ``<|endoftext|>``, or render an OOD prefix the
inference deployment would never feed back in.

What it checks per rendered training example
============================================

- ``<think>`` / ``</think>`` balance inside the trained span. An open
  ``<think>`` without a closing ``</think>`` teaches ``P(<|endoftext|> | <think>) > 0``
  *without* teaching ``</think>`` first, which is the textbook setup for
  the model to keep generating reasoning until ``max_tokens`` cuts it off.
- Presence of ``<|endoftext|>`` (id 154820) in the trained tokens. Without
  it, the model never learns to stop and the deployment will run until
  ``max_tokens`` even on simple answers.
- Whether ``<|endoftext|>`` is the final trained token (sanity check on
  ``max_seq_len`` truncation).
- Empty vs non-empty think distribution across the dataset. A 100% empty
  think distribution means the SFT signal has to overpower GLM-5.1's
  strong reasoning prior; if the dataset is small, the prior wins and
  the model still emits reasoning at inference. If you do not want
  reasoning, this is a *signal-strength* problem, not a renderer bug.
- Unbalanced ``<think>`` *substrings* in the raw ``content`` string of any
  assistant message (catches hand-edited rows where someone left a stray
  ``<think>`` in ``content`` instead of in ``reasoning_content``).

What this script does *not* check (do these on the deployment side)
==================================================================

- That the deployment honours ``<|endoftext|>`` (id 154820) as a stop
  token. Verify via ``stop`` in your sampler config or the served
  model's ``generation_config.json``.
- That the deployment's generation prompt ends in ``<|assistant|><think>``
  (matches ``apply_chat_template(add_generation_prompt=True,
  enable_thinking=True)``). For ``/v1/chat/completions`` on the Fireworks
  ``glm-5p1`` serverless deployment this is the default; if you call
  ``/v1/completions`` and construct your own prompt, this is on you.
- That the deployment tokenizer encodes ``<think>`` as the special token
  id 154841 (and ``</think>`` as 154842). To check, call the deployment's
  tokenizer endpoint or compare ``len(tok.encode("<|assistant|><think>"))``;
  it must be 2 tokens.

Usage
=====

::

    # in the cookbook venv (training deps installed)
    python training/examples/tools/check_glm5_sft_dataset.py \\
        --dataset path/to/sft.jsonl \\
        --train-on-what all_assistant_messages \\
        --max-bad-samples 10

    # JSON output for piping into other tooling
    python training/examples/tools/check_glm5_sft_dataset.py \\
        --dataset path/to/sft.jsonl --json

    # Sample only the first N rows for a quick smoke test
    python training/examples/tools/check_glm5_sft_dataset.py \\
        --dataset path/to/sft.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Iterator

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from transformers import AutoTokenizer

from training.renderer.glm5 import GLM5Renderer  # noqa: F401  (registers "glm5")
from training.utils import supervised as _supervised
from training.utils.supervised import normalize_messages, render_messages_to_datum
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat

# ``render_messages_to_datums`` (plural) was added in cookbook PR #397 to
# support renderer-side splitting of multi-turn examples. Fall back to the
# singular form when running against an older cookbook so this diagnostic
# is usable while triaging an incident on either side of that PR.
_render_to_datums = getattr(_supervised, "render_messages_to_datums", None)


def _render_one(messages, *, renderer, train_on_what):
    if _render_to_datums is not None:
        return list(
            _render_to_datums(messages, renderer=renderer, train_on_what=train_on_what)
        )
    return [
        render_messages_to_datum(
            messages, renderer=renderer, train_on_what=train_on_what
        )
    ]


# GLM-5.1 special-token ids. Stable across the ``zai-org/GLM-5.1`` and
# ``zai-org/GLM-5.1-FP8`` checkpoints (they ship identical tokenizers).
_DEFAULT_TOKENIZER = "zai-org/GLM-5.1"
_OPEN_THINK_TOKEN = "<think>"
_CLOSE_THINK_TOKEN = "</think>"


def _iter_jsonl(path: str, limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"row {i}: invalid JSON ({e})") from e
            yield i, row


def _raw_assistant_think_imbalance(messages: list[dict[str, Any]]) -> int:
    """Count assistant messages whose ``content`` string has unbalanced ``<think>``.

    Catches rows where someone left a literal ``<think>`` in ``content``
    without a matching ``</think>``. The renderer would faithfully emit
    that imbalance; flagging here points at the dataset, not the renderer.
    """
    bad = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if c.count("<think>") != c.count("</think>"):
            bad += 1
    return bad


def _classify_example(
    token_ids: list[int],
    open_id: int,
    close_id: int,
    eos_id: int,
) -> dict[str, Any]:
    o = token_ids.count(open_id)
    c = token_ids.count(close_id)
    e = token_ids.count(eos_id)
    classification = {
        "n_open_think": o,
        "n_close_think": c,
        "n_eos": e,
        "ends_with_eos": bool(token_ids and token_ids[-1] == eos_id),
        "unbalanced_think": o != c,
        "no_eos": e == 0,
    }
    if o == 1 and c == 1:
        # Find the close right after the open => empty think block
        try:
            idx = token_ids.index(open_id)
            classification["empty_think"] = token_ids[idx + 1] == close_id
        except (IndexError, ValueError):
            classification["empty_think"] = False
    else:
        classification["empty_think"] = False
    return classification


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, help="Path to SFT JSONL file.")
    p.add_argument(
        "--tokenizer",
        default=_DEFAULT_TOKENIZER,
        help=f"HF model id for tokenizer (default: {_DEFAULT_TOKENIZER}).",
    )
    p.add_argument(
        "--train-on-what",
        default="all_assistant_messages",
        choices=[v.value for v in TrainOnWhat],
        help="train_on_what mode used by the SFT loop (default: all_assistant_messages).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only inspect the first N rows.",
    )
    p.add_argument(
        "--max-bad-samples",
        type=int,
        default=10,
        help="Print at most N offending rows (default: 10).",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print results as a single JSON object on stdout.",
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    renderer = get_renderer("glm5", tok)

    open_id = tok.encode(_OPEN_THINK_TOKEN, add_special_tokens=False)
    close_id = tok.encode(_CLOSE_THINK_TOKEN, add_special_tokens=False)
    eos_id = getattr(tok, "eos_token_id", None)
    if len(open_id) != 1 or len(close_id) != 1 or eos_id is None:
        raise SystemExit(
            "Tokenizer does not encode <think>/</think>/<|endoftext|> as single "
            "special tokens. This script only supports the GLM-5.1 tokenizer."
        )
    open_id, close_id = open_id[0], close_id[0]

    train_on_what = TrainOnWhat(args.train_on_what)

    stats: Counter[str] = Counter()
    bad_rows: list[dict[str, Any]] = []

    for row_idx, row in _iter_jsonl(args.dataset, args.limit):
        messages = row.get("messages", [])
        if not messages:
            stats["empty_messages"] += 1
            continue

        stats["rows"] += 1

        if _raw_assistant_think_imbalance(messages) > 0:
            stats["raw_content_unbalanced_think"] += 1

        try:
            normalized = normalize_messages(messages)
        except Exception as e:
            stats["normalize_errors"] += 1
            if len(bad_rows) < args.max_bad_samples:
                bad_rows.append({"row": row_idx, "error": f"normalize: {e}"})
            continue

        try:
            examples = _render_one(
                normalized,
                renderer=renderer,
                train_on_what=train_on_what,
            )
        except Exception as e:
            stats["render_errors"] += 1
            if len(bad_rows) < args.max_bad_samples:
                bad_rows.append({"row": row_idx, "error": f"render: {e}"})
            continue

        stats["rendered_examples"] += len(examples)

        for split_idx, ex in enumerate(examples):
            ids = list(ex.token_ids)
            cls = _classify_example(ids, open_id, close_id, eos_id)

            if cls["unbalanced_think"]:
                stats["unbalanced_think"] += 1
            if cls["no_eos"]:
                stats["no_eos"] += 1
            if cls["n_eos"] > 0 and not cls["ends_with_eos"]:
                stats["eos_not_last"] += 1
            if cls["empty_think"]:
                stats["empty_think"] += 1
            elif cls["n_open_think"] >= 1 and cls["n_close_think"] >= 1:
                stats["nonempty_think"] += 1

            if (
                cls["unbalanced_think"] or cls["no_eos"]
            ) and len(bad_rows) < args.max_bad_samples:
                bad_rows.append(
                    {
                        "row": row_idx,
                        "split": split_idx,
                        **cls,
                        # Show a short prefix/suffix of the trained tokens for triage.
                        "decoded_head": tok.decode(ids[:80]),
                        "decoded_tail": tok.decode(ids[-80:]),
                    }
                )

    rendered = stats.get("rendered_examples", 0)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "train_on_what": args.train_on_what,
        "rows_inspected": stats.get("rows", 0),
        "rendered_examples": rendered,
        "counts": {k: stats[k] for k in sorted(stats)},
    }
    if rendered:
        summary["rates"] = {
            "unbalanced_think": stats["unbalanced_think"] / rendered,
            "no_eos": stats["no_eos"] / rendered,
            "empty_think": stats["empty_think"] / rendered,
            "nonempty_think": stats["nonempty_think"] / rendered,
        }
    summary["bad_samples"] = bad_rows

    # Compute the verdict before emitting JSON so the exit code is consistent
    # across JSON and human-readable modes (0 = clean, 2 = issues found).
    has_issues = (
        stats.get("unbalanced_think", 0) > 0
        or stats.get("no_eos", 0) > 0
        or stats.get("raw_content_unbalanced_think", 0) > 0
        or (rendered and stats.get("empty_think", 0) / rendered > 0.95)
    )

    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if has_issues else 0

    print("=" * 72)
    print(f"GLM-5.1 SFT dataset diagnostic: {args.dataset}")
    print("=" * 72)
    print(f"Tokenizer:       {args.tokenizer}")
    print(f"train_on_what:   {args.train_on_what}")
    print(f"Rows inspected:  {stats.get('rows', 0)}")
    print(f"Rendered datums: {rendered}")
    print()
    if rendered == 0:
        print("No rendered examples — dataset is empty or every row failed.")
        return 1

    print("Per-example signals (counts; rates in parentheses):")
    for key in (
        "unbalanced_think",
        "no_eos",
        "eos_not_last",
        "empty_think",
        "nonempty_think",
        "raw_content_unbalanced_think",
        "normalize_errors",
        "render_errors",
    ):
        n = stats.get(key, 0)
        rate = n / rendered if rendered else 0.0
        print(f"  {key:35s} {n:8d}   ({rate:6.1%})")
    print()

    # Verdict block. Tuned thresholds are deliberately loud — false positives
    # are cheaper than missing the actual cause of a thinking-loop incident.
    # Keep the trigger conditions in sync with ``has_issues`` above.
    issues: list[str] = []
    if stats.get("unbalanced_think", 0) > 0:
        issues.append(
            "* Some rows train an open <think> with no matching </think>. "
            "These directly teach the model to enter reasoning without learning "
            "to close it — the textbook thinking-loop signal. Filter or "
            "sanitize these rows before re-running SFT."
        )
    if stats.get("no_eos", 0) > 0:
        issues.append(
            "* Some rows render with no <|endoftext|> in the trained span. "
            "The model never learns to stop. Inspect for max_seq_len truncation "
            "or empty assistant content."
        )
    if rendered and stats.get("empty_think", 0) / rendered > 0.95:
        issues.append(
            "* >95% of examples render with an empty <think></think> block. "
            "If you see thinking loops at inference, the SFT signal is fighting "
            "GLM-5.1's strong reasoning prior. Consider: (a) adding "
            "reasoning_content to the dataset, or (b) deploying with "
            "enable_thinking=False (gen prompt <|assistant|></think>) — but the "
            "current registered glm5 renderer always trains the thinking-mode "
            "prefix, so option (b) requires a renderer/deployment alignment "
            "change, not just an inference flag flip."
        )
    if stats.get("raw_content_unbalanced_think", 0) > 0:
        issues.append(
            "* Some assistant messages have unbalanced <think>/</think> "
            "substrings in their raw 'content' field. Move reasoning to the "
            "top-level 'reasoning_content' field, or close the <think> blocks."
        )

    if not issues:
        print("No dataset-level smoking guns found.")
        print(
            "If the deployment still loops, check the deployment-side items "
            "listed in the script docstring (stop tokens, generation prompt, "
            "tokenizer alignment)."
        )
    else:
        print("Likely contributors to thinking-loop behaviour:")
        for line in issues:
            print(line)

    if bad_rows:
        print()
        print(f"Sample offending rows (showing up to {args.max_bad_samples}):")
        for b in bad_rows:
            print(f"  - {b}")

    return 2 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
