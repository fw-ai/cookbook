#!/usr/bin/env python3
"""Prepare GSM8K-style rows for privileged-context OPD.

The OPD loop consumes normalized rows:

  {
    "messages": [...],          # student prompt, no privileged solution
    "teacher_messages": [...],  # same problem plus privileged worked solution
    "expected_answer": "..."
  }

This script converts the public GSM8K sample used elsewhere in the cookbook
into that format. The student sees only the problem. The frozen teacher sees
the problem plus the dataset's worked solution as privileged context.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


DEFAULT_SOURCE = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"
DEFAULT_OUTPUT = Path(__file__).with_name("dataset.jsonl")

STUDENT_SYSTEM_PROMPT = (
    "You are a careful math reasoner. Solve the problem using normal reasoning. "
    "End with exactly one line: Final: <answer>."
)

TEACHER_SYSTEM_PROMPT = (
    "You are a careful math reasoner. Use the privileged worked solution as private context. "
    "Solve the problem with normal reasoning. End with exactly one line: Final: {answer}."
)


def normalize_answer(answer: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(answer).strip())
    return normalized.rstrip(".")


def extract_ground_truth_answer(ground_truth: Any) -> str:
    text = str(ground_truth or "")
    answer_tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_tag_match:
        return normalize_answer(answer_tag_match.group(1))

    gsm8k_match = re.search(r"####\s*([^\n\r]+)", text)
    if gsm8k_match:
        return normalize_answer(gsm8k_match.group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("ground_truth does not contain a final answer")
    return normalize_answer(lines[-1])


def load_jsonl(path_or_url: str) -> list[dict[str, Any]]:
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        lines = response.text.splitlines()
    else:
        lines = Path(path_or_url).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def problem_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row is missing messages")
    kept = [
        dict(message)
        for message in messages
        if str(message.get("role", "")).lower() not in {"assistant", "system"}
    ]
    if not kept:
        raise ValueError("row does not contain a user problem message")
    return kept


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    ground_truth = row.get("ground_truth")
    if ground_truth is None:
        raise ValueError("row is missing ground_truth")
    answer = extract_ground_truth_answer(ground_truth)
    problem = problem_messages(row)

    return {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            *problem,
        ],
        "teacher_messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT.format(answer=answer)},
            *problem,
            {
                "role": "user",
                "content": (
                    "Privileged worked solution for the teacher only:\n"
                    f"{str(ground_truth).strip()}\n\n"
                    "Use this privileged context when scoring or answering. "
                    f"The final line must be exactly: Final: {answer}."
                ),
            },
        ],
        "expected_answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K privileged OPD JSONL")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.source)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    converted = [convert_row(row) for row in rows]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote {len(converted)} rows to {output}")


if __name__ == "__main__":
    main()
