"""OpenAI Responses API judge for HealthBench Professional."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from .scoring import RubricItem, coerce_rubric_items, score_response
    from .trajectory import load_atif_trajectory
except ImportError:  # Standalone copy generated under a Harbor task's /tests.
    from scoring import RubricItem, coerce_rubric_items, score_response
    from trajectory import load_atif_trajectory

CANONICAL_JUDGE_MODEL = "gpt-5.4-2026-03-05"
CANONICAL_REASONING_EFFORT = "low"
DEFAULT_MAX_ATTEMPTS = 3
MAX_ALLOWED_ATTEMPTS = 5
DEFAULT_MAX_WORKERS = 8
MAX_ALLOWED_WORKERS = 32

# Copied verbatim from OpenAI simple-evals healthbench_eval.py. HealthBench
# Professional uses the same rubric-item judge contract.
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class ResponsesClient(Protocol):
    responses: Any


@dataclass(frozen=True)
class JudgeConfig:
    model: str = CANONICAL_JUDGE_MODEL
    reasoning_effort: str = CANONICAL_REASONING_EFFORT
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    retry_delay_seconds: float = 1.0
    max_workers: int = DEFAULT_MAX_WORKERS

    def __post_init__(self) -> None:
        if self.model != CANONICAL_JUDGE_MODEL:
            raise ValueError(
                f"HealthBench Professional judge must use {CANONICAL_JUDGE_MODEL}"
            )
        if self.reasoning_effort != CANONICAL_REASONING_EFFORT:
            raise ValueError(
                "HealthBench Professional judge must use low reasoning effort"
            )
        if not 1 <= self.max_attempts <= MAX_ALLOWED_ATTEMPTS:
            raise ValueError(
                f"max_attempts must be between 1 and {MAX_ALLOWED_ATTEMPTS}"
            )
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if not 1 <= self.max_workers <= MAX_ALLOWED_WORKERS:
            raise ValueError(f"max_workers must be between 1 and {MAX_ALLOWED_WORKERS}")


def parse_json_to_dict(json_string: str) -> dict[str, Any]:
    """Parse the judge's JSON object, allowing only an outer markdown fence."""

    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        value = json.loads(json_cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("HealthBench judge returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("HealthBench judge response must be a JSON object")
    if not isinstance(value.get("criteria_met"), bool):
        raise ValueError("HealthBench judge response must contain boolean criteria_met")
    explanation = value.get("explanation")
    if not isinstance(explanation, str):
        raise ValueError("HealthBench judge response must contain string explanation")
    return {"explanation": explanation, "criteria_met": value["criteria_met"]}


def render_conversation(
    messages: Sequence[Mapping[str, Any]], response_text: str
) -> str:
    conversation = [*messages, {"role": "assistant", "content": response_text}]
    rendered: list[str] = []
    for index, message in enumerate(conversation):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"Conversation message {index} requires string role and content"
            )
        rendered.append(f"{role}: {content}")
    return "\n\n".join(rendered)


def build_grader_prompt(
    messages: Sequence[Mapping[str, Any]], response_text: str, rubric_item: RubricItem
) -> str:
    return GRADER_TEMPLATE.replace(
        "<<conversation>>", render_conversation(messages, response_text)
    ).replace("<<rubric_item>>", str(rubric_item))


def _extract_output_text(response: Any) -> str:
    output_text = (
        response.get("output_text")
        if isinstance(response, Mapping)
        else getattr(response, "output_text", None)
    )
    if isinstance(output_text, str):
        return output_text

    output = (
        response.get("output")
        if isinstance(response, Mapping)
        else getattr(response, "output", None)
    )
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = (
                item.get("content")
                if isinstance(item, Mapping)
                else getattr(item, "content", None)
            )
            if not isinstance(content, list):
                continue
            for part in content:
                text = (
                    part.get("text")
                    if isinstance(part, Mapping)
                    else getattr(part, "text", None)
                )
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "".join(texts)
    raise ValueError("HealthBench judge response did not contain output text")


def judge_rubric_item(
    client: ResponsesClient,
    messages: Sequence[Mapping[str, Any]],
    response_text: str,
    rubric_item: RubricItem,
    *,
    config: JudgeConfig = JudgeConfig(),
) -> dict[str, Any]:
    """Grade exactly one rubric item with a bounded retry loop."""

    prompt = build_grader_prompt(messages, response_text, rubric_item)
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = client.responses.create(
                model=config.model,
                reasoning={"effort": config.reasoning_effort},
                input=[{"role": "user", "content": prompt}],
            )
            return parse_json_to_dict(_extract_output_text(response))
        except Exception as exc:  # Retry API failures and malformed judge output alike.
            last_error = exc
            if attempt < config.max_attempts and config.retry_delay_seconds:
                time.sleep(config.retry_delay_seconds)

    raise RuntimeError(
        f"HealthBench judge failed after {config.max_attempts} attempts for one rubric item"
    ) from last_error


def grade_healthbench_response(
    client: ResponsesClient,
    source: Mapping[str, Any],
    response_text: str,
    *,
    config: JudgeConfig = JudgeConfig(),
) -> dict[str, Any]:
    """Judge every rubric independently and return un-clipped sample metrics."""

    messages = source.get("messages")
    rubric_values = source.get("rubric_items")
    if not isinstance(messages, list) or not isinstance(rubric_values, list):
        raise ValueError("HealthBench source requires messages and rubric_items lists")

    rubric_items = coerce_rubric_items(rubric_values)
    if not rubric_items:
        raise ValueError("HealthBench source requires at least one rubric item")

    def grade_one(item: RubricItem) -> dict[str, Any]:
        return judge_rubric_item(client, messages, response_text, item, config=config)

    # executor.map preserves input order, so saved grades remain aligned with
    # rubric_items even though independent judge calls run concurrently.
    with ThreadPoolExecutor(
        max_workers=min(config.max_workers, len(rubric_items))
    ) as executor:
        rubric_grades = list(executor.map(grade_one, rubric_items))
    slices = {
        field: source.get(field)
        for field in ("use_case", "type", "difficulty", "specialty")
    }
    metrics = score_response(rubric_items, rubric_grades, response_text, slices=slices)
    return {
        "dataset_id": source.get("dataset_id"),
        "dataset_repo": source.get("dataset_repo"),
        "dataset_revision": source.get("dataset_revision"),
        "dataset_index": source.get("dataset_index"),
        "slices": slices,
        "judge": {
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "max_attempts": config.max_attempts,
            "max_workers": config.max_workers,
        },
        "metrics": metrics,
        "response_characters": len(response_text),
        "rubric_grades": [
            {**item.to_dict(), **grade}
            for item, grade in zip(rubric_items, rubric_grades, strict=True)
        ],
    }


def grade_answer_files(
    *,
    source_path: Path,
    answer_path: Path,
    client: ResponsesClient,
    trajectory_path: Path | None = None,
    config: JudgeConfig = JudgeConfig(),
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("HealthBench source.json must contain an object")
    response_text = answer_path.read_text(encoding="utf-8")
    if trajectory_path is not None:
        exact = load_atif_trajectory(trajectory_path)
        if exact.visible_text != response_text:
            raise ValueError(
                "HealthBench answer does not exactly match the validated trajectory visible_text"
            )
    return grade_healthbench_response(client, source, response_text, config=config)


def harbor_reward(result: Mapping[str, Any]) -> dict[str, float]:
    """Return Harbor's single scalar reward; detailed metrics stay sidecar-only."""

    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("HealthBench result is missing metrics")
    score = metrics.get("overall_score_length_adjusted")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(
            "HealthBench result requires numeric overall_score_length_adjusted"
        )
    return {"reward": float(score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tests/source.json"))
    parser.add_argument("--answer", type=Path, default=Path("/workspace/answer.txt"))
    parser.add_argument(
        "--trajectory", type=Path, default=Path("/logs/agent/trajectory.json")
    )
    parser.add_argument(
        "--reward", type=Path, default=Path("/logs/verifier/reward.json")
    )
    parser.add_argument(
        "--details", type=Path, default=Path("/logs/verifier/healthbench_result.json")
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - verifier image integration
        raise RuntimeError("openai is required to run the HealthBench judge") from exc

    result = grade_answer_files(
        source_path=args.source,
        answer_path=args.answer,
        trajectory_path=args.trajectory,
        client=OpenAI(),
    )
    # The detail file is the source for summaries and RL export. Write it first,
    # then write Harbor's scalar reward as the completion marker.
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(
        json.dumps(harbor_reward(result), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
