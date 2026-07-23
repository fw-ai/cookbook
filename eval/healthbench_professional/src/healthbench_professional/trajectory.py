"""Fail-closed token trajectory handling for HealthBench Professional.

The model response text is deliberately not tokenized here.  Prompt token IDs,
completion token IDs, and behavior-policy log probabilities must all come from
the same Fireworks response.  This module has no Harbor import so the adapter
can also copy it into generated verifier tasks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATIF_SCHEMA_VERSION = "ATIF-v1.7"
TRAJECTORY_FILENAME = "trajectory.json"

_MISSING = object()
_FORBIDDEN_SAMPLING_SETTINGS = {"echo", "reasoning_effort", "top_k"}


class TrajectoryContractError(ValueError):
    """Raised when an inference response is not safe to use for RL."""


@dataclass(frozen=True)
class ExactTokenTrajectory:
    """One exact token-in/token-out inference call."""

    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    finish_reason: str
    visible_text: str
    raw_completion: str | None
    reasoning_content: str | None
    model: str
    settings: dict[str, Any]
    provider_model: str | None = None

    def rollout_detail(self) -> dict[str, Any]:
        """Return Harbor's per-chat rollout-details representation."""

        extra: dict[str, list[Any]] = {
            "finish_reason": [self.finish_reason],
            "model": [self.model],
            "settings": [dict(self.settings)],
            "visible_text": [self.visible_text],
        }
        if self.raw_completion is not None:
            extra["raw_completion"] = [self.raw_completion]
        if self.provider_model is not None:
            extra["provider_model"] = [self.provider_model]
        return {
            "prompt_token_ids": [list(self.prompt_token_ids)],
            "completion_token_ids": [list(self.completion_token_ids)],
            "logprobs": [list(self.sampling_logprobs)],
            "extra": extra,
        }


def validate_sampling_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the sampling settings recorded with a trajectory."""

    copied = dict(settings)
    forbidden = sorted(_FORBIDDEN_SAMPLING_SETTINGS.intersection(copied))
    if forbidden:
        raise TrajectoryContractError(
            "sampling settings must omit " + ", ".join(forbidden)
        )

    required_exact = {
        "raw_output": True,
        "return_token_ids": True,
        "logprobs": True,
        "context_length_exceeded_behavior": "error",
        "n": 1,
        "stream": False,
    }
    for key, expected in required_exact.items():
        if key not in copied or copied[key] != expected:
            raise TrajectoryContractError(
                f"sampling setting {key!r} must be {expected!r}"
            )

    temperature = _finite_number(copied.get("temperature"), "temperature")
    if temperature < 0:
        raise TrajectoryContractError("temperature must be >= 0")
    top_p = _finite_number(copied.get("top_p"), "top_p")
    if not 0 < top_p <= 1:
        raise TrajectoryContractError("top_p must be in (0, 1]")
    max_tokens = copied.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise TrajectoryContractError("max_tokens must be a positive integer")
    return copied


def extract_fireworks_response(
    response: Any,
    *,
    model: str,
    settings: Mapping[str, Any],
) -> ExactTokenTrajectory:
    """Extract exact token arrays from a non-streaming Fireworks chat response.

    This intentionally has no text-tokenization fallback and never substitutes
    ``logprob`` for ``sampling_logprob``.
    """

    recorded_settings = validate_sampling_settings(settings)
    if not isinstance(model, str) or not model:
        raise TrajectoryContractError("model must be a non-empty string")

    prompt_token_ids = _token_ids(
        _get(response, "prompt_token_ids", None),
        "response.prompt_token_ids",
    )
    choices = _list_value(_get(response, "choices", None), "response.choices")
    if len(choices) != 1:
        raise TrajectoryContractError(
            f"expected exactly one completion choice, got {len(choices)}"
        )
    choice = choices[0]
    raw_output = _get(choice, "raw_output", None)
    if raw_output is None:
        raise TrajectoryContractError("choice.raw_output is required")

    completion_token_ids = _token_ids(
        _get(raw_output, "completion_token_ids", None),
        "choice.raw_output.completion_token_ids",
    )
    raw_prompt_ids = _get(raw_output, "prompt_token_ids", None)
    if raw_prompt_ids is not None:
        parsed_raw_prompt_ids = _token_ids(
            raw_prompt_ids,
            "choice.raw_output.prompt_token_ids",
        )
        if parsed_raw_prompt_ids != prompt_token_ids:
            raise TrajectoryContractError(
                "response.prompt_token_ids do not match raw_output.prompt_token_ids"
            )

    raw_completion_logprobs = _get(raw_output, "completion_logprobs", None)
    if raw_completion_logprobs is None:
        raise TrajectoryContractError(
            "raw_output.completion_logprobs is required for token-native RL"
        )
    sampling_logprobs = _sampling_logprobs(
        _get(raw_completion_logprobs, "content", None),
        completion_token_ids,
        "choice.raw_output.completion_logprobs",
    )

    # choice.logprobs can cover only parsed visible content when reasoning is
    # separated. Cross-check it only when it explicitly covers the same raw
    # token sequence; it is never authoritative for RL export.
    choice_logprobs = _get(choice, "logprobs", None)
    if choice_logprobs is not None:
        choice_content = _get(choice_logprobs, "content", None)
        if _content_covers_token_ids(choice_content, completion_token_ids):
            candidate = _sampling_logprobs(
                choice_content,
                completion_token_ids,
                "choice.logprobs",
            )
            source = "choice.logprobs"
        else:
            candidate = None
            source = ""
    else:
        candidate = None
        source = ""
    if candidate is not None:
        if candidate != sampling_logprobs:
            raise TrajectoryContractError(
                f"{source} disagrees with raw_output.completion_logprobs"
            )

    message = _get(choice, "message", None)
    if message is None:
        raise TrajectoryContractError("choice.message is required")
    content = _get(message, "content", None)
    if content is None:
        visible_text = ""
    elif isinstance(content, str):
        visible_text = content
    else:
        raise TrajectoryContractError("choice.message.content must be text")

    reasoning_content = _get(message, "reasoning_content", None)
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TrajectoryContractError("reasoning_content must be text when present")
    raw_completion = _get(raw_output, "completion", None)
    if raw_completion is not None and not isinstance(raw_completion, str):
        raise TrajectoryContractError("raw_output.completion must be text when present")

    finish_reason = _get(choice, "finish_reason", None)
    if not isinstance(finish_reason, str) or not finish_reason:
        raise TrajectoryContractError("finish_reason must be non-empty text")
    provider_model = _get(response, "model", None)
    if provider_model is not None and not isinstance(provider_model, str):
        raise TrajectoryContractError("response.model must be text when present")

    return ExactTokenTrajectory(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        sampling_logprobs=sampling_logprobs,
        finish_reason=finish_reason,
        visible_text=visible_text,
        raw_completion=raw_completion,
        reasoning_content=reasoning_content,
        model=model,
        settings=recorded_settings,
        provider_model=provider_model,
    )


def build_atif_trajectory(
    exact: ExactTokenTrajectory,
    *,
    instruction: str,
    messages: list[dict[str, Any]],
    agent_name: str,
    agent_version: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal ATIF-v1.7 document for one HealthBench response."""

    metrics_extra: dict[str, Any] = {
        "finish_reason": exact.finish_reason,
        "settings": dict(exact.settings),
    }
    if exact.raw_completion is not None:
        metrics_extra["raw_completion"] = exact.raw_completion
    if exact.provider_model is not None:
        metrics_extra["provider_model"] = exact.provider_model

    context_steps: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message["role"]
        source = {"system": "system", "user": "user", "assistant": "agent"}.get(
            role,
            "user",
        )
        context_step: dict[str, Any] = {
            "step_id": index + 1,
            "source": source,
            "message": message["content"],
        }
        if role == "assistant":
            context_step["is_copied_context"] = True
        if role not in {"system", "user", "assistant"}:
            context_step["extra"] = {"original_role": role}
        context_steps.append(context_step)

    agent_step: dict[str, Any] = {
        "step_id": len(context_steps) + 1,
        "source": "agent",
        "model_name": exact.model,
        "message": exact.visible_text,
        "llm_call_count": 1,
        "metrics": {
            "prompt_tokens": len(exact.prompt_token_ids),
            "completion_tokens": len(exact.completion_token_ids),
            "prompt_token_ids": list(exact.prompt_token_ids),
            "completion_token_ids": list(exact.completion_token_ids),
            "logprobs": list(exact.sampling_logprobs),
            "extra": metrics_extra,
        },
    }
    if exact.reasoning_content is not None:
        agent_step["reasoning_content"] = exact.reasoning_content

    document: dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": exact.model,
            "extra": {"settings": dict(exact.settings)},
        },
        "steps": context_steps + [agent_step],
        "final_metrics": {
            "total_prompt_tokens": len(exact.prompt_token_ids),
            "total_completion_tokens": len(exact.completion_token_ids),
            "total_steps": len(context_steps) + 1,
        },
        "extra": {"instruction": instruction, "messages": messages},
    }
    if session_id:
        document["session_id"] = session_id
    validate_atif_trajectory(document)
    return document


def write_atif_trajectory(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Validate and persist an ATIF document."""

    validate_atif_trajectory(document)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_atif_trajectory(path: Path | str) -> ExactTokenTrajectory:
    """Load a persisted ATIF document and validate its RL fields."""

    trajectory_path = Path(path)
    try:
        document = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryContractError(
            f"could not load ATIF trajectory {trajectory_path}: {exc}"
        ) from exc
    return validate_atif_trajectory(document)


def validate_atif_trajectory(document: Mapping[str, Any]) -> ExactTokenTrajectory:
    """Validate an ATIF-v1.7 document and extract its exact rollout."""

    if not isinstance(document, Mapping):
        raise TrajectoryContractError("ATIF trajectory must be a JSON object")
    if document.get("schema_version") != ATIF_SCHEMA_VERSION:
        raise TrajectoryContractError(f"schema_version must be {ATIF_SCHEMA_VERSION!r}")

    agent = document.get("agent")
    if not isinstance(agent, Mapping):
        raise TrajectoryContractError("ATIF agent must be an object")
    model = agent.get("model_name")
    if not isinstance(model, str) or not model:
        raise TrajectoryContractError("ATIF agent.model_name must be non-empty")

    root_extra = document.get("extra")
    if not isinstance(root_extra, Mapping):
        raise TrajectoryContractError("ATIF extra.messages is required")
    root_messages = _validate_messages(root_extra.get("messages"))

    steps = _list_value(document.get("steps"), "ATIF steps")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or step.get("step_id") != index:
            raise TrajectoryContractError("ATIF step IDs must be sequential from 1")
    current_agent_steps = [
        step
        for step in steps
        if step.get("source") == "agent"
        and step.get("llm_call_count") == 1
        and isinstance(step.get("metrics"), Mapping)
    ]
    if len(current_agent_steps) != 1:
        raise TrajectoryContractError(
            "expected exactly one ATIF agent step with llm_call_count=1 and metrics"
        )
    step = current_agent_steps[0]
    copied_agent_steps = [
        candidate
        for candidate in steps
        if candidate.get("source") == "agent" and candidate is not step
    ]
    for copied_step in copied_agent_steps:
        if copied_step.get("is_copied_context") is not True:
            raise TrajectoryContractError(
                "non-current ATIF agent steps must be copied context"
            )
        if (
            copied_step.get("metrics") is not None
            or copied_step.get("llm_call_count") is not None
        ):
            raise TrajectoryContractError(
                "copied assistant context must not contain metrics or llm_call_count"
            )

    context_steps = [candidate for candidate in steps if candidate is not step]
    if len(context_steps) != len(root_messages):
        raise TrajectoryContractError(
            "ATIF context steps must match the original dataset messages"
        )
    for index, (context_step, message) in enumerate(
        zip(context_steps, root_messages, strict=True)
    ):
        role = message["role"]
        expected_source = {
            "system": "system",
            "user": "user",
            "assistant": "agent",
        }.get(role, "user")
        if context_step.get("source") != expected_source:
            raise TrajectoryContractError(
                f"ATIF context step {index} does not preserve message role"
            )
        if context_step.get("message") != message["content"]:
            raise TrajectoryContractError(
                f"ATIF context step {index} does not preserve message content"
            )
        if role == "assistant" and context_step.get("is_copied_context") is not True:
            raise TrajectoryContractError(
                f"ATIF assistant context step {index} must be marked copied"
            )
        if role not in {"system", "user", "assistant"}:
            extra = context_step.get("extra")
            if not isinstance(extra, Mapping) or extra.get("original_role") != role:
                raise TrajectoryContractError(
                    f"ATIF context step {index} must preserve its original role"
                )
    if step.get("model_name") != model:
        raise TrajectoryContractError("ATIF step and agent model names must match")
    visible_text = step.get("message")
    if not isinstance(visible_text, str):
        raise TrajectoryContractError("ATIF agent message must be text")
    reasoning_content = step.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TrajectoryContractError("ATIF reasoning_content must be text")

    metrics = step.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TrajectoryContractError("ATIF agent metrics are required")
    prompt_token_ids = _token_ids(
        metrics.get("prompt_token_ids"), "ATIF metrics.prompt_token_ids"
    )
    completion_token_ids = _token_ids(
        metrics.get("completion_token_ids"), "ATIF metrics.completion_token_ids"
    )
    sampling_logprobs = _direct_logprobs(
        metrics.get("logprobs"), len(completion_token_ids), "ATIF metrics.logprobs"
    )
    if metrics.get("prompt_tokens") != len(prompt_token_ids):
        raise TrajectoryContractError("ATIF prompt token count is inconsistent")
    if metrics.get("completion_tokens") != len(completion_token_ids):
        raise TrajectoryContractError("ATIF completion token count is inconsistent")

    metrics_extra = metrics.get("extra")
    if not isinstance(metrics_extra, Mapping):
        raise TrajectoryContractError("ATIF metrics.extra is required")
    settings = validate_sampling_settings(
        _mapping_value(metrics_extra.get("settings"), "ATIF metrics.extra.settings")
    )
    finish_reason = metrics_extra.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise TrajectoryContractError("ATIF finish_reason must be non-empty text")
    raw_completion = metrics_extra.get("raw_completion")
    if raw_completion is not None and not isinstance(raw_completion, str):
        raise TrajectoryContractError("ATIF raw_completion must be text")
    provider_model = metrics_extra.get("provider_model")
    if provider_model is not None and not isinstance(provider_model, str):
        raise TrajectoryContractError("ATIF provider_model must be text")

    agent_extra = agent.get("extra")
    if not isinstance(agent_extra, Mapping) or agent_extra.get("settings") != settings:
        raise TrajectoryContractError(
            "ATIF agent and metrics must contain identical sampling settings"
        )

    final_metrics = document.get("final_metrics")
    if not isinstance(final_metrics, Mapping):
        raise TrajectoryContractError("ATIF final_metrics are required")
    if final_metrics.get("total_prompt_tokens") != len(prompt_token_ids):
        raise TrajectoryContractError("ATIF final prompt token count is inconsistent")
    if final_metrics.get("total_completion_tokens") != len(completion_token_ids):
        raise TrajectoryContractError(
            "ATIF final completion token count is inconsistent"
        )
    if final_metrics.get("total_steps") != len(steps):
        raise TrajectoryContractError("ATIF final step count is inconsistent")

    return ExactTokenTrajectory(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        sampling_logprobs=sampling_logprobs,
        finish_reason=finish_reason,
        visible_text=visible_text,
        raw_completion=raw_completion,
        reasoning_content=reasoning_content,
        model=model,
        settings=settings,
        provider_model=provider_model,
    )


def _validate_messages(value: Any) -> list[dict[str, Any]]:
    messages = _list_value(value, "messages")
    if not messages:
        raise TrajectoryContractError("messages must not be empty")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TrajectoryContractError(f"messages[{index}] must be an object")
        if not isinstance(message.get("role"), str):
            raise TrajectoryContractError(f"messages[{index}].role must be text")
        if "content" not in message:
            raise TrajectoryContractError(f"messages[{index}].content is required")
        if not isinstance(message["content"], str):
            raise TrajectoryContractError(f"messages[{index}].content must be text")
    return messages


def _sampling_logprobs(
    value: Any,
    completion_token_ids: tuple[int, ...],
    source: str,
) -> tuple[float, ...]:
    content = _list_value(value, f"{source}.content")
    if len(content) != len(completion_token_ids):
        raise TrajectoryContractError(
            f"{source}.content has {len(content)} entries for "
            f"{len(completion_token_ids)} completion token IDs"
        )
    values: list[float] = []
    for index, (item, expected_token_id) in enumerate(
        zip(content, completion_token_ids, strict=True)
    ):
        token_id = _get(item, "token_id", _MISSING)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TrajectoryContractError(
                f"{source}.content[{index}].token_id is required"
            )
        if token_id != expected_token_id:
            raise TrajectoryContractError(
                f"{source}.content[{index}].token_id={token_id} does not match "
                f"completion_token_ids[{index}]={expected_token_id}"
            )
        sampling_logprob = _get(item, "sampling_logprob", _MISSING)
        values.append(
            _finite_number(
                sampling_logprob,
                f"{source}.content[{index}].sampling_logprob",
            )
        )
    return tuple(values)


def _content_covers_token_ids(
    value: Any,
    completion_token_ids: tuple[int, ...],
) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if len(value) != len(completion_token_ids):
        return False
    for item, expected in zip(value, completion_token_ids, strict=True):
        token_id = _get(item, "token_id", None)
        if isinstance(token_id, bool) or token_id != expected:
            return False
    return True


def _direct_logprobs(value: Any, expected: int, source: str) -> tuple[float, ...]:
    items = _list_value(value, source)
    if len(items) != expected:
        raise TrajectoryContractError(
            f"{source} has {len(items)} entries for {expected} completion tokens"
        )
    return tuple(
        _finite_number(item, f"{source}[{index}]") for index, item in enumerate(items)
    )


def _token_ids(value: Any, source: str) -> tuple[int, ...]:
    items = _list_value(value, source)
    if not items:
        raise TrajectoryContractError(f"{source} must not be empty")
    token_ids: list[int] = []
    for index, token_id in enumerate(items):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise TrajectoryContractError(
                f"{source}[{index}] must be a non-negative integer"
            )
        token_ids.append(token_id)
    return tuple(token_ids)


def _finite_number(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryContractError(f"{source} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise TrajectoryContractError(f"{source} must be a finite number")
    return number


def _list_value(value: Any, source: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise TrajectoryContractError(f"{source} must be a list")


def _mapping_value(value: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectoryContractError(f"{source} must be an object")
    return value


def _get(value: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
    elif value is not None and hasattr(value, key):
        return getattr(value, key)
    if default is _MISSING:
        raise TrajectoryContractError(f"missing required field {key!r}")
    return default


__all__ = [
    "ATIF_SCHEMA_VERSION",
    "ExactTokenTrajectory",
    "TRAJECTORY_FILENAME",
    "TrajectoryContractError",
    "build_atif_trajectory",
    "extract_fireworks_response",
    "load_atif_trajectory",
    "validate_atif_trajectory",
    "validate_sampling_settings",
    "write_atif_trajectory",
]
