"""Shared OPD token alignment and teacher scoring helpers."""

from __future__ import annotations

import logging
from typing import Any

from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.data import prepare_sampling_messages

logger = logging.getLogger(__name__)


def _align_completion_logprobs(
    completion_logprobs: list[float],
    *,
    prompt_len: int,
    target_len: int,
    echoed: bool,
) -> list[float] | None:
    """Align API completion logprobs to ``target_tokens`` length."""
    if echoed:
        response_start = max(0, prompt_len - 1)
        if len(completion_logprobs) < target_len and target_len > response_start:
            return None
        aligned = list(completion_logprobs)
        if len(aligned) < target_len:
            aligned.extend([0.0] * (target_len - len(aligned)))
        return aligned[:target_len]

    return _align_response_logprobs(
        completion_logprobs,
        prompt_len=prompt_len,
        target_len=target_len,
    )


def _align_response_logprobs(
    response_logprobs: list[float],
    *,
    prompt_len: int,
    target_len: int,
) -> list[float] | None:
    """Align response-only logprobs to ``target_tokens`` length."""
    response_start = max(0, prompt_len - 1)
    response_len = max(0, target_len - response_start)
    if len(response_logprobs) < response_len:
        return None
    aligned = [0.0] * response_start + list(response_logprobs[:response_len])
    if len(aligned) < target_len:
        aligned.extend([0.0] * (target_len - len(aligned)))
    return aligned[:target_len]


def _extract_scored_token_logprobs(
    response: dict[str, Any],
    *,
    target_len: int,
) -> list[float] | None:
    """Extract echo logprobs for ``tokens[1:]`` from a completions response."""
    choices = response.get("choices", [])
    if not choices:
        return None

    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    content = logprobs.get("content")
    if not isinstance(content, list) or len(content) < 2:
        return None

    # Echo responses include an unconditional first-token logprob, then one
    # logprob per next-token target.  A generated extra token may follow; trim it.
    target_content = content[1 : 1 + target_len]
    if len(target_content) < target_len:
        return None
    return [float(item.get("logprob", 0.0)) for item in target_content]


def _slice_response_logprobs(
    token_logprobs: list[float],
    *,
    prompt_len: int,
    response_len: int,
) -> list[float] | None:
    """Return only response-token logprobs from a full echoed sequence."""
    if response_len <= 0:
        return None
    response_start = max(0, prompt_len - 1)
    response_logprobs = token_logprobs[response_start : response_start + response_len]
    if len(response_logprobs) < response_len:
        return None
    return response_logprobs


def _teacher_messages_for_row(
    row: dict[str, Any],
    fallback_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick privileged teacher messages when the dataset provides them."""
    for key in ("teacher_messages", "privileged_messages", "teacher_prompt_messages"):
        messages = row.get(key)
        if messages:
            return prepare_sampling_messages(messages)
    return fallback_messages


def _tokenize_teacher_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> list[int]:
    """Tokenize the teacher-side prompt with the same chat-template path as sampling."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def _build_teacher_scoring_tokens(
    teacher_prompt_tokens: list[int],
    student_full_tokens: list[int],
    *,
    student_prompt_len: int,
) -> tuple[list[int], int] | None:
    """Combine the privileged teacher prompt with the sampled student response."""
    completion_tokens = list(student_full_tokens[student_prompt_len:])
    if not completion_tokens:
        return None
    return list(teacher_prompt_tokens) + completion_tokens, len(completion_tokens)


async def _score_with_teacher(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    prompt_len: int,
    response_len: int,
    top_logprobs: int,
    http_timeout: int,
) -> list[float] | None:
    """Score sampled response tokens with the teacher deployment."""
    target_len = max(0, len(token_ids) - 1)
    if target_len == 0 or response_len <= 0:
        return None

    request_kwargs: dict[str, Any] = {
        "logprobs": True,
        "echo": True,
        "raw_output": True,
        "http_timeout": http_timeout,
    }
    if top_logprobs > 0:
        request_kwargs["top_logprobs"] = top_logprobs

    try:
        response, _metrics = await sampler.async_completions_stream(
            prompt=token_ids,
            max_tokens=1,
            temperature=0.0,
            **request_kwargs,
        )
    except Exception as exc:
        logger.warning("Teacher scoring failed: %s", exc)
        return None

    token_logprobs = _extract_scored_token_logprobs(response, target_len=target_len)
    if token_logprobs is None:
        return None
    return _slice_response_logprobs(
        token_logprobs,
        prompt_len=prompt_len,
        response_len=response_len,
    )
