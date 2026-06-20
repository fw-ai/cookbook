"""Shared distillation token alignment and teacher scoring helpers.

Covers both the sampled-token path (one teacher logprob per response token,
used by ``importance_sampling``) and the top-K path (top-K ids + logprobs per
response position, used by SDFT top-K helper losses). Both paths share the same
echo request and ``content[1:]`` alignment so there is one source of truth for
how the response is walked.

Top-K selection and fixed-index gather are separate operations. Selection picks
the K token ids for a model; gather reads another model's logprobs at those
fixed ids via ``target_tokens=[N, K]``. Only one model should select per
position. The other model and the training ``forward_backward`` call should
gather at the same selected ids.

For SDFT, teacher top-K comes from the inference deployment response. The
number of sparse entries that cookbook can train on is therefore bounded by the
deployment's ``top_logprobs`` response support.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.data import prepare_sampling_messages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopKDist:
    """Top-K distribution at one response position.

    ``token_ids`` and ``logprobs`` are parallel arrays. ``logprobs`` are raw
    model logprobs; renormalization happens in the datum builders.
    """

    token_ids: list[int]
    logprobs: list[float]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError(
                "TopKDist.token_ids and TopKDist.logprobs must have the same length."
            )
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in self.token_ids
        ):
            raise TypeError("TopKDist.token_ids must be integers.")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("TopKDist.token_ids must be non-negative.")
        if any(not math.isfinite(float(logprob)) for logprob in self.logprobs):
            raise ValueError("TopKDist.logprobs must be finite.")


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


def _echo_target_content(
    response: dict[str, Any],
    *,
    target_len: int,
) -> list[dict[str, Any]] | None:
    """Return echo ``content`` entries that predict ``tokens[1:]``."""
    choices = response.get("choices", [])
    if not choices:
        return None

    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    content = logprobs.get("content")
    if not isinstance(content, list) or len(content) < 2:
        return None

    target_content = content[1 : 1 + target_len]
    if len(target_content) < target_len:
        return None
    if not all(isinstance(item, dict) for item in target_content):
        return None
    return target_content


def _extract_scored_token_logprobs(
    response: dict[str, Any],
    *,
    target_len: int,
) -> list[float] | None:
    """Extract echo logprobs for ``tokens[1:]`` from a completions response."""
    target_content = _echo_target_content(response, target_len=target_len)
    if target_content is None:
        return None
    return [float(item.get("logprob", 0.0)) for item in target_content]


def _candidate_token_id(entry: dict[str, Any], tokenizer: Any | None) -> int | None:
    """Resolve one logprob content or candidate entry to a single token id."""
    for key in ("token_id", "id"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    if tokenizer is None:
        return None
    token = entry.get("token")
    if not isinstance(token, str):
        return None

    token_ids = tokenizer.encode(token, add_special_tokens=False)
    return int(token_ids[0]) if len(token_ids) == 1 else None


def _extract_teacher_topk(
    response: dict[str, Any],
    *,
    prompt_len: int,
    response_len: int,
    target_len: int,
    tokenizer: Any | None = None,
) -> list[TopKDist] | None:
    """Extract per-response-position teacher top-K from an inference echo response.

    Requires the scoring request to include ``top_logprobs``. Candidate token id
    resolution prefers explicit integer ids and falls back to tokenizer
    round-tripping of token strings, keeping only unambiguous single-token
    strings.
    """
    target_content = _echo_target_content(response, target_len=target_len)
    if target_content is None:
        return None

    response_start = max(0, prompt_len - 1)
    response_window = target_content[response_start : response_start + response_len]
    if len(response_window) < response_len:
        return None

    topk_by_pos: list[TopKDist] = []
    for slot in response_window:
        candidates = slot.get("top_logprobs")
        token_ids: list[int] = []
        logprobs: list[float] = []
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                token_id = _candidate_token_id(candidate, tokenizer)
                if token_id is None:
                    continue
                token_ids.append(token_id)
                logprobs.append(float(candidate.get("logprob", 0.0)))

        if not token_ids:
            return None

        topk_by_pos.append(TopKDist(token_ids=token_ids, logprobs=logprobs))
    return topk_by_pos


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
    *,
    teacher_messages_key: str = "teacher_messages",
) -> list[dict[str, Any]]:
    """Pick privileged teacher messages when the dataset provides them."""
    if not teacher_messages_key:
        raise ValueError("teacher_messages_key must be non-empty.")

    messages = row.get(teacher_messages_key)
    if messages:
        return prepare_sampling_messages(messages)

    if teacher_messages_key == "teacher_messages":
        for key in ("privileged_messages", "teacher_prompt_messages"):
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


async def _request_teacher_echo(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    top_logprobs: int,
    http_timeout: int,
) -> dict[str, Any] | None:
    """Run the shared echo+logprobs teacher request."""
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
    return response


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

    response = await _request_teacher_echo(
        sampler,
        token_ids,
        top_logprobs=top_logprobs,
        http_timeout=http_timeout,
    )
    if response is None:
        return None

    token_logprobs = _extract_scored_token_logprobs(response, target_len=target_len)
    if token_logprobs is None:
        return None
    return _slice_response_logprobs(
        token_logprobs,
        prompt_len=prompt_len,
        response_len=response_len,
    )


async def _score_teacher_topk(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    prompt_len: int,
    response_len: int,
    top_logprobs: int,
    http_timeout: int,
    tokenizer: Any | None = None,
) -> list[TopKDist] | None:
    """Return teacher candidates from inference ``top_logprobs``."""
    target_len = max(0, len(token_ids) - 1)
    if target_len == 0 or response_len <= 0:
        return None

    response = await _request_teacher_echo(
        sampler,
        token_ids,
        top_logprobs=top_logprobs,
        http_timeout=http_timeout,
    )
    if response is None:
        return None

    topk_by_pos = _extract_teacher_topk(
        response,
        prompt_len=prompt_len,
        response_len=response_len,
        target_len=target_len,
        tokenizer=tokenizer,
    )
    if topk_by_pos is None:
        raise ValueError(
            "Teacher inference response did not include usable top_logprobs "
            "for every response token."
        )
    for pos, dist in enumerate(topk_by_pos):
        if len(dist.token_ids) < top_logprobs:
            raise ValueError(
                "Teacher inference top_logprobs returned fewer candidates than requested "
                f"at response position {pos}: got {len(dist.token_ids)}, requested {top_logprobs}."
            )
    return topk_by_pos
