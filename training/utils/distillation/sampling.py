"""Teacher scoring and token alignment for sampled-token distillation.

The teacher scores the student's sampled response tokens with one echo request
(``echo=True, logprobs=True, max_tokens=1``) over ``teacher_prompt +
student_response``. Echo ``content[1:]`` predicts ``tokens[1:]``, so the
response slice starts at ``prompt_len - 1``. The same alignment places student
and teacher logprobs on the training datum's ``target_tokens``.

On serverless training the student and teacher are Tinker sampling clients
instead of deployments; the last section adapts their responses to the same
:class:`SampledCompletion` rows and response-only teacher logprobs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import tinker
from fireworks.training.sdk import SampledCompletion
from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.data import prepare_sampling_messages

logger = logging.getLogger(__name__)

TEACHER_MESSAGES_KEYS = ("teacher_messages", "privileged_messages", "teacher_prompt_messages")
"""Row keys that may hold the teacher's privileged prompt, in lookup order."""


def align_completion_logprobs(
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

    return align_response_logprobs(
        completion_logprobs,
        prompt_len=prompt_len,
        target_len=target_len,
    )


def align_response_logprobs(
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

    target_content = content[1 : 1 + target_len]
    if len(target_content) < target_len:
        return None
    if not all(isinstance(item, dict) for item in target_content):
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


def teacher_messages_for_row(
    row: dict[str, Any],
    student_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The teacher's privileged prompt when the row has one, else the student prompt."""
    for key in TEACHER_MESSAGES_KEYS:
        messages = row.get(key)
        if messages:
            return prepare_sampling_messages(messages)
    return student_messages


def tokenize_teacher_prompt(
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


async def score_with_teacher(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    prompt_len: int,
    response_len: int,
    http_timeout: int,
) -> list[float] | None:
    """Teacher logprobs for the last ``response_len`` tokens of ``token_ids``.

    Returns ``None`` when the request fails or the response is incomplete.
    """
    target_len = max(0, len(token_ids) - 1)
    if target_len == 0 or response_len <= 0:
        return None

    try:
        response, _metrics = await sampler.async_completions_stream(
            prompt=token_ids,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            echo=True,
            raw_output=True,
            http_timeout=http_timeout,
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


async def score_teacher_logprobs(
    sampler: DeploymentSampler,
    teacher_prompt_tokens: Sequence[int],
    sampled: Sequence[Any],
    *,
    http_timeout: int,
) -> list[list[float] | None]:
    """Score every sampled response under the teacher prompt, concurrently.

    Each entry holds the teacher's logprobs on that sample's response tokens,
    or ``None`` for an empty completion or a failed scoring request.
    """

    async def score(sample: Any) -> list[float] | None:
        completion = list(sample.full_tokens[sample.prompt_len :])
        if not completion:
            return None
        return await score_with_teacher(
            sampler,
            list(teacher_prompt_tokens) + completion,
            prompt_len=len(teacher_prompt_tokens),
            response_len=len(completion),
            http_timeout=http_timeout,
        )

    return list(await asyncio.gather(*(score(sample) for sample in sampled)))


# -- Tinker sampling clients (serverless) ---------------------------------------


def completions_from_sample_response(
    prompt_tokens: Sequence[int],
    response: Any,
) -> list[SampledCompletion]:
    """One :class:`SampledCompletion` per sequence of a Tinker ``SampleResponse``.

    Tinker sampling clients return completion-only tokens and logprobs, so the
    student prompt is prepended to form ``full_tokens``. ``text`` is left
    empty: distillation trains on tokens only.
    """
    prompt = [int(token) for token in prompt_tokens]
    completions: list[SampledCompletion] = []
    for sequence in response.sequences:
        tokens = [int(token) for token in sequence.tokens]
        logprobs = sequence.logprobs
        completions.append(
            SampledCompletion(
                text="",
                full_tokens=prompt + tokens,
                prompt_len=len(prompt),
                finish_reason=str(sequence.stop_reason),
                completion_len=len(tokens),
                sampling_logprobs=None if logprobs is None else [float(lp) for lp in logprobs],
            )
        )
    return completions


def score_teacher_logprobs_with_sampling_client(
    sampler: Any,
    teacher_prompt_tokens: Sequence[int],
    sampled: Sequence[SampledCompletion],
    *,
    timeout: float,
) -> list[list[float] | None]:
    """Score every sampled response under the teacher prompt with ``compute_logprobs``.

    The Tinker sampling-client counterpart of :func:`score_teacher_logprobs`:
    all requests are in flight at once, and each entry holds the teacher's
    logprobs on that sample's response tokens, or ``None`` for an empty
    completion.
    """
    prompt = [int(token) for token in teacher_prompt_tokens]
    futures = []
    for sample in sampled:
        completion = list(sample.full_tokens[sample.prompt_len :])
        futures.append(
            sampler.compute_logprobs(tinker.ModelInput.from_ints(prompt + completion))
            if completion
            else None
        )
    # compute_logprobs returns one entry per input token (``None`` for the
    # first), so the response logprobs start at the teacher prompt length.
    return [
        None
        if future is None
        else [float(lp) for lp in future.result(timeout=timeout)[len(prompt) :]]
        for future in futures
    ]
