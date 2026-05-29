"""Shared OPD/OPSD token alignment and teacher scoring helpers.

Covers both the sampled-token path (one teacher logprob per response token,
used by ``importance_sampling``) and the top-K path (top-K ids + logprobs per
response position, used by the OPSD top-K modes in ``opd.py``). Both paths share
the same echo request and the same ``content[1:]`` alignment so there is a
single source of truth for how the response is walked.

source-K (selection) vs gather (fixed indices)
----------------------------------------------
Two different operations, do not confuse them:

* SELECTION ("source K"): pick the top-K token *indices* (+ logprobs) of a
  model. Either ``_extract_teacher_topk`` here (inference ``top_logprobs``) or
  the trainer ``forward(loss_fn_config={"top_k": K})`` (exact, ``@no_grad``).
  Produces the index set used as ``target_tokens``.
* GATHER ("at fixed indices"): read a model's logprobs AT given indices. This is
  a plain forward at ``target_tokens=[N,K]`` (PR #27269), NOT a top-K selection.

Rule: only ONE selection per position (the source K). The OTHER model and the
training ``forward_backward`` must GATHER at those same indices -- never run a
second ``topK`` selection (it would choose different tokens and misalign the
per-token KL terms).

Which model is the source K depends on KL direction:
* reverse KL (OPSD, ``TOPK_REVERSE_KL``): expectation over ``pi_S`` -> source =
  STUDENT top-K. For correctness extract it on the TRAINER, forward-only, AFTER
  inference (the inference ``top_logprobs`` carry the train/inference gap: quant
  weights, different kernels, truncated/renormalized nucleus, cap <=5). Then
  GATHER the teacher at those student indices.
* forward KL (SDFT, ``TOPK_FORWARD_KL``): expectation over ``pi_T`` -> source =
  TEACHER top-K; ``_extract_teacher_topk`` (inference) is the natural source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.data import prepare_sampling_messages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopKDist:
    """Teacher top-K distribution at one response position.

    ``token_ids`` and ``logprobs`` are parallel, length ``<= K`` (the API may
    return fewer than K candidates). ``logprobs`` are raw teacher logprobs;
    renormalization happens in the datum builders (``opd.py``) so the same
    extraction feeds both forward- and reverse-KL modes.
    """

    token_ids: list[int]
    logprobs: list[float]


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
    """Return the ``content`` entries that predict ``tokens[1:]``.

    Echo responses include an unconditional first-token slot, then one slot per
    next-token target.  A generated extra token may follow; it is trimmed.
    Shared by both the sampled-token and top-K extraction paths.
    """
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
    """Resolve a top_logprobs / token entry to a single token id.

    Prefers an explicit integer ``token_id``/``id`` (raw_output-style). Falls
    back to ``tokenizer`` re-encoding of the token string, keeping only
    unambiguous single-id results. See ``_extract_teacher_topk`` for the
    lossiness this implies.
    """
    for key in ("token_id", "id"):
        val = entry.get(key)
        if isinstance(val, int):
            return val
    if tokenizer is None:
        return None
    token_str = entry.get("token")
    if not isinstance(token_str, str):
        return None
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def _extract_teacher_topk(
    response: dict[str, Any],
    *,
    prompt_len: int,
    response_len: int,
    target_len: int,
    tokenizer: Any | None = None,
) -> list[TopKDist] | None:
    """Extract per-response-position teacher top-K from an echo response.

    Requires the teacher to have been scored with ``top_logprobs=K`` so that
    each ``content`` slot carries a ``top_logprobs`` candidate list.

    Token-id resolution is the sharp edge: the completions API returns
    OpenAI-style ``top_logprobs`` (token *strings*, no ids). ``_candidate_token_id``
    prefers an explicit id field and otherwise re-encodes the string, dropping
    ambiguous multi-id tokens (which silently shrinks K). A raw_output-style
    top-K-ids API would remove this lossiness; until then the top-K modes are
    only sound for tokenizers whose top_logprobs strings round-trip 1:1.
    """
    target_content = _echo_target_content(response, target_len=target_len)
    if target_content is None:
        return None
    response_start = max(0, prompt_len - 1)
    window = target_content[response_start : response_start + response_len]
    if len(window) < response_len:
        return None

    out: list[TopKDist] = []
    for slot in window:
        candidates = slot.get("top_logprobs") if isinstance(slot, dict) else None
        ids: list[int] = []
        lps: list[float] = []
        for cand in candidates or []:
            tok_id = _candidate_token_id(cand, tokenizer)
            if tok_id is None:
                continue
            ids.append(tok_id)
            lps.append(float(cand.get("logprob", 0.0)))
        if not ids:
            # No usable candidates: fall back to the sampled token alone.
            sampled_id = _candidate_token_id(slot, tokenizer)
            if sampled_id is None:
                return None
            ids = [sampled_id]
            lps = [float(slot.get("logprob", 0.0))]
        out.append(TopKDist(token_ids=ids, logprobs=lps))
    return out


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


async def _request_teacher_echo(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    top_logprobs: int,
    http_timeout: int,
) -> dict[str, Any] | None:
    """Run the shared echo+logprobs teacher request used by both score paths."""
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
    """Score sampled response tokens with the teacher deployment (sampled-token path)."""
    target_len = max(0, len(token_ids) - 1)
    if target_len == 0 or response_len <= 0:
        return None

    response = await _request_teacher_echo(
        sampler, token_ids, top_logprobs=top_logprobs, http_timeout=http_timeout
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
    """Score sampled response tokens, returning teacher top-K per position.

    Shares ``_request_teacher_echo`` with :func:`_score_with_teacher`; only the
    extraction differs (top-K candidates vs the single sampled logprob).
    """
    target_len = max(0, len(token_ids) - 1)
    if target_len == 0 or response_len <= 0:
        return None

    response = await _request_teacher_echo(
        sampler, token_ids, top_logprobs=top_logprobs, http_timeout=http_timeout
    )
    if response is None:
        return None

    return _extract_teacher_topk(
        response,
        prompt_len=prompt_len,
        response_len=response_len,
        target_len=target_len,
        tokenizer=tokenizer,
    )
