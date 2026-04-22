"""Client-side tokenization helpers used by RL rollout plumbing.

The training loop needs two things no external rollout source reliably
provides:

1. **Chat-template tokenization** — turning a message list + assistant reply
   into the exact prompt/completion token IDs the trainer expects.
2. **Inference-logprob recovery** — for importance-sampling-family losses
   (GRPO/DAPO/CISPO/GSPO) we need per-token logprobs aligned with the
   tokenized completion.  Samplers that produce text-only responses
   (remote agents, EP rollouts, judge pipelines) do not return these, so
   we recover them via an ``echo=True`` prefill call.

Both helpers are dependency-light and framework-neutral: they expect an
HF-compatible tokenizer and an HTTP endpoint speaking the OpenAI
``/v1/completions`` schema.
"""

from __future__ import annotations

from typing import Any

import requests


def tokenize_chat_turn(
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[int]]:
    """Return ``(prompt_ids, completion_ids)`` for one assistant turn.

    ``prompt_ids`` are what the sampler would have seen (rendered via the
    tokenizer's chat template with ``add_generation_prompt=True``).
    ``completion_ids`` are the delta introduced by appending the assistant
    turn — computed as ``apply_chat_template([*messages, assistant_message])``
    minus the prompt prefix, so it includes any template-added assistant
    role markers / end-of-turn tokens.

    Args:
        messages: Conversation before this turn (system/user/tool messages).
        assistant_message: The assistant turn to encode.
        tokenizer: HuggingFace-compatible tokenizer with
            ``apply_chat_template``.
        add_generation_prompt: Passed through when rendering the prompt.
    """
    prompt_ids = list(
        tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
    )
    full_ids = list(
        tokenizer.apply_chat_template(
            [*messages, assistant_message],
            add_generation_prompt=False,
            tokenize=True,
        )
    )
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Chat-template rendering is not a strict prefix extension. "
            "Tokenizer likely rewrote earlier tokens when the assistant "
            "turn was appended; cannot derive completion_ids by diff."
        )
    completion_ids = full_ids[len(prompt_ids) :]
    return prompt_ids, completion_ids


def _normalize_inference_base_url(url: str) -> str:
    """Strip trailing ``/inference`` or ``/inference/v1`` from ``url``."""
    normalized = url.rstrip("/")
    for suffix in ("/inference/v1", "/inference"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def get_prefill_logprobs(
    *,
    url: str,
    tokens: list[int],
    api_key: str,
    model: str,
    timeout: float = 180.0,
) -> list[float]:
    """Recover per-token inference logprobs for a known token sequence.

    Issues an ``echo=True`` + ``max_tokens=1`` completion so the server
    scores ``tokens`` under the current (possibly hotloaded) policy without
    generating anything new.  Returns ``len(tokens) - 1`` logprobs aligned
    so that index ``i`` is the logprob of ``tokens[i + 1]`` conditioned on
    ``tokens[:i + 1]``.

    Used when the rollout source returned text only (no token-aligned
    logprobs) and the configured loss needs ``inf_logprobs`` for
    importance-sampling correction.
    """
    if not tokens:
        return []
    if len(tokens) < 2:
        return [0.0] * (len(tokens) - 1)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Api-Key": api_key,
    }
    payload: dict[str, Any] = {
        "prompt": tokens,
        "max_tokens": 1,
        "echo": True,
        "logprobs": True,
        "prompt_cache_max_len": 0,
    }
    if model:
        payload["model"] = model

    base_url = _normalize_inference_base_url(url)
    response = requests.post(
        f"{base_url}/inference/v1/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()

    choices = result.get("choices") or []
    if not choices:
        return [0.0] * (len(tokens) - 1)

    logprobs_data = choices[0].get("logprobs") or {}
    token_logprobs = logprobs_data.get("token_logprobs") or []

    # Echo returns one logprob per token; the first is always ``None`` (no
    # conditioning).  Drop it and pad to the expected length so the caller
    # can assume ``len(out) == len(tokens) - 1``.
    aligned = [float(lp) if lp is not None else 0.0 for lp in token_logprobs[1 : len(tokens)]]
    expected = len(tokens) - 1
    if len(aligned) < expected:
        aligned.extend([0.0] * (expected - len(aligned)))
    return aligned
