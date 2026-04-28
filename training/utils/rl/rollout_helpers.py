"""Helpers for token-native multi-turn rollout assembly.

Two recurring tasks in any custom rollout function:

* Pulling the engine's output token IDs and per-token logprobs out of
  the inference response in the same shape :class:`InferenceCall`
  expects.
* Computing the chat-template scaffolding that goes between an
  assistant turn and the next sampling call, *without* re-tokenizing the
  assistant content (which can drift on BPE merges across the boundary).

These helpers don't try to abstract the rollout loop -- the user still
writes their own ``async def rollout(...)``.  They just remove the
two error-prone pieces.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

from training.utils.rl.trajectory_assembler import InferenceCall


__all__ = [
    "extract_completion",
    "precompute_chat_suffix",
]


def extract_completion(
    choice: Mapping[str, Any],
    *,
    input_tokens: Sequence[int],
) -> InferenceCall:
    """Build an :class:`InferenceCall` from a completions-API choice dict.

    Expects Fireworks/OpenAI-shaped fields:

    * ``token_ids``: list of int, the engine's output token IDs.
    * ``logprobs.token_logprobs``: list of float aligned 1:1 with
      ``token_ids``.  ``None`` entries are coerced to ``0.0`` (some
      providers omit the logprob for the first token).
    * ``finish_reason``: str (default ``"stop"``).

    Raises:
        ValueError: if ``token_ids`` is missing/empty or ``logprobs``
            doesn't align with ``token_ids``.  Both are required for
            token-native training.
    """
    raw_token_ids = choice.get("token_ids")
    if not raw_token_ids:
        raise ValueError(
            "completion choice is missing 'token_ids'; this is required for "
            "token-native rollout assembly.  Enable token-id passthrough on "
            "the inference call (e.g. set ``logprobs=True`` and request "
            "``token_ids`` from the Fireworks Completions API).",
        )
    output_tokens = [int(t) for t in raw_token_ids if t is not None]

    raw_logprobs = choice.get("logprobs")
    if isinstance(raw_logprobs, Mapping):
        token_logprobs: Iterable[Any] = raw_logprobs.get("token_logprobs") or []
    else:
        token_logprobs = []
    output_logprobs = [float(lp) if lp is not None else 0.0 for lp in token_logprobs]

    # Defensive align: some providers truncate or pad the logprob list by one.
    if len(output_logprobs) > len(output_tokens):
        output_logprobs = output_logprobs[: len(output_tokens)]
    if len(output_logprobs) != len(output_tokens):
        raise ValueError(
            f"completion has {len(output_tokens)} token_ids but "
            f"{len(output_logprobs)} logprobs; the inference call must "
            f"return per-token logprobs aligned with token_ids "
            f"(slime/AReaL convention).",
        )

    finish_reason = choice.get("finish_reason") or "stop"

    return InferenceCall(
        input_tokens=list(input_tokens),
        output_tokens=output_tokens,
        output_logprobs=output_logprobs,
        finish_reason=str(finish_reason),
    )


def precompute_chat_suffix(
    tokenizer: Any,
    *,
    follow_up_content: str,
    follow_up_role: str = "user",
    add_generation_prompt: bool = True,
) -> List[int]:
    """Tokenize the chat-template scaffolding that follows an assistant turn.

    Returns the token IDs for ``[end-of-assistant-turn + follow_up_role
    wrapper + follow_up_content + generation prompt]``, computed by
    diffing two ``apply_chat_template`` outputs.  Append this to the
    engine's assistant output tokens to build the next prompt without
    re-tokenizing the assistant content.

    AReaL reference: ``MultiTurnWorkflow.__init__``
    (``areal/workflow/multi_turn.py:41-57``).

    Raises:
        RuntimeError: if the tokenizer's chat template doesn't extend
            cleanly when a follow-up message is appended -- usually
            because the renderer mutates earlier turns (e.g. strips
            ``<think>`` blocks).  In that case there is no stable
            suffix and you need a renderer-aware assembly path.
    """
    base = [{"role": "assistant", "content": "x"}]
    s1 = list(tokenizer.apply_chat_template(base, tokenize=True))

    extended = base + [{"role": follow_up_role, "content": follow_up_content}]
    s2 = list(
        tokenizer.apply_chat_template(
            extended,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        ),
    )

    if len(s2) < len(s1) or s2[: len(s1)] != s1:
        raise RuntimeError(
            "tokenizer's chat template does not extend cleanly when adding a "
            "follow-up message: the prefix re-tokenizes differently after the "
            "second turn is appended.  This usually means the renderer mutates "
            "earlier turns (e.g. Qwen3 strips <think> blocks, KimiK2 injects "
            "default system prompts).  No stable suffix exists for this model "
            "-- use renderer-aware assembly that re-renders each turn through "
            "the renderer's own helpers.",
        )
    return s2[len(s1) :]
