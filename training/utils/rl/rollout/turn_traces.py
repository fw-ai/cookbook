"""Pack multi-turn token traces into one trainable rollout sample.

A multi-turn rollout makes one model call per turn and records it as a trace::

    {"prompt_ids": [...], "completion_ids": [...], "completion_logprobs": [...]}

``prompt_ids`` is every token the model saw, ``completion_ids`` the tokens the
turn trains on (the sampled tokens, after any forced prefill), and
``completion_logprobs`` one behavior logprob per completion token (``0.0`` for
a forced prefill token, which the sampler emitted with probability one).

When each next prompt is built from tokens -- the previous prompt, the
completion verbatim, the turn's close (:func:`append_turn_close`) and the
tool reply (:func:`tool_reply_ids`) -- every prompt extends the previous
prompt and completion, and the whole episode is one sequence: completions are
trained, everything between them is context. eval-protocol rollout processors
store these traces in ``row.execution_metadata.extra["token_turn_traces"]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jinja2 import TemplateError

from training.utils.rl.rollout.assembler import InferenceCall, TrajectoryAssembler
from training.utils.rl.rollout.types import RolloutSample

__all__ = ["append_turn_close", "tool_reply_ids", "turn_spans", "turn_traces_to_sample"]

TokenTurnTrace = Mapping[str, Any]


def append_turn_close(
    completion_ids: Sequence[int], close_ids: Sequence[int]
) -> list[int]:
    """Return ``completion_ids`` followed by the turn close, never doubled.

    A model that stops cleanly emits the end-of-turn tokens itself, so the
    longest tail of ``completion_ids`` matching the head of ``close_ids`` is
    not repeated. Use the result to build the next prompt: the close is context
    there, so a length-truncated turn never trains on a close it did not emit.
    """
    completion = [int(token) for token in completion_ids]
    close = [int(token) for token in close_ids]
    for overlap in range(min(len(completion), len(close)), 0, -1):
        if completion[-overlap:] == close[:overlap]:
            return completion + close[overlap:]
    return completion + close


def tool_reply_ids(
    tokenizer: Any, tool_message: Mapping[str, Any], **chat_template_kwargs: Any
) -> list[int]:
    """Tokens of a tool reply and the next assistant opener, as the template renders them.

    The reply is rendered on its own. Templates that reject a conversation
    without a user query (Qwen3.5/3.6) render it after a placeholder user turn,
    which is cut off again.
    """
    message = {key: value for key, value in tool_message.items() if value is not None}
    try:
        text = tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
    except TemplateError:
        context = [{"role": "user", "content": "."}]
        before = tokenizer.apply_chat_template(context, tokenize=False, **chat_template_kwargs)
        after = tokenizer.apply_chat_template(
            context + [message], tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        if not after.startswith(before):
            raise ValueError(
                "The chat template re-renders the user turn when a tool reply follows it."
            ) from None
        text = after[len(before) :]
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def turn_traces_to_sample(
    traces: Sequence[TokenTurnTrace],
    *,
    reward: float,
    finish_reason: str = "stop",
) -> RolloutSample:
    """Pack one episode's turn traces into a token-native training sample.

    Raises:
        PrefixMismatch: a prompt does not extend the previous prompt and
            completion, i.e. the rollout re-rendered text between turns.
        ValueError: no traces, or a turn lacks one logprob per completion token.
    """
    if not traces:
        raise ValueError("episode has no turn traces")
    assembler = TrajectoryAssembler()
    for trace in traces:
        assembler.add_call(
            InferenceCall(
                input_tokens=[int(token) for token in trace["prompt_ids"]],
                output_tokens=[int(token) for token in trace["completion_ids"]],
                output_logprobs=[
                    float(logprob) for logprob in trace.get("completion_logprobs") or []
                ],
            )
        )
    tokens, logprobs, loss_mask = assembler.to_flat()
    return RolloutSample(
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        reward=float(reward),
        finish_reason=finish_reason,
    )


def turn_spans(traces: Sequence[TokenTurnTrace]) -> list[tuple[int, int]]:
    """``[start, end)`` of each turn's completion in the packed episode."""
    spans = []
    for trace in traces:
        start = len(trace["prompt_ids"])
        spans.append((start, start + len(trace["completion_ids"])))
    return spans
