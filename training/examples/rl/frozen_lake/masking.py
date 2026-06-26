"""Shared multi-turn masking logic for training and visualization.

Single source of truth for which token positions are model-generated
completions vs prompt/environment tokens in a multi-turn rollout episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_model_output_spans(
    token_turn_traces: List[Dict[str, Any]],
    model_request_traces: List[Dict[str, Any]],
) -> List[Tuple[int, int, int]]:
    """Compute (token_start, length, turn_index) for each turn's model output.

    ``token_start`` is the position in the full episode token array where
    the model's completion begins for that turn.  ``turn_index`` is 1-based.

    Both the training loss mask and the UI visualization mask derive from
    this same set of spans so they stay aligned by construction.

    Contract: ``length`` is the *trained* span — the tool-call prefill plus the
    engine's completion tokens verbatim (``assistant_turn_len`` for intermediate
    turns, ``len(completion_ids)`` for the last turn). It deliberately does NOT
    include the synthesized end-of-turn close that bridges one turn to the next.
    The rollout appends that close only into the *next* turn's prompt, so it sits
    just after this span and is left unmasked (loss_mask=0). This implements the
    TITO/renderer rule: never compute loss on a close token the model did not
    actually emit (e.g. a synthesized close after a length-truncated turn). A
    close the model *did* emit on a clean stop is part of ``completion_ids`` and
    stays inside the span with its real logprob.
    """
    spans: List[Tuple[int, int, int]] = []
    num_turns = len(token_turn_traces)
    for k in range(num_turns):
        turn_prompt_len = len(token_turn_traces[k].get("prompt_ids") or [])
        if k < num_turns - 1:
            mrt_k = model_request_traces[k] if k < len(model_request_traces) else {}
            model_output_len = int(mrt_k.get("assistant_turn_len") or 0)
        else:
            last_completion_ids = token_turn_traces[k].get("completion_ids") or []
            model_output_len = len(last_completion_ids)
        if model_output_len == 0:
            model_output_len = len(token_turn_traces[k].get("completion_ids") or [])
        spans.append((turn_prompt_len, model_output_len, k + 1))
    return spans


def build_training_loss_mask(
    spans: List[Tuple[int, int, int]],
    model_input_len: int,
) -> List[float]:
    """Build a per-position loss mask in logprob coordinate (shifted by -1).

    Position ``p`` in the loss mask corresponds to predicting
    ``full_tokens[p+1]``, so a model-output token at token-position ``s``
    maps to loss-mask position ``s - 1``.

    Returns a list of length ``model_input_len`` with 1.0 for model-generated
    positions and 0.0 elsewhere.
    """
    mask = [0.0] * model_input_len
    for token_start, length, _ in spans:
        for j in range(length):
            pos = token_start - 1 + j
            if 0 <= pos < model_input_len:
                mask[pos] = 1.0
    return mask


def build_ui_token_mask(
    spans: List[Tuple[int, int, int]],
    full_token_len: int,
) -> List[int]:
    """Build a per-token mask for the visualization UI.

    Returns a list of length ``full_token_len`` where 0 means
    prompt/environment (masked) and a positive integer is the 1-based
    turn index of the model completion.
    """
    mask = [0] * full_token_len
    for token_start, length, turn_idx in spans:
        for j in range(length):
            pos = token_start + j
            if 0 <= pos < full_token_len:
                mask[pos] = turn_idx
    return mask
