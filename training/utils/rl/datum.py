"""Datum construction helpers for RL recipes.

These are pure plumbing helpers for converting a sampled completion's
token IDs into the ``tinker.Datum`` shape the trainer expects. They are
deliberately small and unopinionated — recipes still own reward, advantage,
and PromptGroup assembly.

Typical use inside a recipe::

    from training.utils.rl.datum import (
        make_policy_datum,
        make_reference_datum,
        align_inference_logprobs,
    )

    for s in sampled:
        tokens = s.full_tokens
        if len(tokens) < 2:
            continue
        rm = build_r3_routing_matrices(...) if cfg.router_replay else None
        policy_data.append(make_policy_datum(tokens, routing_matrices=rm))
        if reference is not None:
            reference_data.append(make_reference_datum(tokens))
        inf_logprobs_aligned.append(
            align_inference_logprobs(
                s.inference_logprobs,
                prompt_len=s.prompt_len,
                total_len=len(tokens) - 1,
                echoed=getattr(s, "logprobs_echoed", False),
            )
        )

The helpers do *not* decide whether router_replay is on, what reward
function to use, or how advantages are normalised — those choices stay
with the recipe.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import tinker

__all__ = [
    "make_policy_datum",
    "make_reference_datum",
    "align_inference_logprobs",
]


def make_policy_datum(
    tokens: Sequence[int],
    *,
    routing_matrices: Optional[List[str]] = None,
) -> tinker.Datum:
    """Build a single policy datum from full-sequence token IDs.

    Constructs the (input, target) shifted pair the trainer expects:
    input is ``tokens[:-1]``, targets are ``tokens[1:]``. ``routing_matrices``
    is forwarded to ``tinker.ModelInput.from_ints`` for R3 (routing replay).

    Raises:
        ValueError: if ``tokens`` has fewer than 2 elements.
    """
    if len(tokens) < 2:
        raise ValueError(
            f"make_policy_datum requires at least 2 tokens, got {len(tokens)}"
        )
    model_input_len = len(tokens) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(
            tokens[:-1], routing_matrices=routing_matrices,
        ),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=list(tokens[1:]),
                dtype="int64",
                shape=[model_input_len],
            ),
        },
    )


def make_reference_datum(tokens: Sequence[int]) -> tinker.Datum:
    """Build a single reference datum from full-sequence token IDs.

    Identical to :func:`make_policy_datum` but never carries routing
    matrices — the reference forward pass should not replay routing.
    """
    if len(tokens) < 2:
        raise ValueError(
            f"make_reference_datum requires at least 2 tokens, got {len(tokens)}"
        )
    model_input_len = len(tokens) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=list(tokens[1:]),
                dtype="int64",
                shape=[model_input_len],
            ),
        },
    )


def align_inference_logprobs(
    inference_logprobs: Sequence[float],
    *,
    prompt_len: int,
    total_len: int,
    echoed: bool,
) -> List[float]:
    """Pad sampling-time logprobs to the training-aligned length.

    The trainer expects per-position logprobs of length ``total_len``
    (= ``len(full_tokens) - 1``), aligned to the targets at
    ``tokens[1:]``. The deployment returns either:

    * **Echoed mode** (``echoed=True``): ``len(inference_logprobs) == total_len``
      — already aligned, returned as-is.
    * **Completion-only mode** (``echoed=False``): logprobs only cover
      completion positions. The prompt prefix is padded with zeros so the
      result has length ``total_len``. Position ``prompt_len - 1`` is the
      first completion target; everything before is masked out by the
      loss anyway.

    Args:
        inference_logprobs: Per-token logprobs returned by the deployment.
        prompt_len: Number of prompt tokens.
        total_len: Target alignment length (typically ``len(tokens) - 1``).
        echoed: True if the deployment returned full-sequence logprobs.

    Returns:
        A list of length ``total_len``.

    Raises:
        ValueError: if ``inference_logprobs`` is empty.
    """
    if not inference_logprobs:
        raise ValueError("inference_logprobs is empty — deployment must return logprobs")
    if echoed:
        return list(inference_logprobs)
    response_start = max(0, prompt_len - 1)
    return [0.0] * response_start + list(inference_logprobs)
