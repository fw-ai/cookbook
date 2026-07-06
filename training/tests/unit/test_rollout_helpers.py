"""Unit tests for ``training.utils.rl.rollout.extract_completion``.

Locks in the token / logprob alignment contract:

  * ``token_ids`` is required; missing or empty raises.
  * ``token_ids`` and ``token_logprobs`` are filtered IN LOCKSTEP
    when the provider emits a null placeholder, so a leading or
    middle ``None`` token doesn't shift every remaining logprob
    onto the wrong token (which would silently corrupt PPO/GRPO
    ratio + KL for the affected completion).
"""

from __future__ import annotations

import pytest

from training.utils.rl.rollout import extract_completion


def _choice(token_ids, token_logprobs=None, finish_reason="stop"):
    return {
        "token_ids": token_ids,
        "logprobs": (
            {"token_logprobs": token_logprobs}
            if token_logprobs is not None
            else None
        ),
        "finish_reason": finish_reason,
    }


def test_basic_extract():
    call = extract_completion(
        _choice([10, 11, 12], [-0.1, -0.2, -0.3]),
        input_tokens=[1, 2, 3],
    )
    assert call.input_tokens == [1, 2, 3]
    assert call.output_tokens == [10, 11, 12]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_missing_token_ids_raises():
    with pytest.raises(ValueError, match="missing 'token_ids'"):
        extract_completion(_choice(None), input_tokens=[1])


def test_empty_token_ids_raises():
    with pytest.raises(ValueError, match="missing 'token_ids'"):
        extract_completion(_choice([]), input_tokens=[1])


def test_null_token_at_start_filters_logprobs_in_lockstep():
    """A provider that emits ``None`` at index 0 of ``token_ids`` used
    to drop the entry from tokens but keep ALL logprobs, then
    tail-trim the resulting length mismatch.  Every remaining
    logprob shifted onto the wrong token, silently corrupting
    PPO/GRPO ratio + KL for that completion.  Filtering both lists
    in lockstep keeps surviving (token, logprob) pairs aligned."""
    call = extract_completion(
        _choice(
            token_ids=[None, 11, 12, 13],
            token_logprobs=[-9.0, -0.1, -0.2, -0.3],
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [11, 12, 13]
    # The logprob paired with the null token (-9.0) is dropped, NOT
    # the trailing logprob (-0.3) — which would have shifted every
    # remaining logprob one slot to the left.
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_null_token_in_middle_filters_logprobs_in_lockstep():
    call = extract_completion(
        _choice(
            token_ids=[10, None, 12, 13],
            token_logprobs=[-0.1, -9.0, -0.2, -0.3],
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [10, 12, 13]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_no_null_tokens_unchanged():
    call = extract_completion(
        _choice(
            token_ids=[10, 11, 12],
            token_logprobs=[-0.1, -0.2, -0.3],
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [10, 11, 12]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_logprob_list_padded_by_one_truncates_to_token_count():
    """Some providers emit ``len(logprobs) == len(tokens) + 1``;
    the helper trims to ``len(tokens)`` (back-compat with the
    pre-lockstep behavior for non-null token cases)."""
    call = extract_completion(
        _choice(
            token_ids=[10, 11, 12],
            token_logprobs=[-0.1, -0.2, -0.3, -0.4],  # 1 too many
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [10, 11, 12]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_logprob_list_too_short_raises():
    with pytest.raises(ValueError, match="aligned with token_ids"):
        extract_completion(
            _choice(
                token_ids=[10, 11, 12],
                token_logprobs=[-0.1],  # 2 too few
            ),
            input_tokens=[1],
        )
