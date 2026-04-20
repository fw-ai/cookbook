"""Unit tests for training.utils.rl.datum."""

from __future__ import annotations

import pytest
import tinker

from training.utils.rl.datum import (
    align_inference_logprobs,
    make_policy_datum,
    make_reference_datum,
)


# ---------------------------------------------------------------------------
# make_policy_datum
# ---------------------------------------------------------------------------


def test_make_policy_datum_shape_and_target_alignment():
    tokens = [10, 20, 30, 40, 50]
    datum = make_policy_datum(tokens)

    target = datum.loss_fn_inputs["target_tokens"]
    assert isinstance(datum, tinker.Datum)
    assert target.shape == [4]
    assert target.dtype == "int64"
    assert list(target.data) == [20, 30, 40, 50]


def test_make_policy_datum_passes_routing_matrices_through():
    tokens = [1, 2, 3]
    rm = ["{}", "{}"]
    datum = make_policy_datum(tokens, routing_matrices=rm)
    assert isinstance(datum, tinker.Datum)


def test_make_policy_datum_rejects_short_token_lists():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        make_policy_datum([42])
    with pytest.raises(ValueError, match="at least 2 tokens"):
        make_policy_datum([])


def test_make_policy_datum_accepts_tuple_input():
    datum = make_policy_datum((100, 200, 300))
    assert list(datum.loss_fn_inputs["target_tokens"].data) == [200, 300]


# ---------------------------------------------------------------------------
# make_reference_datum
# ---------------------------------------------------------------------------


def test_make_reference_datum_matches_policy_datum_shape():
    tokens = [5, 6, 7, 8]
    pd = make_policy_datum(tokens)
    rd = make_reference_datum(tokens)
    assert (
        pd.loss_fn_inputs["target_tokens"].shape
        == rd.loss_fn_inputs["target_tokens"].shape
    )
    assert list(pd.loss_fn_inputs["target_tokens"].data) == list(
        rd.loss_fn_inputs["target_tokens"].data
    )


def test_make_reference_datum_rejects_short_token_lists():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        make_reference_datum([0])


# ---------------------------------------------------------------------------
# align_inference_logprobs
# ---------------------------------------------------------------------------


def test_align_inference_logprobs_echoed_returns_unchanged():
    lp = [-1.0, -2.0, -3.0, -4.0]
    out = align_inference_logprobs(lp, prompt_len=2, total_len=4, echoed=True)
    assert out == lp
    assert out is not lp  # defensive copy


def test_align_inference_logprobs_completion_only_pads_with_zeros():
    completion_lp = [-0.5, -0.6, -0.7]  # 3 completion tokens
    out = align_inference_logprobs(
        completion_lp, prompt_len=4, total_len=6, echoed=False,
    )
    # response_start = prompt_len - 1 = 3, so 3 zeros prepended
    assert out == [0.0, 0.0, 0.0, -0.5, -0.6, -0.7]


def test_align_inference_logprobs_handles_prompt_len_zero():
    out = align_inference_logprobs([-1.0, -2.0], prompt_len=0, total_len=2, echoed=False)
    assert out == [-1.0, -2.0]


def test_align_inference_logprobs_handles_prompt_len_one():
    # response_start = max(0, 1 - 1) = 0, no padding
    out = align_inference_logprobs([-1.0], prompt_len=1, total_len=1, echoed=False)
    assert out == [-1.0]


def test_align_inference_logprobs_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        align_inference_logprobs([], prompt_len=4, total_len=6, echoed=False)
    with pytest.raises(ValueError, match="empty"):
        align_inference_logprobs([], prompt_len=4, total_len=6, echoed=True)
