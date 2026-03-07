"""Tests that the training loss mask and UI display mask stay aligned.

Both masks are derived from the same ``compute_model_output_spans`` function,
so by construction they agree on *which* tokens are model-generated.

This test verifies the property explicitly: for every position that the
training loss mask marks as 1.0 (model-generated), the UI mask must mark it
with a nonzero turn index — and vice-versa, after accounting for the
1-position shift between token-coordinate and logprob-coordinate.
"""

from __future__ import annotations

import pytest

from training.examples.frozen_lake.masking import (
    build_training_loss_mask,
    build_ui_token_mask,
    compute_model_output_spans,
)


def _make_traces(turns: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build minimal token_turn_traces and model_request_traces from a spec.

    Each element of ``turns`` is a dict with:
      - prompt_ids: list[int]
      - completion_ids: list[int]
      - assistant_turn_len: int (optional, for intermediate turns)
    """
    token_turn_traces = []
    model_request_traces = []
    for t in turns:
        token_turn_traces.append({
            "prompt_ids": t["prompt_ids"],
            "completion_ids": t["completion_ids"],
        })
        if "assistant_turn_len" in t:
            model_request_traces.append({"assistant_turn_len": t["assistant_turn_len"]})
        else:
            model_request_traces.append({})
    return token_turn_traces, model_request_traces


SINGLE_TURN = [
    {
        "prompt_ids": [1, 2, 3, 4, 5],
        "completion_ids": [10, 11, 12],
    },
]

MULTI_TURN_2 = [
    {
        "prompt_ids": [1, 2, 3],
        "completion_ids": [10, 11],
        "assistant_turn_len": 2,
    },
    {
        "prompt_ids": [1, 2, 3, 10, 11, 20, 21, 22],
        "completion_ids": [30, 31, 32],
    },
]

MULTI_TURN_3 = [
    {
        "prompt_ids": [1, 2],
        "completion_ids": [10],
        "assistant_turn_len": 1,
    },
    {
        "prompt_ids": [1, 2, 10, 20, 21],
        "completion_ids": [30],
        "assistant_turn_len": 1,
    },
    {
        "prompt_ids": [1, 2, 10, 20, 21, 30, 40, 41, 42],
        "completion_ids": [50, 51],
    },
]

KIMI_STYLE_MULTI_TURN = [
    {
        "prompt_ids": [1, 2, 3],
        "completion_ids": [99, 10, 90],  # prefill, raw tool call token(s), im_end
        "assistant_turn_len": 3,
    },
    {
        "prompt_ids": [1, 2, 3, 99, 10, 90, 20, 21],
        "completion_ids": [99, 30, 90],
    },
]


@pytest.mark.parametrize("scenario", [SINGLE_TURN, MULTI_TURN_2, MULTI_TURN_3],
                         ids=["single", "multi_2", "multi_3"])
def test_training_and_ui_masks_agree(scenario):
    """The training mask (logprob coord) and UI mask (token coord) must agree.

    Specifically: position p in UI mask is model-generated (>0) iff position
    p-1 in the training mask is 1.0 — because the training mask is shifted by
    one to predict the *next* token.
    """
    ttt, mrt = _make_traces(scenario)
    spans = compute_model_output_spans(ttt, mrt)

    last = ttt[-1]
    full_ids = list(last["prompt_ids"]) + list(last["completion_ids"])
    full_len = len(full_ids)
    model_input_len = full_len - 1

    ui_mask = build_ui_token_mask(spans, full_len)
    train_mask = build_training_loss_mask(spans, model_input_len)

    for p in range(full_len):
        ui_is_model = ui_mask[p] > 0
        if p == 0:
            assert not ui_is_model, "position 0 should always be prompt"
            continue
        train_is_model = train_mask[p - 1] > 0.5
        assert ui_is_model == train_is_model, (
            f"Mismatch at token position {p}: "
            f"ui_mask[{p}]={ui_mask[p]}, train_mask[{p-1}]={train_mask[p-1]}"
        )


@pytest.mark.parametrize("scenario", [SINGLE_TURN, MULTI_TURN_2, MULTI_TURN_3],
                         ids=["single", "multi_2", "multi_3"])
def test_spans_are_shared(scenario):
    """Both masks use the exact same spans — the key alignment invariant."""
    ttt, mrt = _make_traces(scenario)
    spans = compute_model_output_spans(ttt, mrt)

    last = ttt[-1]
    full_ids = list(last["prompt_ids"]) + list(last["completion_ids"])
    full_len = len(full_ids)
    model_input_len = full_len - 1

    ui_mask = build_ui_token_mask(spans, full_len)
    train_mask = build_training_loss_mask(spans, model_input_len)

    ui_model_positions = {p for p in range(full_len) if ui_mask[p] > 0}
    train_model_positions = {(p + 1) for p in range(model_input_len) if train_mask[p] > 0.5}

    assert ui_model_positions == train_model_positions, (
        f"Model token positions differ:\n"
        f"  UI:    {sorted(ui_model_positions)}\n"
        f"  Train: {sorted(train_model_positions)}"
    )


def test_single_turn_mask_values():
    """Verify exact mask values for the single-turn case."""
    ttt, mrt = _make_traces(SINGLE_TURN)
    spans = compute_model_output_spans(ttt, mrt)

    assert spans == [(5, 3, 1)]

    ui_mask = build_ui_token_mask(spans, 8)
    assert ui_mask == [0, 0, 0, 0, 0, 1, 1, 1]

    train_mask = build_training_loss_mask(spans, 7)
    assert train_mask == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_multi_turn_mask_values():
    """Verify exact mask values for the 2-turn case."""
    ttt, mrt = _make_traces(MULTI_TURN_2)
    spans = compute_model_output_spans(ttt, mrt)

    assert spans == [(3, 2, 1), (8, 3, 2)]

    full_len = 11
    model_input_len = 10

    ui_mask = build_ui_token_mask(spans, full_len)
    assert ui_mask == [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2]

    train_mask = build_training_loss_mask(spans, model_input_len)
    assert train_mask == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_empty_traces():
    """Empty traces should produce empty spans and masks."""
    spans = compute_model_output_spans([], [])
    assert spans == []
    assert build_training_loss_mask(spans, 0) == []
    assert build_ui_token_mask(spans, 0) == []


def test_fallback_when_assistant_turn_len_missing():
    """When model_request_traces is empty, falls back to completion_ids length."""
    ttt, _ = _make_traces(MULTI_TURN_2)
    spans_with_mrt = compute_model_output_spans(ttt, [{"assistant_turn_len": 2}])
    spans_no_mrt = compute_model_output_spans(ttt, [])

    assert spans_no_mrt[0] == (3, 2, 1), "should fall back to completion_ids len"
    assert spans_with_mrt[0] == spans_no_mrt[0]


def test_kimi_style_mask_includes_prefill_and_im_end_tokens():
    """Assistant framing tokens should be masked as model output too."""
    ttt, mrt = _make_traces(KIMI_STYLE_MULTI_TURN)
    spans = compute_model_output_spans(ttt, mrt)

    assert spans == [(3, 3, 1), (8, 3, 2)]

    ui_mask = build_ui_token_mask(spans, 11)
    assert ui_mask == [0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2]

    train_mask = build_training_loss_mask(spans, 10)
    assert train_mask == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
