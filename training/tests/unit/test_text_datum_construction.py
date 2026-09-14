"""Text construction preserves the canonical next-token loss coordinates."""

import numpy as np
import pytest
import tinker
import torch
from training.renderer.supervised import datum_from_model_input_weights
from training.utils.supervised import (
    build_datum_from_token_mask,
    build_datum_from_tokens_and_weights,
    build_training_datum_from_token_mask,
)


def _assert_datum(actual, expected, include_loss_mask):
    inputs = dict(expected.loss_fn_inputs)
    if include_loss_mask:
        inputs["loss_mask"] = inputs["weights"]
    assert actual.model_input.to_ints() == expected.model_input.to_ints()
    assert actual.loss_fn_inputs.keys() == inputs.keys()
    for name, tensor in inputs.items():
        assert actual.loss_fn_inputs[name].dtype == tensor.dtype
        assert actual.loss_fn_inputs[name].shape == tensor.shape
        np.testing.assert_array_equal(
            actual.loss_fn_inputs[name].to_numpy(), tensor.to_numpy()
        )


@pytest.mark.parametrize(
    "max_seq_len,reduction,include_loss_mask,weights",
    [
        (None, "none", False, [9, 0, 0.1, 1, -0.3, 2]),
        (2, "none", True, [9, 0, 1, 1, 1, 1]),
        (4, "none", True, [9, 0, 0.1, 1, -0.3, 2]),
        (20, "mean", False, [9, 0, 0.1, 1, -0.3, 2]),
        (None, "mean", True, [0] * 6),
    ],
)
def test_text_datum_matches_canonical_loss(
    max_seq_len, reduction, include_loss_mask, weights
):
    tokens = [10, 11, 12, 13, 14, 15]
    weights = list(weights)
    expected = datum_from_model_input_weights(
        tinker.ModelInput.from_ints(tokens),
        torch.tensor(weights, dtype=torch.float32),
        max_length=max_seq_len,
        reduction=reduction,
    )
    rendered = build_datum_from_tokens_and_weights(
        tokens,
        weights,
        max_seq_len=max_seq_len,
        reduction=reduction,
        include_loss_mask=include_loss_mask,
    )
    _assert_datum(rendered.datum, expected, include_loss_mask)
    assert rendered.token_ids == tokens[:max_seq_len]
    assert rendered.token_weights == [0.0] + expected.loss_fn_inputs["weights"].data
    # Returned metadata belongs to this datum, not the caller's mutable lists.
    tokens[0] = 99
    weights[1] = 99
    assert rendered.token_ids[0] == 10
    assert rendered.token_weights[1] == 0


@pytest.mark.parametrize(
    "builder", [build_datum_from_token_mask, build_training_datum_from_token_mask]
)
@pytest.mark.parametrize("max_seq_len,include_loss_mask", [(None, False), (5, True)])
def test_mask_datum_matches_canonical_loss_and_owns_arrays(
    builder, max_seq_len, include_loss_mask
):
    tokens = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    mask = np.array([1, -2, 0.1, float("nan"), 0, 1])
    expected = datum_from_model_input_weights(
        tinker.ModelInput.from_ints(tokens.tolist()),
        torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.float32),
        max_length=max_seq_len,
    )
    result = builder(
        tokens, mask, max_seq_len=max_seq_len, include_loss_mask=include_loss_mask
    )
    actual = result.datum if builder is build_datum_from_token_mask else result
    tokens[:] = 99
    mask[:] = 0
    _assert_datum(actual, expected, include_loss_mask)
    if builder is build_datum_from_token_mask:
        assert result.token_ids == [10, 11, 12, 13, 14, 15][:max_seq_len]
        assert result.token_weights == [0.0] + expected.loss_fn_inputs["weights"].data


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["1", "2", "3"], [1, 2, 3]),
        (np.array([1, 2, 3], dtype=np.uint64), [1, 2, 3]),
        ([2**60 + 1, 2.9, 3], [2**60 + 1, 2, 3]),
    ],
)
def test_text_token_coercion_preserves_values(tokens, expected):
    rendered = build_datum_from_token_mask(tokens, [0, 1, 1])
    assert rendered.token_ids == expected
    assert rendered.datum.model_input.to_ints() == expected[:-1]
    assert rendered.datum.loss_fn_inputs["target_tokens"].data == expected[1:]


@pytest.mark.parametrize(
    "tokens,weights,kwargs,error",
    [
        ([1], [1], {}, ValueError),
        ([1, 2], [1], {}, ValueError),
        ([1, 2], [1, 1], {"max_seq_len": 1}, ValueError),
        ([1, 2], [None, 1], {}, TypeError),
        (np.array([1, 2**63], dtype=np.uint64), [0, 1], {}, OverflowError),
        ([1, 2**63], [0, 1], {}, OverflowError),
        ([1, -(2**63) - 1], [0, 1], {}, OverflowError),
        ([2**63, np.longdouble(1.25), 1, "1.9"], [1] * 4, {}, ValueError),
        ([1, None], [0, 1], {}, TypeError),
    ],
)
def test_text_datum_validation(tokens, weights, kwargs, error):
    with pytest.raises(error):
        build_datum_from_tokens_and_weights(tokens, weights, **kwargs)


def test_unknown_reduction_is_rejected():
    with pytest.raises(ValueError, match="Unknown reduction"):
        build_datum_from_tokens_and_weights([1, 2], [1, 1], reduction="sum")
