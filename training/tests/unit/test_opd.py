"""Unit tests for sampled-token OPD helpers."""

from __future__ import annotations

from math import exp

import pytest
import tinker

from training.recipes.opd_loop import (
    _align_completion_logprobs,
    _extract_scored_token_logprobs,
)
from training.utils.opd import (
    OPDPromptGroup,
    build_opd_server_datums,
    combine_opd_prompt_groups,
)


def _server_importance_sampling_loss(
    current_logprobs: list[float],
    sampling_logprobs: list[float],
    advantages: list[float],
) -> float:
    return -sum(
        exp(current_lp - sampling_lp) * advantage
        for current_lp, sampling_lp, advantage in zip(
            current_logprobs,
            sampling_logprobs,
            advantages,
            strict=True,
        )
    )


def _datum(
    *,
    tokens: list[int] | None = None,
    loss_mask: list[float] | None = None,
) -> tinker.Datum:
    if tokens is None:
        tokens = [10, 11, 12, 13, 14]
    target_tokens = tokens[1:]
    inputs = {
        "target_tokens": tinker.TensorData(
            data=target_tokens,
            dtype="int64",
            shape=[len(target_tokens)],
        ),
    }
    if loss_mask is not None:
        inputs["loss_mask"] = tinker.TensorData(
            data=loss_mask,
            dtype="float32",
            shape=[len(loss_mask)],
        )

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs=inputs,
    )


def test_build_opd_server_datums_encodes_teacher_minus_sampling_advantages() -> None:
    datum = _datum()
    teacher_lp = [[-9.0, -0.2, -1.0, -0.3]]
    sampling_lp = [[-8.0, -0.5, -0.8, -0.7]]

    server_datums, metrics = build_opd_server_datums(
        [datum],
        teacher_lp,
        sampling_lp,
        prompt_lens=[2],
        loss_scale=2.0,
    )

    inputs = server_datums[0].loss_fn_inputs
    assert inputs["target_tokens"].data == [11, 12, 13, 14]
    assert inputs["logprobs"].data == sampling_lp[0]
    assert inputs["advantages"].data == pytest.approx([0.0, 0.6, -0.4, 0.8])
    assert metrics["opd_active_tokens"] == pytest.approx(3.0)
    assert metrics["opd_sampled_reverse_kl"] == pytest.approx((-0.3 + 0.2 - 0.4) / 3)
    assert metrics["opd_advantage"] == pytest.approx((0.6 - 0.4 + 0.8) / 3)


def test_opd_datums_make_server_loss_equal_sampled_reverse_kl_at_rollout_policy() -> None:
    datum = _datum()
    teacher_lp = [[-9.0, -0.2, -1.0, -0.3]]
    sampling_lp = [[-8.0, -0.5, -0.8, -0.7]]

    server_datums, _metrics = build_opd_server_datums(
        [datum],
        teacher_lp,
        sampling_lp,
        prompt_lens=[2],
    )

    inputs = server_datums[0].loss_fn_inputs
    old_lp = inputs["logprobs"].data
    advantages = inputs["advantages"].data
    loss = _server_importance_sampling_loss(old_lp, old_lp, advantages)

    active_positions = [1, 2, 3]
    expected_sampled_reverse_kl = sum(
        sampling_lp[0][pos] - teacher_lp[0][pos] for pos in active_positions
    )
    assert loss == pytest.approx(expected_sampled_reverse_kl)

    pos = 1
    eps = 1e-6
    plus_lp = list(old_lp)
    minus_lp = list(old_lp)
    plus_lp[pos] += eps
    minus_lp[pos] -= eps
    finite_difference_grad = (
        _server_importance_sampling_loss(plus_lp, old_lp, advantages)
        - _server_importance_sampling_loss(minus_lp, old_lp, advantages)
    ) / (2 * eps)
    assert finite_difference_grad == pytest.approx(sampling_lp[0][pos] - teacher_lp[0][pos])


def test_build_opd_server_datums_folds_loss_mask_into_advantages() -> None:
    datum = _datum(loss_mask=[1.0, 1.0, 0.0, 1.0])

    server_datums, metrics = build_opd_server_datums(
        [datum],
        teacher_logprobs=[[-1.0, -0.1, -0.1, -0.1]],
        sampling_logprobs=[[-1.0, -0.5, -0.5, -0.5]],
        prompt_lens=[2],
    )

    assert server_datums[0].loss_fn_inputs["advantages"].data == pytest.approx(
        [0.0, 0.4, 0.0, 0.4]
    )
    assert metrics["opd_active_tokens"] == pytest.approx(2.0)


def test_combine_opd_prompt_groups_uses_explicit_teacher_and_sampling_fields() -> None:
    datum = _datum()
    group = OPDPromptGroup(
        data=[datum],
        teacher_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
        sampling_logprobs=[[-0.5, -0.6, -0.7, -0.8]],
        prompt_len=2,
        rewards=[0.0],
    )

    data, teacher_logprobs, prompt_lens, sampling_logprobs = combine_opd_prompt_groups([group])

    assert data == [datum]
    assert teacher_logprobs == group.teacher_logprobs
    assert prompt_lens == [2]
    assert sampling_logprobs == group.sampling_logprobs


def test_combine_opd_prompt_groups_rejects_mismatched_teacher_scores() -> None:
    group = OPDPromptGroup(
        data=[_datum()],
        teacher_logprobs=[],
        sampling_logprobs=[[-0.5, -0.6, -0.7, -0.8]],
        prompt_len=2,
        rewards=[0.0],
    )

    with pytest.raises(ValueError, match="teacher_logprobs length"):
        combine_opd_prompt_groups([group])


def test_build_opd_server_datums_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="teacher_logprobs must have length 1"):
        build_opd_server_datums(
            [_datum()],
            teacher_logprobs=[],
            sampling_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
            prompt_lens=[2],
        )


def test_align_completion_logprobs_pads_prompt_prefix_for_non_echo_generation() -> None:
    aligned = _align_completion_logprobs(
        [-0.4, -0.5],
        prompt_len=3,
        target_len=4,
        echoed=False,
    )

    assert aligned == [0.0, 0.0, -0.4, -0.5]


def test_align_completion_logprobs_trims_echoed_logprobs() -> None:
    aligned = _align_completion_logprobs(
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        prompt_len=3,
        target_len=4,
        echoed=True,
    )

    assert aligned == [-0.1, -0.2, -0.3, -0.4]


def test_extract_scored_token_logprobs_drops_unconditional_and_extra_token() -> None:
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"logprob": 0.0},
                        {"logprob": -0.1},
                        {"logprob": -0.2},
                        {"logprob": -0.3},
                        {"logprob": -99.0},
                    ],
                },
            }
        ]
    }

    assert _extract_scored_token_logprobs(response, target_len=3) == [-0.1, -0.2, -0.3]
