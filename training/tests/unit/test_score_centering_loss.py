from __future__ import annotations

import math

import pytest
import tinker
import torch

from training.utils.rl.score_centering import (
    MAX_SCORE_CENTERING_TOP_K,
    ScoreCenteringConfig,
    build_score_centering_datums,
    make_score_centering_loss_fn,
    validate_score_centering_config,
)


def _source_datum() -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[11, 12], dtype="int64", shape=[2]
            ),
            "weights": tinker.TensorData(
                data=[0, 1], dtype="int64", shape=[2]
            ),
        },
    )


def test_score_centering_matches_topk_tail_formula_and_gradients() -> None:
    config = ScoreCenteringConfig(top_k=2)
    data, sampler_logprobs = build_score_centering_datums(
        [_source_datum()],
        sampler_topk_token_ids=[[[], [20, 21]]],
        sampler_topk_logprobs=[
            [[], [math.log(0.6), math.log(0.2)]]
        ],
        config=config,
    )
    assert data[0].loss_fn_inputs["target_tokens"].shape == [2, 3]
    assert data[0].loss_fn_inputs["target_tokens"].data == [
        11,
        0,
        0,
        12,
        20,
        21,
    ]
    assert data[0].loss_fn_inputs["weights"].data == [0, 0.0, 0.0, 1, 0.0, 0.0]

    trainer_logprobs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [math.log(0.4), math.log(0.5), math.log(0.25)],
        ],
        requires_grad=True,
    )
    loss_fn = make_score_centering_loss_fn(
        advantages=[2.0],
        sampler_topk_logprobs=sampler_logprobs,
        config=config,
    )
    loss, metrics = loss_fn(data, [trainer_logprobs])
    loss.backward()

    # q_tail=0.2, p_tail=0.25, rho=0.8, residual=[0.2, 0.0].
    expected_centered = math.log(0.4) - 0.2 * math.log(0.5)
    assert loss.item() == pytest.approx(-2.0 * expected_centered)
    torch.testing.assert_close(
        trainer_logprobs.grad,
        torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.4, 0.0]]),
    )
    assert metrics["score_centering/sampler_head_mass_mean"] == pytest.approx(0.8)
    assert metrics["score_centering/trainer_head_mass_mean"] == pytest.approx(0.75)
    assert metrics["score_centering/tail_mass_ratio_mean"] == pytest.approx(0.8)


def test_score_centering_requires_topk_on_active_positions() -> None:
    with pytest.raises(ValueError, match="fewer than top_k"):
        build_score_centering_datums(
            [_source_datum()],
            sampler_topk_token_ids=[[[], [20]]],
            sampler_topk_logprobs=[[[], [-0.1]]],
            config=ScoreCenteringConfig(top_k=2),
        )


def test_score_centering_accepts_dedicated_top_k_eight() -> None:
    validate_score_centering_config(ScoreCenteringConfig(top_k=MAX_SCORE_CENTERING_TOP_K))
    assert ScoreCenteringConfig().top_k == 5

    with pytest.raises(ValueError, match="top_logprobs cap"):
        validate_score_centering_config(
            ScoreCenteringConfig(top_k=MAX_SCORE_CENTERING_TOP_K + 1)
        )
