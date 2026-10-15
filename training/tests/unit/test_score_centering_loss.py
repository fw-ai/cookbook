from __future__ import annotations

import math

import pytest
import torch
from fireworks.training import sdk
from training.utils.rl.algorithm.score_centering import (
    MAX_SCORE_CENTERING_TOP_K,
    ScoreCenteringConfig,
    build_datums,
    make_loss_fn,
    sampling_kwargs,
)
from training.utils.rl.losses import PolicyBatch


def _source_datum() -> sdk.Datum:
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints([10, 11]),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=[11, 12], dtype="int64", shape=[2]),
            "weights": sdk.TensorData(data=[0, 1], dtype="int64", shape=[2]),
        },
    )


def _batch(topk_ids, topk_logprobs, *, advantage: float = 2.0, **kwargs) -> PolicyBatch:
    return PolicyBatch(
        data=[_source_datum()],
        advantages=[advantage],
        prompt_lens=[1],
        rollout_logprobs=[[0.0, -0.5]],
        inference_topk_token_ids=topk_ids,
        inference_topk_logprobs=topk_logprobs,
        **kwargs,
    )


def test_score_centering_matches_topk_tail_formula_and_gradients() -> None:
    config = ScoreCenteringConfig(top_k=2)
    batch = _batch([[[], [20, 21]]], [[[], [math.log(0.6), math.log(0.2)]]])
    data = build_datums(batch, config)
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
    loss_fn = make_loss_fn(batch, config)
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
        build_datums(_batch([[[], [20]]], [[[], [-0.1]]]), ScoreCenteringConfig(top_k=2))


def test_score_centering_accepts_dedicated_top_k_eight() -> None:
    config = ScoreCenteringConfig(top_k=MAX_SCORE_CENTERING_TOP_K)
    assert sampling_kwargs(config) == {"top_logprobs": MAX_SCORE_CENTERING_TOP_K}
    assert ScoreCenteringConfig().top_k == 5

    with pytest.raises(ValueError, match="top_logprobs cap"):
        ScoreCenteringConfig(top_k=MAX_SCORE_CENTERING_TOP_K + 1)


def test_score_centering_requires_one_topk_row_per_datum() -> None:
    with pytest.raises(ValueError, match="one top-k distribution per datum"):
        build_datums(_batch([], []), ScoreCenteringConfig(top_k=2))


def test_score_centering_rejects_token_advantages() -> None:
    batch = _batch(
        [[[], [20, 21]]],
        [[[], [-0.5, -1.6]]],
        token_advantages=[[0.0, 1.0]],
    )
    with pytest.raises(ValueError, match="token_advantages"):
        make_loss_fn(batch, ScoreCenteringConfig(top_k=2))
