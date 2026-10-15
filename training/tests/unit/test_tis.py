"""Truncated importance sampling scales only trained-token advantages."""

import math

import pytest
from fireworks.training import sdk
from training.utils.rl.losses import PolicyBatch, build_policy_datums
from training.utils.rl.tis import apply_tis


def _datum(n: int = 3):
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(list(range(1, n + 1))),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=list(range(2, n + 2)), dtype="int64", shape=[n]),
            "weights": sdk.TensorData(data=[0] + [1] * (n - 1), dtype="float32", shape=[n]),
        },
    )


def _batch(rollout, anchor, advantage=2.0, rows=1):
    return PolicyBatch(
        data=[_datum()] * rows,
        advantages=[advantage] * rows,
        prompt_lens=[2] * rows,
        rollout_logprobs=[rollout] * rows,
        anchor_logprobs=[anchor] * rows,
    )


def _datums(rollout, anchor, advantage=2.0):
    return build_policy_datums(_batch(rollout, anchor, advantage))


def test_token_weights_are_truncated_at_cap():
    rollout = [0.0, math.log(0.5), math.log(0.1)]
    anchor = [0.0, math.log(0.25), math.log(0.4)]  # ratios 0.5 and 4.0
    (datum,), metrics = apply_tis(_datums(rollout, anchor), _batch(rollout, anchor), cap=2.0)

    assert datum.loss_fn_inputs["advantages"].data == pytest.approx([0.0, 1.0, 4.0])
    assert datum.loss_fn_inputs["logprobs"].data == pytest.approx(anchor)
    assert metrics == pytest.approx({"tis_weight_mean": 1.25, "tis_clip_frac": 0.5})


def test_sequence_weight_uses_the_summed_log_ratio():
    rollout = [0.0, math.log(0.5), math.log(0.5)]
    anchor = [0.0, math.log(0.25), math.log(0.75)]  # product 0.5 * 1.5 = 0.75
    (datum,), metrics = apply_tis(_datums(rollout, anchor), _batch(rollout, anchor), cap=2.0, level="sequence")

    assert datum.loss_fn_inputs["advantages"].data == pytest.approx([0.0, 1.5, 1.5])
    assert metrics == pytest.approx({"tis_weight_mean": 0.75, "tis_clip_frac": 0.0})


def test_rollout_anchor_is_a_no_op():
    rollout = [0.0, math.log(0.5), math.log(0.1)]
    datums = _datums(rollout, rollout)
    (datum,), metrics = apply_tis(datums, _batch(rollout, rollout), cap=2.0)

    assert datum.loss_fn_inputs["advantages"].data == datums[0].loss_fn_inputs["advantages"].data
    assert metrics == {"tis_weight_mean": 1.0, "tis_clip_frac": 0.0}


@pytest.mark.parametrize(
    "kwargs,rows,message",
    [({"cap": 0.5}, 1, "cap"), ({"cap": 2.0, "level": "batch"}, 1, "level"), ({"cap": 2.0}, 2, "one rollout")],
)
def test_rejects_invalid_settings(kwargs, rows, message):
    rollout = [0.0, -1.0, -1.0]
    with pytest.raises(ValueError, match=message):
        apply_tis(_datums(rollout, rollout), _batch(rollout, rollout, rows=rows), **kwargs)


def test_documented_fork_step_composes_anchor_tis_and_metrics():
    """The rl-custom-loss.md recipe: recompute the anchor, apply TIS, log metrics."""
    from types import SimpleNamespace

    from training.utils.rl.algorithm import grpo as objective
    from training.utils.rl.anchor import anchor_drift_metrics, recompute_anchor
    from training.utils.rl.common import inference_drift_metrics
    from training.utils.supervised import build_datum_from_token_mask

    def logprob_rows(*rows):
        return SimpleNamespace(
            loss_fn_outputs=[{"logprobs": sdk.TensorData(data=row, dtype="float32", shape=[len(row)])} for row in rows]
        )

    snapshot = [-0.4, -0.2, -0.3]
    calls = []

    class Policy:
        def forward(self, data, loss_fn):
            calls.append(("forward", loss_fn))
            return logprob_rows(snapshot)

        def forward_backward(self, datums, loss_fn, loss_fn_config):
            calls.append(("forward_backward", loss_fn, loss_fn_config))
            return SimpleNamespace(metrics={}, **vars(logprob_rows([-0.4, -0.1, -0.3])))

    datum = build_datum_from_token_mask(token_ids=[10, 11, 12, 13], token_mask=[0, 1, 1, 1]).datum
    batch = PolicyBatch(
        data=[datum],
        advantages=[1.0],
        prompt_lens=[2],
        rollout_logprobs=[[-0.4, -0.3, -0.3]],
        raw_inference_logprobs=[[-0.4, -0.3, -0.3]],
    )
    config = objective.GRPOConfig()
    policy = Policy()

    batch, _forward = recompute_anchor(policy, batch)
    metrics = anchor_drift_metrics(batch)
    datums = objective.build_datums(batch, config)
    datums, tis = apply_tis(datums, batch, cap=2.0)
    result = policy.forward_backward(datums, objective.LOSS_FN, loss_fn_config=objective.loss_fn_config(config))
    result.metrics.update(objective.metrics(batch, result, config))
    result.metrics.update(inference_drift_metrics(batch, result))
    result.metrics.update(metrics)
    result.metrics.update(tis)

    assert [c[0] for c in calls] == ["forward", "forward_backward"]
    assert datums[0].loss_fn_inputs["logprobs"].data == pytest.approx([0.0, -0.2, -0.3])
    assert datums[0].loss_fn_inputs["advantages"].data == pytest.approx([0.0, math.exp(0.1), 1.0])
    assert result.metrics["tis_weight_mean"] == pytest.approx((math.exp(0.1) + 1.0) / 2)
    assert result.metrics["anchor_k1"] == pytest.approx(0.05)
    # The policy ratio is against the recomputed anchor: exp(0.1) and exp(0.0).
    assert result.metrics["ppo_ratio_mean"] == pytest.approx((math.exp(0.1) + 1.0) / 2)
    assert {"ppo_ratio_mean", "ppo_clip_frac", "ppo_kl", "inference_k3"} <= result.metrics.keys()
