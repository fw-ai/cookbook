"""Unit tests for the KL-regularization port in ``countdown_rl``.

Pure-math tests against a stubbed reference client -- no Fireworks infra.
"""

from __future__ import annotations

import math

import torch

from training.examples.serverless_rl.countdown_rl import (
    _discounted_future_sum_vectorized,
    _incorporate_kl_penalty,
    _remove_mask,
)


class _StubFuture:
    """Mimics the ``.result()`` interface of a Tinker concurrent future."""

    def __init__(self, value: list[float]) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> list[float]:  # noqa: ARG002
        return self._value


class _StubRefClient:
    """Reference sampling client returning a fixed logprob list (first entry BOS)."""

    def __init__(self, logprobs: list[float]) -> None:
        self._logprobs = logprobs
        self.calls = 0

    def compute_logprobs(self, prompt) -> _StubFuture:  # noqa: ARG002
        self.calls += 1
        return _StubFuture(list(self._logprobs))


class _StubModelInput:
    """Stand-in for ``tinker.ModelInput``: records the appended last target token."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = list(tokens)
        self.appended: int | None = None

    def append_int(self, token: int) -> "_StubModelInput":
        self.appended = token
        # Return a copy representing the full sequence; the stub ref client
        # ignores the actual content.
        return _StubModelInput(self.tokens + [token])


class _StubDatum:
    """Minimal stand-in for ``tinker.Datum`` with a dict ``loss_fn_inputs``."""

    def __init__(self, model_input: _StubModelInput, loss_fn_inputs: dict) -> None:
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


def _make_datum(
    *,
    model_input_len: int,
    sampled_logprobs: list[float],
    target_tokens: list[int],
    mask: list[float],
    advantages: list[float],
) -> _StubDatum:
    # All loss_fn_inputs lists are length == model_input.length (Tinker's layout).
    assert len(target_tokens) == model_input_len
    assert len(sampled_logprobs) == model_input_len
    assert len(mask) == model_input_len
    assert len(advantages) == model_input_len
    return _StubDatum(
        model_input=_StubModelInput(list(target_tokens)),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": sampled_logprobs,
            "mask": mask,
            "advantages": advantages,
        },
    )


def test_incorporate_kl_penalty_matches_tinker_formula():
    # Two datums: model_input.length=4, prompt at position 0 (mask 0), response 1..3.
    # compute_logprobs returns 5 values (full seq = 4 + appended last token); [1:] -> 4.
    kl_coef = 0.5
    ref_logprobs = [None, -1.0, -2.0, -3.0, -4.0]  # [1:] -> [-1,-2,-3,-4]
    ref = _StubRefClient(ref_logprobs)

    d0 = _make_datum(
        model_input_len=4,
        sampled_logprobs=[0.0, -0.5, -1.5, -3.5],
        target_tokens=[0, 7, 8, 9],
        mask=[0.0, 1.0, 1.0, 1.0],
        advantages=[0.0, 1.0, 1.0, 1.0],
    )
    d1 = _make_datum(
        model_input_len=4,
        sampled_logprobs=[0.0, -0.5, -0.5, -0.5],
        target_tokens=[0, 7, 8, 9],
        mask=[0.0, 1.0, 1.0, 1.0],
        advantages=[0.0, -2.0, -2.0, -2.0],
    )
    datums = [d0, d1]
    # Capture originals before the in-place mutation.
    original_advantages = [list(d.loss_fn_inputs["advantages"]) for d in datums]

    avg = _incorporate_kl_penalty(datums, ref, kl_coef, kl_discount_factor=0.0)

    # Recompute the expected adjustment independently of the implementation.
    sampled = [torch.tensor(d.loss_fn_inputs["logprobs"], dtype=torch.float32) for d in datums]
    masks = [torch.tensor(d.loss_fn_inputs["mask"], dtype=torch.float32) for d in datums]
    base = torch.tensor(ref_logprobs[1:], dtype=torch.float32)
    diffs = [(s - base) * m for s, m in zip(sampled, masks)]
    expected_avg = sum(d.sum() for d in diffs) / sum(m.sum() for m in masks)
    assert math.isclose(avg, float(expected_avg), rel_tol=1e-6)

    for d, orig_adv, diff, mask in zip(datums, original_advantages, diffs, masks):
        expected_adj = (kl_coef * mask * (expected_avg - diff)).tolist()
        expected_adv = [orig + adj for orig, adj in zip(orig_adv, expected_adj)]
        for got, exp in zip(d.loss_fn_inputs["advantages"], expected_adv):
            assert math.isclose(got, exp, rel_tol=1e-6)

    assert ref.calls == 2  # one compute_logprobs per datum
    assert d0.model_input.appended == 9  # last target token appended
    assert d1.model_input.appended == 9


def test_incorporate_kl_penalty_discounted_future_sum():
    # kl_discount_factor > 0: penalty becomes a discounted future sum.
    kl_coef = 0.25
    gamma = 0.9
    ref_logprobs = [None, 0.0, 0.0, 0.0, 0.0]  # [1:] -> [0,0,0,0]
    ref = _StubRefClient(ref_logprobs)

    d = _make_datum(
        model_input_len=4,
        sampled_logprobs=[0.0, 1.0, 2.0, 3.0],
        target_tokens=[0, 7, 8, 9],
        mask=[0.0, 1.0, 1.0, 1.0],
        advantages=[0.0, 0.0, 0.0, 0.0],
    )
    datums = [d]
    _incorporate_kl_penalty(datums, ref, kl_coef, kl_discount_factor=gamma)

    sampled = torch.tensor(d.loss_fn_inputs["logprobs"], dtype=torch.float32)
    mask = torch.tensor(d.loss_fn_inputs["mask"], dtype=torch.float32)
    base = torch.tensor(ref_logprobs[1:], dtype=torch.float32)
    diff = (sampled - base) * mask
    expected_avg = diff.sum() / mask.sum()
    raw_penalty = kl_coef * mask * (expected_avg - diff)
    expected_adj = _discounted_future_sum_vectorized(raw_penalty, gamma).tolist()
    for got, exp in zip(d.loss_fn_inputs["advantages"], expected_adj):
        assert math.isclose(got, exp, rel_tol=1e-6)


def test_remove_mask_strips_mask_key():
    datums = [
        _StubDatum(
            model_input=_StubModelInput([1, 2]),
            loss_fn_inputs={"target_tokens": [1, 2], "logprobs": [0.0, 0.0], "mask": [1.0, 1.0]},
        ),
        _StubDatum(
            model_input=_StubModelInput([3, 4]),
            loss_fn_inputs={"target_tokens": [3, 4], "logprobs": [0.0, 0.0]},
        ),
    ]
    _remove_mask(datums)
    assert "mask" not in datums[0].loss_fn_inputs
    assert "mask" not in datums[1].loss_fn_inputs
    # Other keys preserved.
    assert datums[0].loss_fn_inputs["target_tokens"] == [1, 2]


def test_kl_off_composition_skips_kl_but_still_strips_mask():
    """coef=0 (ref_client None): skip KL, still strip mask, run importance_sampling."""
    datums = [
        _StubDatum(
            model_input=_StubModelInput([1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": [0, 2, 3],
                "logprobs": [0.0, 0.2, 0.3],
                "advantages": [0.0, 1.0, 1.0],
                "mask": [0.0, 1.0, 1.0],
            },
        ),
        _StubDatum(
            model_input=_StubModelInput([4, 5, 6]),
            loss_fn_inputs={
                "target_tokens": [0, 5, 6],
                "logprobs": [0.0, 0.5, 0.6],
                "advantages": [0.0, -1.0, -1.0],
                "mask": [0.0, 1.0, 1.0],
            },
        ),
    ]
    ref_client = None

    captured = {}

    class _SpyTrainingClient:
        def forward_backward(self, data, loss_fn):
            captured["data"] = data
            captured["loss_fn"] = loss_fn
            return object()

    # Replicate countdown_rl._step's step-4 ordering.
    kl_policy_base = None
    if datums:
        if ref_client is not None:
            kl_policy_base = _incorporate_kl_penalty(
                datums, ref_client, kl_penalty_coef=0.0, kl_discount_factor=0.0
            )
        _remove_mask(datums)
        _SpyTrainingClient().forward_backward(datums, "importance_sampling")

    assert kl_policy_base is None
    assert all("mask" not in d.loss_fn_inputs for d in datums)
    assert datums[0].loss_fn_inputs["advantages"] == [0.0, 1.0, 1.0]
    assert datums[1].loss_fn_inputs["advantages"] == [0.0, -1.0, -1.0]
    assert captured["loss_fn"] == "importance_sampling"
    assert captured["data"] is datums

