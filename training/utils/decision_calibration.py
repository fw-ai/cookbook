"""Temperature fit for K-way decision probabilities.

Trainer option-code logprobs are the serving logits shifted by one constant, so they give the
same K-way softmax. A temperature T divides the logits: argmax and accuracy stay, NLL moves.

``fit_temperature`` minimizes held-out NLL. NLL is convex in beta = 1/T, so a
golden-section search over a bounded beta range finds the minimum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fireworks.training.sdk.decision import DecisionCalibration, SamplerCheckpointMetadata

DEFAULT_T_MIN = 0.25
DEFAULT_T_MAX = 4.0
_GOLDEN_SECTION_ITERS = 80
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def mean_nll(
    logit_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float = 1.0,
) -> float:
    _check_rows(logit_rows, labels)
    total = 0.0
    for logits, label in zip(logit_rows, labels, strict=True):
        scaled = [logit / temperature for logit in logits]
        peak = max(scaled)
        log_norm = peak + math.log(sum(math.exp(value - peak) for value in scaled))
        total += log_norm - scaled[label]
    return total / len(labels)


def fit_temperature(
    logit_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
) -> float:
    """The temperature in ``[t_min, t_max]`` that minimizes mean NLL."""
    if not 0.0 < t_min < t_max:
        raise ValueError(f"need 0 < t_min < t_max, got t_min={t_min}, t_max={t_max}")
    _check_rows(logit_rows, labels)

    def nll_at_beta(beta: float) -> float:
        return mean_nll(logit_rows, labels, temperature=1.0 / beta)

    lo, hi = 1.0 / t_max, 1.0 / t_min
    a = hi - _INV_PHI * (hi - lo)
    b = lo + _INV_PHI * (hi - lo)
    fa, fb = nll_at_beta(a), nll_at_beta(b)
    for _ in range(_GOLDEN_SECTION_ITERS):
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - _INV_PHI * (hi - lo)
            fa = nll_at_beta(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + _INV_PHI * (hi - lo)
            fb = nll_at_beta(b)
    return 1.0 / ((lo + hi) / 2.0)


def calibration_metadata(
    logit_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    question_types: Sequence[str],
) -> SamplerCheckpointMetadata:
    """One temperature per question type, as the metadata of the final sampler save."""
    temperature: dict[str, float] = {}
    fit_rows: dict[str, int] = {}
    metrics: dict[str, dict[str, float]] = {}
    for question_type in sorted(set(question_types)):
        idx = [i for i, t in enumerate(question_types) if t == question_type]
        rows, gold = [logit_rows[i] for i in idx], [labels[i] for i in idx]
        t = fit_temperature(rows, gold)
        temperature[question_type] = t
        fit_rows[question_type] = len(idx)
        metrics[question_type] = {"nll_t1": mean_nll(rows, gold), "nll_fit": mean_nll(rows, gold, temperature=t)}
    return SamplerCheckpointMetadata(
        decision_calibration=DecisionCalibration(temperature=temperature, fit_rows=fit_rows, metrics=metrics)
    )


def _check_rows(logit_rows: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
    if not labels:
        raise ValueError("need at least one labeled row")
    if len(logit_rows) != len(labels):
        raise ValueError(f"{len(logit_rows)} logit rows for {len(labels)} labels")
    for logits, label in zip(logit_rows, labels, strict=True):
        if not 0 <= label < len(logits):
            raise ValueError(f"label {label} out of range for {len(logits)} options")
