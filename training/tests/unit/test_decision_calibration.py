"""Temperature fitting for K-way decision probabilities and the calibration save metadata."""

from __future__ import annotations

import math
import random

import pytest

from training.utils.decision_calibration import calibration_metadata, fit_temperature, mean_nll


def _sample(n: int, k: int, true_t: float, seed: int) -> tuple[list[list[float]], list[int]]:
    """Logits whose labels are drawn from softmax(logits / true_t)."""
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        logits = [rng.gauss(0.0, 3.0) for _ in range(k)]
        peak = max(logits)
        weights = [math.exp((x - peak) / true_t) for x in logits]
        labels.append(rng.choices(range(k), weights=weights)[0])
        rows.append(logits)
    return rows, labels


def test_calibration_metadata_recovers_one_temperature_per_question_type() -> None:
    noul_rows, noul_labels = _sample(800, 2, true_t=2.0, seed=4)
    choice_rows, choice_labels = _sample(600, 4, true_t=0.5, seed=5)
    metadata = calibration_metadata(
        noul_rows + choice_rows, noul_labels + choice_labels, ["noul"] * 800 + ["choice"] * 600
    )
    calibration = metadata.decision_calibration
    assert calibration.contract == "systemone-letter-v1"
    assert calibration.fit_rows == {"choice": 600, "noul": 800}
    assert calibration.temperature["noul"] == pytest.approx(2.0, rel=0.15)
    assert calibration.temperature["choice"] == pytest.approx(0.5, rel=0.15)
    for question_type in ("noul", "choice"):
        assert calibration.metrics[question_type]["nll_fit"] <= calibration.metrics[question_type]["nll_t1"]


def test_fit_temperature_bounds_and_row_checks() -> None:
    rows, labels = [[5.0, 0.0]] * 50, [0] * 50
    assert fit_temperature(rows, labels, t_min=0.5, t_max=3.0) == pytest.approx(0.5, rel=1e-3)
    assert mean_nll([[10.0, -10.0]], [0]) == pytest.approx(math.log1p(math.exp(-20.0)))
    for call, match in [
        (lambda: fit_temperature(rows, labels, t_min=2.0, t_max=1.0), "t_min < t_max"),
        (lambda: mean_nll([], []), "at least one"),
        (lambda: mean_nll([[0.0, 1.0]], [2]), "out of range"),
    ]:
        with pytest.raises(ValueError, match=match):
            call()
