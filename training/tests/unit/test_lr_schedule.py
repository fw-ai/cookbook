"""Unit tests for training.utils.lr_schedule."""

from __future__ import annotations

import math

import pytest

from training.utils.lr_schedule import build_lr_per_step, resolve_step_lr


# ---------------------------------------------------------------------------
# build_lr_per_step
# ---------------------------------------------------------------------------


class TestBuildLrPerStep:
    """Tests for the schedule builder."""

    def test_constant_schedule(self) -> None:
        """Constant schedule returns peak_lr for every step."""
        lr_list = build_lr_per_step(total_steps=5, peak_lr=1e-4, schedule="constant")
        assert len(lr_list) == 5
        assert all(lr == pytest.approx(1e-4) for lr in lr_list)

    def test_constant_with_warmup(self) -> None:
        """Constant + warmup: linear ramp then flat."""
        lr_list = build_lr_per_step(
            total_steps=10, peak_lr=1e-4, schedule="constant", warmup_steps=4,
        )
        assert len(lr_list) == 10
        # Warmup should start at 0 and increase
        assert lr_list[0] == pytest.approx(0.0)
        assert lr_list[1] < lr_list[2] < lr_list[3]
        # After warmup, all values should be peak_lr
        for lr in lr_list[4:]:
            assert lr == pytest.approx(1e-4)

    def test_cosine_schedule_endpoints(self) -> None:
        """Cosine schedule starts at peak and ends near min_lr."""
        lr_list = build_lr_per_step(
            total_steps=100, peak_lr=1e-3, schedule="cosine", min_lr_ratio=0.1,
        )
        assert len(lr_list) == 100
        assert lr_list[0] == pytest.approx(1e-3)
        assert lr_list[-1] == pytest.approx(1e-4, rel=0.05)

    def test_cosine_schedule_is_monotonically_decreasing(self) -> None:
        """Cosine decay without warmup should be monotonically non-increasing."""
        lr_list = build_lr_per_step(
            total_steps=50, peak_lr=1e-3, schedule="cosine", min_lr_ratio=0.0,
        )
        for i in range(1, len(lr_list)):
            assert lr_list[i] <= lr_list[i - 1] + 1e-15

    def test_linear_schedule_endpoints(self) -> None:
        """Linear schedule starts at peak and ends near min_lr."""
        lr_list = build_lr_per_step(
            total_steps=100, peak_lr=1e-3, schedule="linear", min_lr_ratio=0.0,
        )
        assert lr_list[0] == pytest.approx(1e-3)
        # Last step (index 99) is decay_step=99 / decay_total=100 => 1e-5
        assert lr_list[-1] == pytest.approx(0.0, abs=2e-5)

    def test_linear_schedule_midpoint(self) -> None:
        """Linear decay midpoint should be exactly half of peak."""
        lr_list = build_lr_per_step(
            total_steps=100, peak_lr=1e-3, schedule="linear", min_lr_ratio=0.0,
        )
        assert lr_list[50] == pytest.approx(0.5e-3, rel=0.02)

    def test_warmup_ratio_vs_warmup_steps(self) -> None:
        """warmup_steps takes precedence over warmup_ratio."""
        lr_a = build_lr_per_step(
            total_steps=100, peak_lr=1e-4, schedule="cosine",
            warmup_steps=10, warmup_ratio=0.5,
        )
        lr_b = build_lr_per_step(
            total_steps=100, peak_lr=1e-4, schedule="cosine",
            warmup_steps=10, warmup_ratio=0.0,
        )
        assert lr_a == lr_b

    def test_warmup_ratio_converts_to_steps(self) -> None:
        """warmup_ratio is used when warmup_steps is 0."""
        lr_list = build_lr_per_step(
            total_steps=100, peak_lr=1e-4, schedule="constant",
            warmup_ratio=0.1,
        )
        # First 10 steps are warmup (10% of 100)
        assert lr_list[0] == pytest.approx(0.0)
        assert lr_list[9] < 1e-4
        assert lr_list[10] == pytest.approx(1e-4)

    def test_min_lr_ratio(self) -> None:
        """Warmup starts from min_lr, not from zero."""
        lr_list = build_lr_per_step(
            total_steps=100, peak_lr=1e-3, schedule="cosine",
            warmup_steps=10, min_lr_ratio=0.1,
        )
        assert lr_list[0] == pytest.approx(1e-4)

    def test_empty_for_zero_steps(self) -> None:
        """Returns empty list when total_steps <= 0."""
        assert build_lr_per_step(total_steps=0, peak_lr=1e-4) == []
        assert build_lr_per_step(total_steps=-1, peak_lr=1e-4) == []

    def test_single_step(self) -> None:
        """Single-step schedule returns just the peak LR."""
        lr_list = build_lr_per_step(total_steps=1, peak_lr=1e-4, schedule="cosine")
        assert len(lr_list) == 1
        assert lr_list[0] == pytest.approx(1e-4)

    def test_invalid_schedule_raises(self) -> None:
        """Unknown schedule raises ValueError."""
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            build_lr_per_step(total_steps=10, peak_lr=1e-4, schedule="exponential")

    def test_cosine_with_warmup(self) -> None:
        """Cosine with warmup: ramp up, then cosine decay."""
        lr_list = build_lr_per_step(
            total_steps=100, peak_lr=1e-3, schedule="cosine",
            warmup_steps=20, min_lr_ratio=0.0,
        )
        # Warmup phase: increasing
        for i in range(1, 20):
            assert lr_list[i] > lr_list[i - 1]
        # Peak at end of warmup
        assert lr_list[20] == pytest.approx(1e-3)
        # Decay phase: decreasing
        for i in range(21, 100):
            assert lr_list[i] <= lr_list[i - 1] + 1e-15


# ---------------------------------------------------------------------------
# resolve_step_lr
# ---------------------------------------------------------------------------


class TestResolveStepLr:
    """Tests for per-step LR resolution with fallback."""

    def test_none_falls_back_to_default(self) -> None:
        """When lr_per_step is None, returns default_lr."""
        assert resolve_step_lr(step=0, lr_per_step=None, default_lr=1e-4) == 1e-4
        assert resolve_step_lr(step=99, lr_per_step=None, default_lr=1e-4) == 1e-4

    def test_empty_list_falls_back_to_default(self) -> None:
        """When lr_per_step is an empty list, returns default_lr."""
        assert resolve_step_lr(step=0, lr_per_step=[], default_lr=1e-4) == 1e-4

    def test_indexes_into_list(self) -> None:
        """Normal case: index directly into the list."""
        schedule = [1e-6, 1e-5, 1e-4, 1e-3]
        assert resolve_step_lr(step=0, lr_per_step=schedule, default_lr=0.0) == 1e-6
        assert resolve_step_lr(step=2, lr_per_step=schedule, default_lr=0.0) == 1e-4

    def test_tail_fallback(self) -> None:
        """When step >= len(lr_per_step), clamps to the last value."""
        schedule = [1e-6, 1e-5, 1e-4]
        assert resolve_step_lr(step=3, lr_per_step=schedule, default_lr=0.0) == 1e-4
        assert resolve_step_lr(step=100, lr_per_step=schedule, default_lr=0.0) == 1e-4

    def test_single_element_list(self) -> None:
        """Single-element list returns that value for all steps."""
        schedule = [5e-5]
        assert resolve_step_lr(step=0, lr_per_step=schedule, default_lr=0.0) == 5e-5
        assert resolve_step_lr(step=99, lr_per_step=schedule, default_lr=0.0) == 5e-5
