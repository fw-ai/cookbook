"""Unit tests for OPD dynamic metrics (arXiv:2604.13016).

All tests use synthetic top-k logprob lists -- no model loading or GPU required.
"""

from __future__ import annotations

import math

import pytest

from training.utils.opd_metrics import (
    compute_entropy_gap,
    compute_opd_metrics,
    compute_overlap_mass,
    compute_overlap_ratio,
    compute_overlap_token_advantage,
    compute_per_position_entropy,
    compute_student_entropy,
    compute_teacher_entropy,
)


def _uniform_topk(token_ids: list[int]) -> list[tuple[int, float]]:
    """Build a top-k list with uniform logprobs over the given tokens."""
    lp = math.log(1.0 / len(token_ids))
    return [(tok, lp) for tok in token_ids]


def _peaked_topk(peak_id: int, other_ids: list[int], peak_prob: float = 0.9) -> list[tuple[int, float]]:
    """Build a top-k list where ``peak_id`` gets ``peak_prob`` and the rest share the remainder."""
    rest_prob = (1.0 - peak_prob) / max(len(other_ids), 1)
    return [(peak_id, math.log(peak_prob))] + [(tok, math.log(rest_prob)) for tok in other_ids]


# ---------------------------------------------------------------------------
# Overlap ratio (Eq. 6)
# ---------------------------------------------------------------------------


class TestOverlapRatio:
    def test_identical_topk_sets(self) -> None:
        """Identical top-k sets -> ratio = 1.0."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 5
        assert compute_overlap_ratio(topk, topk) == pytest.approx(1.0)

    def test_disjoint_topk_sets(self) -> None:
        """No shared tokens -> ratio = 0.0."""
        student = [_uniform_topk([1, 2, 3, 4])] * 5
        teacher = [_uniform_topk([5, 6, 7, 8])] * 5
        assert compute_overlap_ratio(student, teacher) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Known partial overlap -> verify exact fraction."""
        student = [_uniform_topk([1, 2, 3, 4])]
        teacher = [_uniform_topk([3, 4, 5, 6])]
        assert compute_overlap_ratio(student, teacher) == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        """Empty sequences -> 0.0."""
        assert compute_overlap_ratio([], []) == 0.0
        assert compute_overlap_ratio([], [_uniform_topk([1, 2])]) == 0.0


# ---------------------------------------------------------------------------
# Overlap-token advantage (Eq. 7)
# ---------------------------------------------------------------------------


class TestOverlapAdvantage:
    def test_identical_distributions(self) -> None:
        """Identical distributions -> advantage = 0 (same logprobs on overlap)."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 3
        assert compute_overlap_token_advantage(topk, topk) == pytest.approx(0.0, abs=1e-9)

    def test_teacher_and_student_mismatch_is_negative(self) -> None:
        """Eq. 7 is non-positive and gets more negative under mismatch."""
        student = [_uniform_topk([1, 2, 3, 4])]
        teacher = [_peaked_topk(1, [2, 3, 4], peak_prob=0.9)]
        adv = compute_overlap_token_advantage(student, teacher)
        assert adv < 0

    def test_student_overconfident_is_negative(self) -> None:
        """Student concentrates more mass on overlap tokens -> advantage < 0."""
        student = [_peaked_topk(1, [2, 3, 4], peak_prob=0.9)]
        teacher = [_uniform_topk([1, 2, 3, 4])]
        adv = compute_overlap_token_advantage(student, teacher)
        assert adv < 0

    def test_no_overlap_returns_zero(self) -> None:
        """No overlapping tokens -> 0.0 (degenerate case)."""
        student = [_uniform_topk([1, 2])]
        teacher = [_uniform_topk([3, 4])]
        assert compute_overlap_token_advantage(student, teacher) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Entropy gap (Eq. 8)
# ---------------------------------------------------------------------------


class TestEntropyGap:
    def test_same_distributions_zero_gap(self) -> None:
        """Same distributions -> gap = 0."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 3
        assert compute_entropy_gap(topk, topk) == pytest.approx(0.0, abs=1e-9)

    def test_different_distributions_nonzero_gap(self) -> None:
        """Different distributions -> gap > 0."""
        student = [_peaked_topk(1, [2, 3, 4], peak_prob=0.97)]
        teacher = [_uniform_topk([1, 2, 3, 4])]
        gap = compute_entropy_gap(student, teacher)
        assert gap > 0

    def test_gap_is_absolute_difference(self) -> None:
        """Gap is symmetric: |H(t) - H(s)| = |H(s) - H(t)|."""
        student = [_peaked_topk(1, [2, 3], peak_prob=0.9)]
        teacher = [_uniform_topk([1, 2, 3])]
        assert compute_entropy_gap(student, teacher) == pytest.approx(
            compute_entropy_gap(teacher, student)
        )


# ---------------------------------------------------------------------------
# Overlap mass (Eqs. 9-10)
# ---------------------------------------------------------------------------


class TestOverlapMass:
    def test_high_prob_on_overlap(self) -> None:
        """When top-k sets are identical, overlap mass should be ~1.0."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 4
        s_mass, t_mass = compute_overlap_mass(topk, topk)
        assert s_mass == pytest.approx(1.0, abs=1e-9)
        assert t_mass == pytest.approx(1.0, abs=1e-9)

    def test_disjoint_zero_mass(self) -> None:
        """Disjoint sets -> overlap mass = 0."""
        student = [_uniform_topk([1, 2])]
        teacher = [_uniform_topk([3, 4])]
        s_mass, t_mass = compute_overlap_mass(student, teacher)
        assert s_mass == pytest.approx(0.0, abs=1e-9)
        assert t_mass == pytest.approx(0.0, abs=1e-9)

    def test_partial_overlap_mass(self) -> None:
        """Partial overlap: mass equals original probability on shared tokens."""
        student = [_uniform_topk([1, 2, 3, 4])]
        teacher = [_uniform_topk([3, 4, 5, 6])]
        s_mass, t_mass = compute_overlap_mass(student, teacher)
        assert s_mass == pytest.approx(0.5, abs=1e-6)
        assert t_mass == pytest.approx(0.5, abs=1e-6)

    def test_overlap_mass_does_not_renormalize_topk(self) -> None:
        """Top-k APIs return full-vocab logprobs, so mass can be below top-k-normalized values."""
        student = [[(1, math.log(0.2)), (2, math.log(0.1)), (3, math.log(0.05))]]
        teacher = [[(2, math.log(0.3)), (4, math.log(0.2)), (5, math.log(0.1))]]
        s_mass, t_mass = compute_overlap_mass(student, teacher)
        assert s_mass == pytest.approx(0.1, abs=1e-9)
        assert t_mass == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# Per-position entropy (Section 6.1)
# ---------------------------------------------------------------------------


class TestPerPositionEntropy:
    def test_bucket_keys(self) -> None:
        """Verify bucketing produces expected keys."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 8
        result = compute_per_position_entropy(topk, bucket_count=4)
        assert set(result.keys()) == {
            "per_position_entropy/q1",
            "per_position_entropy/q2",
            "per_position_entropy/q3",
            "per_position_entropy/q4",
        }

    def test_uniform_entropy_across_positions(self) -> None:
        """Uniform distribution at every position -> all buckets equal."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 8
        result = compute_per_position_entropy(topk, bucket_count=4)
        expected = math.log(4)
        for val in result.values():
            assert val == pytest.approx(expected, abs=1e-6)

    def test_empty_returns_zeros(self) -> None:
        """Empty input -> all buckets 0.0."""
        result = compute_per_position_entropy([], bucket_count=4)
        for val in result.values():
            assert val == 0.0

    def test_custom_bucket_count(self) -> None:
        """Custom bucket_count=2 produces q1 and q2."""
        topk = [_uniform_topk([1, 2])] * 4
        result = compute_per_position_entropy(topk, bucket_count=2)
        assert set(result.keys()) == {
            "per_position_entropy/q1",
            "per_position_entropy/q2",
        }


# ---------------------------------------------------------------------------
# Student / teacher entropy
# ---------------------------------------------------------------------------


class TestIndividualEntropy:
    def test_uniform_entropy(self) -> None:
        """Uniform distribution -> entropy = log(k)."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 3
        expected = math.log(4)
        assert compute_student_entropy(topk) == pytest.approx(expected, abs=1e-6)
        assert compute_teacher_entropy(topk) == pytest.approx(expected, abs=1e-6)

    def test_peaked_low_entropy(self) -> None:
        """Peaked distribution -> entropy < uniform."""
        peaked = [_peaked_topk(1, [2, 3, 4], peak_prob=0.97)] * 3
        uniform = [_uniform_topk([1, 2, 3, 4])] * 3
        assert compute_student_entropy(peaked) < compute_student_entropy(uniform)


# ---------------------------------------------------------------------------
# All-in-one compute_opd_metrics
# ---------------------------------------------------------------------------


class TestComputeOpdMetrics:
    def test_all_keys_present(self) -> None:
        """Verify all-in-one returns complete metric dict."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 4
        metrics = compute_opd_metrics(topk, topk)
        expected_keys = {
            "distill/overlap_ratio",
            "distill/overlap_advantage",
            "distill/entropy_gap",
            "distill/overlap_mass_student",
            "distill/overlap_mass_teacher",
            "distill/student_entropy",
            "distill/teacher_entropy",
            "distill/per_position_entropy/q1",
            "distill/per_position_entropy/q2",
            "distill/per_position_entropy/q3",
            "distill/per_position_entropy/q4",
        }
        assert set(metrics.keys()) == expected_keys

    def test_identical_distributions_healthy_signature(self) -> None:
        """Identical distributions -> healthy signature values."""
        topk = [_uniform_topk([1, 2, 3, 4])] * 4
        metrics = compute_opd_metrics(topk, topk)
        assert metrics["distill/overlap_ratio"] == pytest.approx(1.0)
        assert metrics["distill/overlap_advantage"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["distill/entropy_gap"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["distill/overlap_mass_student"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["distill/overlap_mass_teacher"] == pytest.approx(1.0, abs=1e-9)

    def test_different_distributions_nonzero_advantage(self) -> None:
        """Different distributions -> advantage is negative."""
        student = [_uniform_topk([1, 2, 3, 4])] * 4
        teacher = [_peaked_topk(1, [2, 3, 4], peak_prob=0.9)] * 4
        metrics = compute_opd_metrics(student, teacher)
        assert metrics["distill/overlap_advantage"] < 0

    def test_custom_bucket_count(self) -> None:
        """Custom bucket_count propagates to per-position keys."""
        topk = [_uniform_topk([1, 2])] * 4
        metrics = compute_opd_metrics(topk, topk, bucket_count=2)
        assert "distill/per_position_entropy/q1" in metrics
        assert "distill/per_position_entropy/q2" in metrics
        assert "distill/per_position_entropy/q3" not in metrics
