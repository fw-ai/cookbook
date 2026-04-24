"""On-policy distillation (OPD) dynamic metrics from arXiv:2604.13016.

Pure functions that take top-k logprobs from student and teacher, aligned
position-by-position, and return all paper metrics (Eqs. 6-10, Section 6.1).

Input format per position: ``List[Tuple[int, float]]`` where each tuple is
``(token_id, logprob)``.  Sequences are ``List[List[Tuple[int, float]]]``
(one inner list per position in the response).

All entropy/mass computations use **renormalized** top-k probabilities, matching
the paper's k=16 setting (Section 2.3, Appendix B.1).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

TopKLogprobs = List[Tuple[int, float]]
PositionLogprobs = List[TopKLogprobs]

DEFAULT_TOP_K = 16


def _to_prob_dict(topk: TopKLogprobs) -> dict[int, float]:
    """Convert top-k logprobs to a dict of token_id -> probability."""
    return {tok: math.exp(lp) for tok, lp in topk}


def _renormalize(prob_dict: dict[int, float]) -> dict[int, float]:
    """Renormalize probabilities to sum to 1."""
    total = sum(prob_dict.values())
    if total <= 0:
        return prob_dict
    return {tok: p / total for tok, p in prob_dict.items()}


def _entropy_from_probs(probs: dict[int, float]) -> float:
    """Shannon entropy from a (renormalized) probability dict."""
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


def compute_overlap_ratio(
    student_topk: PositionLogprobs,
    teacher_topk: PositionLogprobs,
) -> float:
    """Eq. 6: fraction of top-k tokens shared, averaged over positions.

    ``|S_student_topk ∩ S_teacher_topk| / k``, averaged across positions.
    """
    if not student_topk or not teacher_topk:
        return 0.0
    n = min(len(student_topk), len(teacher_topk))
    total = 0.0
    for i in range(n):
        s_ids = {tok for tok, _ in student_topk[i]}
        t_ids = {tok for tok, _ in teacher_topk[i]}
        k = max(len(s_ids), len(t_ids), 1)
        total += len(s_ids & t_ids) / k
    return total / n


def compute_overlap_token_advantage(
    student_topk: PositionLogprobs,
    teacher_topk: PositionLogprobs,
) -> float:
    """Eq. 7: distributional agreement within shared tokens.

    For each position, renormalize student and teacher probs onto the
    intersection token set, then compute
    ``sum_over_overlap(teacher_renorm - student_renorm) / |overlap|``.
    Approaching 0 = good alignment.
    """
    if not student_topk or not teacher_topk:
        return 0.0
    n = min(len(student_topk), len(teacher_topk))
    total = 0.0
    valid_positions = 0
    for i in range(n):
        s_probs = _to_prob_dict(student_topk[i])
        t_probs = _to_prob_dict(teacher_topk[i])
        overlap = set(s_probs.keys()) & set(t_probs.keys())
        if not overlap:
            continue
        s_overlap = {tok: s_probs[tok] for tok in overlap}
        t_overlap = {tok: t_probs[tok] for tok in overlap}
        s_renorm = _renormalize(s_overlap)
        t_renorm = _renormalize(t_overlap)
        adv = sum(t_renorm[tok] - s_renorm[tok] for tok in overlap) / len(overlap)
        total += adv
        valid_positions += 1
    return total / max(valid_positions, 1)


def compute_entropy_gap(
    student_topk: PositionLogprobs,
    teacher_topk: PositionLogprobs,
) -> float:
    """Eq. 8: ``|H(teacher) - H(student)|`` averaged over positions.

    Entropy is computed from renormalized top-k probabilities.
    """
    if not student_topk or not teacher_topk:
        return 0.0
    n = min(len(student_topk), len(teacher_topk))
    total = 0.0
    for i in range(n):
        s_probs = _renormalize(_to_prob_dict(student_topk[i]))
        t_probs = _renormalize(_to_prob_dict(teacher_topk[i]))
        s_ent = _entropy_from_probs(s_probs)
        t_ent = _entropy_from_probs(t_probs)
        total += abs(t_ent - s_ent)
    return total / n


def compute_overlap_mass(
    student_topk: PositionLogprobs,
    teacher_topk: PositionLogprobs,
) -> Tuple[float, float]:
    """Eqs. 9-10: probability mass on shared top-k tokens.

    Returns ``(student_overlap_mass, teacher_overlap_mass)`` averaged over
    positions.  In healthy runs both should be 97-99%.
    """
    if not student_topk or not teacher_topk:
        return (0.0, 0.0)
    n = min(len(student_topk), len(teacher_topk))
    s_mass_total = 0.0
    t_mass_total = 0.0
    for i in range(n):
        s_probs = _renormalize(_to_prob_dict(student_topk[i]))
        t_probs = _renormalize(_to_prob_dict(teacher_topk[i]))
        overlap = set(s_probs.keys()) & set(t_probs.keys())
        s_mass_total += sum(s_probs[tok] for tok in overlap)
        t_mass_total += sum(t_probs[tok] for tok in overlap)
    return (s_mass_total / n, t_mass_total / n)


def compute_per_position_entropy(
    student_topk: PositionLogprobs,
    *,
    bucket_count: int = 4,
) -> Dict[str, float]:
    """Section 6.1: mean student entropy per position quartile.

    Splits positions into ``bucket_count`` equal-sized buckets and reports
    the mean renormalized-top-k entropy per bucket.  Detects depth
    degradation (instability starts at later tokens).
    """
    if not student_topk:
        return {f"per_position_entropy/q{i + 1}": 0.0 for i in range(bucket_count)}
    n = len(student_topk)
    entropies = []
    for pos_topk in student_topk:
        probs = _renormalize(_to_prob_dict(pos_topk))
        entropies.append(_entropy_from_probs(probs))

    bucket_size = max(1, n // bucket_count)
    result: Dict[str, float] = {}
    for b in range(bucket_count):
        start = b * bucket_size
        end = start + bucket_size if b < bucket_count - 1 else n
        bucket_vals = entropies[start:end]
        mean_ent = sum(bucket_vals) / max(len(bucket_vals), 1)
        result[f"per_position_entropy/q{b + 1}"] = mean_ent
    return result


def compute_student_entropy(student_topk: PositionLogprobs) -> float:
    """Mean student policy entropy from renormalized top-k probs."""
    if not student_topk:
        return 0.0
    total = 0.0
    for pos_topk in student_topk:
        probs = _renormalize(_to_prob_dict(pos_topk))
        total += _entropy_from_probs(probs)
    return total / len(student_topk)


def compute_teacher_entropy(teacher_topk: PositionLogprobs) -> float:
    """Mean teacher policy entropy from renormalized top-k probs."""
    if not teacher_topk:
        return 0.0
    total = 0.0
    for pos_topk in teacher_topk:
        probs = _renormalize(_to_prob_dict(pos_topk))
        total += _entropy_from_probs(probs)
    return total / len(teacher_topk)


def compute_opd_metrics(
    student_topk: PositionLogprobs,
    teacher_topk: PositionLogprobs,
    *,
    bucket_count: int = 4,
) -> Dict[str, float]:
    """All-in-one: compute all paper metrics and return a flat dict.

    Keys are prefixed with ``distill/`` for easy integration with wandb/JSONL.
    """
    overlap_ratio = compute_overlap_ratio(student_topk, teacher_topk)
    overlap_advantage = compute_overlap_token_advantage(student_topk, teacher_topk)
    entropy_gap = compute_entropy_gap(student_topk, teacher_topk)
    s_mass, t_mass = compute_overlap_mass(student_topk, teacher_topk)
    per_pos = compute_per_position_entropy(student_topk, bucket_count=bucket_count)
    s_entropy = compute_student_entropy(student_topk)
    t_entropy = compute_teacher_entropy(teacher_topk)

    metrics: Dict[str, float] = {
        "distill/overlap_ratio": overlap_ratio,
        "distill/overlap_advantage": overlap_advantage,
        "distill/entropy_gap": entropy_gap,
        "distill/overlap_mass_student": s_mass,
        "distill/overlap_mass_teacher": t_mass,
        "distill/student_entropy": s_entropy,
        "distill/teacher_entropy": t_entropy,
    }
    for key, val in per_pos.items():
        metrics[f"distill/{key}"] = val
    return metrics
