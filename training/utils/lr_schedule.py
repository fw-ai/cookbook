"""Pre-compute a per-step learning rate schedule.

Provides ``build_lr_per_step`` -- a convenience function that generates
a complete LR schedule as a ``list[float]``.  Recipes use this list to
set a different LR at each optimizer step.  The function is also
available to customers who want a standard schedule without computing
it themselves.

The managed service calls ``build_lr_per_step`` from the orchestrator
when it knows dataset size; customers can call it directly or provide
their own list.
"""

from __future__ import annotations

import math

__all__ = [
    "build_lr_per_step",
    "resolve_step_lr",
]


def build_lr_per_step(
    total_steps: int,
    peak_lr: float,
    schedule: str = "constant",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.0,
) -> list[float]:
    """Pre-compute the full LR schedule as a list of per-step values.

    Args:
        total_steps: Total number of optimizer steps.
        peak_lr: Peak (maximum) learning rate.
        schedule: Decay schedule after warmup -- ``"constant"``,
            ``"cosine"``, or ``"linear"``.
        warmup_steps: Number of linear warmup steps.  Takes precedence
            over *warmup_ratio* when both are non-zero.
        warmup_ratio: Fraction of *total_steps* used for warmup.
            Ignored when *warmup_steps* > 0.
        min_lr_ratio: Minimum LR as a fraction of *peak_lr*.
            Cosine/linear decay anneals to ``peak_lr * min_lr_ratio``.

    Returns:
        A list of length *total_steps* where ``result[i]`` is the LR
        for optimizer step *i*.

    Raises:
        ValueError: If *schedule* is not one of the supported values.
    """
    if total_steps <= 0:
        return []

    effective_warmup = warmup_steps if warmup_steps > 0 else int(total_steps * warmup_ratio)
    min_lr = peak_lr * min_lr_ratio

    schedule_values: list[float] = []
    for step in range(total_steps):
        schedule_values.append(
            _lr_at_step(step, total_steps, peak_lr, min_lr, effective_warmup, schedule)
        )
    return schedule_values


def resolve_step_lr(
    step: int,
    lr_per_step: list[float] | None,
    default_lr: float,
) -> float:
    """Return the learning rate for a given optimizer step.

    If *lr_per_step* is provided, index into it (clamping to the last
    value if *step* exceeds the list length).  Otherwise fall back to
    *default_lr*.
    """
    if lr_per_step is None:
        return default_lr
    if not lr_per_step:
        return default_lr
    idx = min(step, len(lr_per_step) - 1)
    return lr_per_step[idx]


def _lr_at_step(
    step: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Compute LR for a single step (linear warmup then decay)."""
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * step / max(warmup_steps, 1)

    if schedule == "constant":
        return peak_lr

    decay_step = step - warmup_steps
    decay_total = max(total_steps - warmup_steps, 1)

    if schedule == "cosine":
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * decay_step / decay_total))
    if schedule == "linear":
        return peak_lr - (peak_lr - min_lr) * decay_step / decay_total

    raise ValueError(f"Unknown lr_schedule: {schedule!r}. Use 'constant', 'cosine', or 'linear'.")
