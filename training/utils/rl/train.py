"""Compatibility exports for the shared batched training loop.

New recipe code should import from :mod:`training.train_loop`.  This
module remains to keep existing RL imports and tests stable.
"""

from training.train_loop import (
    DynamicFilterFn,
    TrainStepFns,
    raw_rows_from_stats,
    run_batched_training_loop,
)

__all__ = [
    "TrainStepFns",
    "DynamicFilterFn",
    "run_rl_loop",
    "raw_rows_from_stats",
]

run_rl_loop = run_batched_training_loop
