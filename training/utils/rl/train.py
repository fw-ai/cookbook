"""Pure helpers for RL training-side bookkeeping.

This module deliberately exposes only one thing:
:func:`dump_trajectory_jsonl`. The previous ``run_rl_loop`` runner and
the ``ref_fwd_bwd`` / ``finish_step`` callback registry have been
deleted — recipes now write their own loop body using direct SDK calls
(``policy.forward``, ``policy.forward_backward_custom``,
``policy.optim_step``, ``weight_syncer.save_and_hotload``).

If you want a reference loop, copy the body out of
``recipes/rl_loop.py`` or ``recipes/async_rl_loop.py``.
"""

from __future__ import annotations

import json
import logging
import os

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = ["dump_trajectory_jsonl"]


def dump_trajectory_jsonl(
    trajectory_dir: str,
    step: int,
    prompt_groups: list[PromptGroup],
) -> None:
    """Write per-step trajectory JSONL: one line per individual completion."""
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"step_{step:04d}.jsonl")
    n_records = 0
    with open(path, "w") as f:
        for pg_idx, pg in enumerate(prompt_groups):
            completions = pg.completions or []
            for comp_idx, comp_text in enumerate(completions):
                record = {
                    "step": step,
                    "prompt_group": pg_idx,
                    "completion_index": comp_idx,
                    "prompt": pg.prompt,
                    "completion": comp_text,
                    "reward": pg.rewards[comp_idx] if comp_idx < len(pg.rewards) else None,
                    "advantage": (
                        pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None
                    ),
                    "completion_len": (
                        pg.completion_lens[comp_idx]
                        if comp_idx < len(pg.completion_lens)
                        else None
                    ),
                    "truncated": (
                        pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None
                    ),
                    "ground_truth": pg.row_meta.get("ground_truth") if pg.row_meta else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1
    logger.info(
        "[step %d] Saved trajectory to %s (%d completions from %d groups)",
        step, path, n_records, len(prompt_groups),
    )
