#!/usr/bin/env python3
# ruff: noqa: E402
"""Reconnect to a running RLOR trainer and continue training with a new learning rate.

Demonstrates two key capabilities:
  1. Reconnecting to an already-running trainer job (no restart needed)
  2. Changing the learning rate mid-training by passing new AdamParams per step

The tinker protocol accepts AdamParams on every optim_step() call, so the
learning rate is never baked into the trainer — you can vary it freely
without restarting or recreating the job.

Usage:
    export FIREWORKS_API_KEY=...

    # Reconnect and continue with a new constant LR:
    python reconnect_and_adjust_lr.py \
        --job-id <policy-job-id> \
        --base-model accounts/fireworks/models/qwen3-8b \
        --new-lr 5e-6

    # Reconnect and use cosine decay from the current step:
    python reconnect_and_adjust_lr.py \
        --job-id <policy-job-id> \
        --base-model accounts/fireworks/models/qwen3-8b \
        --new-lr 1e-5 \
        --lr-schedule cosine \
        --lr-min 1e-6 \
        --total-steps 100

    # Also reconnect a reference trainer:
    python reconnect_and_adjust_lr.py \
        --job-id <policy-job-id> \
        --ref-job-id <ref-job-id> \
        --base-model accounts/fireworks/models/qwen3-8b \
        --new-lr 5e-6
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

import tinker
from fireworks.training.sdk import TrainerJobManager
from training.utils import DEFAULT_ADAM, ReconnectableClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------


def constant_lr(base_lr: float, **_kwargs) -> float:
    """Return a fixed learning rate regardless of step."""
    return base_lr


def linear_warmup_lr(
    base_lr: float, step: int, warmup_steps: int = 10, **_kwargs
) -> float:
    """Linear warmup from 0 to base_lr over warmup_steps, then constant."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def cosine_decay_lr(
    base_lr: float,
    step: int,
    total_steps: int,
    lr_min: float = 0.0,
    warmup_steps: int = 0,
    **_kwargs,
) -> float:
    """Cosine decay from base_lr to lr_min, with optional linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))


LR_SCHEDULES = {
    "constant": constant_lr,
    "linear_warmup": linear_warmup_lr,
    "cosine": cosine_decay_lr,
}


def get_lr(schedule_name: str, **kwargs) -> float:
    """Compute learning rate for the current step using the named schedule."""
    fn = LR_SCHEDULES[schedule_name]
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Reconnect to a running trainer and adjust the learning rate."
    )
    parser.add_argument("--job-id", required=True, help="Policy RLOR trainer job ID")
    parser.add_argument("--ref-job-id", default=None, help="Reference RLOR trainer job ID (optional)")
    parser.add_argument("--base-model", required=True, help="Base model resource name")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--base-url", default=None, help="Fireworks API base URL")

    # LR config
    parser.add_argument("--new-lr", type=float, required=True, help="New learning rate")
    parser.add_argument(
        "--lr-schedule",
        choices=list(LR_SCHEDULES.keys()),
        default="constant",
        help="LR schedule to use (default: constant)",
    )
    parser.add_argument("--lr-min", type=float, default=0.0, help="Minimum LR for cosine schedule")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for linear/cosine schedules")
    parser.add_argument("--total-steps", type=int, default=100, help="Total steps for cosine schedule")

    # Training config
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to run (demo)")
    parser.add_argument("--start-step", type=int, default=0, help="Step offset (for LR schedule alignment)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = args.base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    # -- Reconnect to existing trainer(s) ----------------------------------

    logger.info("Reconnecting to policy trainer %s ...", args.job_id)
    policy_ep = mgr.reconnect_and_wait(args.job_id, timeout_s=300)
    logger.info("Policy connected: %s", policy_ep.base_url)

    policy = ReconnectableClient(
        mgr,
        policy_ep.job_id,
        args.base_model,
        args.lora_rank,
        fw_api_key=api_key,
        endpoint=policy_ep,
    )

    reference = None
    if args.ref_job_id:
        logger.info("Reconnecting to reference trainer %s ...", args.ref_job_id)
        ref_ep = mgr.reconnect_and_wait(args.ref_job_id, timeout_s=300)
        logger.info("Reference connected: %s", ref_ep.base_url)
        reference = ReconnectableClient(
            mgr,
            ref_ep.job_id,
            args.base_model,
            args.lora_rank,
            fw_api_key=api_key,
            endpoint=ref_ep,
        )

    # -- Demonstrate dynamic LR per step -----------------------------------

    logger.info(
        "Schedule: %s | base_lr=%.2e | lr_min=%.2e | warmup=%d | total=%d",
        args.lr_schedule,
        args.new_lr,
        args.lr_min,
        args.warmup_steps,
        args.total_steps,
    )

    for step_idx in range(args.steps):
        step = args.start_step + step_idx

        lr = get_lr(
            args.lr_schedule,
            base_lr=args.new_lr,
            step=step,
            total_steps=args.total_steps,
            lr_min=args.lr_min,
            warmup_steps=args.warmup_steps,
        )
        adam_params = tinker.AdamParams(learning_rate=lr, **DEFAULT_ADAM)

        logger.info("[step %d] lr=%.2e", step, lr)

        # ---------------------------------------------------------------
        # In a real training loop you would:
        #   1. Sample completions from the deployment
        #   2. Compute rewards and advantages
        #   3. (Optional) reference forward pass
        #   4. policy.forward_backward_custom(data, loss_fn)
        #   5. policy.optim_step(adam_params)   <-- LR applied here
        #
        # The key insight: adam_params is passed per-step, so the trainer
        # uses whatever LR you give it. No restart needed.
        # ---------------------------------------------------------------

        # Uncomment to actually run a training step:
        # fwd_bwd_result = policy.forward_backward_custom(datums, loss_fn)
        # optim_result = policy.optim_step(adam_params)
        # logger.info("[step %d] optim metrics: %s", step, optim_result.metrics)

    logger.info("Done. Policy trainer %s is still running (not deleted).", args.job_id)
    if reference:
        logger.info("Reference trainer %s is still running.", args.ref_job_id)


if __name__ == "__main__":
    main()
