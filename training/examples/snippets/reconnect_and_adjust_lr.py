#!/usr/bin/env python3
# ruff: noqa: E402
"""Reconnect to a running RLOR trainer and adjust the learning rate.

**Reference example only** — this script demonstrates the reconnect and
per-step LR APIs but does NOT run real training (no data, no loss function,
no forward/backward).  Use it as a starting point; plug in your own data
pipeline and loss to make it functional.

What this shows:
  1. Reconnecting to an already-running trainer job (no restart needed).
  2. Changing the learning rate mid-training by passing new AdamParams
     on every optim_step() call.  The tinker protocol accepts AdamParams
     per step, so the LR is never baked into the trainer.

Usage:
    export FIREWORKS_API_KEY=...

    python reconnect_and_adjust_lr.py \
        --job-id <policy-job-id> \
        --base-model accounts/fireworks/models/qwen3-8b \
        --new-lr 5e-6

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


def main():
    parser = argparse.ArgumentParser(
        description="Reconnect to a running trainer and adjust the learning rate."
    )
    parser.add_argument("--job-id", required=True, help="Policy RLOR trainer job ID")
    parser.add_argument("--ref-job-id", default=None, help="Reference RLOR trainer job ID (optional)")
    parser.add_argument("--base-model", required=True, help="Base model resource name")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--base-url", default=None, help="Fireworks API base URL")
    parser.add_argument("--new-lr", type=float, required=True, help="New learning rate")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to run (demo)")

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

    adam_params = tinker.AdamParams(learning_rate=args.new_lr, **DEFAULT_ADAM)
    logger.info("Using LR=%.2e for %d steps", args.new_lr, args.steps)

    for step_idx in range(args.steps):
        logger.info("[step %d] lr=%.2e", step_idx, args.new_lr)

        # -----------------------------------------------------------------
        # In a real training loop you would:
        #   1. Sample completions from the deployment
        #   2. Compute rewards and advantages
        #   3. (Optional) reference forward pass
        #   4. policy.forward_backward_custom(data, loss_fn)
        #   5. policy.optim_step(adam_params)   <-- LR applied here
        #
        # The key insight: adam_params is passed per-step, so the trainer
        # uses whatever LR you give it. No restart needed.
        # -----------------------------------------------------------------

        # Uncomment to actually run a training step:
        # fwd_bwd_result = policy.forward_backward_custom(datums, loss_fn)
        # optim_result = policy.optim_step(adam_params)
        # logger.info("[step %d] optim metrics: %s", step_idx, optim_result.metrics)

    logger.info("Done. Policy trainer %s is still running (not deleted).", args.job_id)
    if reference:
        logger.info("Reference trainer %s is still running.", args.ref_job_id)


if __name__ == "__main__":
    main()
