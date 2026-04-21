#!/usr/bin/env python3
"""Manual test: PER_DEPLOYMENT (deployment-first) flow end-to-end.

Drives ``rl_loop.main`` with ``DeployConfig.weight_sync_scope =
WeightSyncScope.PER_DEPLOYMENT``. Under this scope:

1. ``setup_infra`` provisions the deployment **first**; its ID is the
   stable bucket owner.
2. Each trainer is then launched with ``hot_load_deployment_id``
   pointing at that deployment — their buckets co-index with the
   deployment, not with the trainer job.
3. The recipe's WeightSyncer hotloads into the deployment normally.

Minimal run: 1 prompt group per step, 2 completions, 5 rows, 1 epoch
on qwen3-4b + the 1xGPU qwen3-4b-minimum training shape. Complements
the trainer-first path that CI
https://github.com/fw-ai/fireworks/actions/runs/24703610604 already
covers.

Usage:
    export FIREWORKS_API_KEY=<pyroworks key>
    python training/examples/manual/test_deployment_first_manual.py

What to look for in the logs:
    Creating deployment: <dep_id>                  # happens BEFORE any trainer
    Creating policy trainer job '...' ...           # then trainers
    (no "Re-attached deployment ..." messages — this scope never PATCHes)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import uuid

from dotenv import load_dotenv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
import training.recipes.rl_loop as rl_loop
from training.utils import (
    DeployConfig,
    InfraConfig,
    WandBConfig,
    WeightSyncConfig,
    WeightSyncScope,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("manual.deployment_first")

_DEEPMATH_DATASET = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "rl", "deepmath", "dataset.jsonl",
    )
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-4b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--training-shape",
        default="accounts/pyroworks/trainingShapes/qwen3-4b-minimum-lora",
    )
    parser.add_argument(
        "--ref-training-shape",
        default=None,
        help="Separate reference shape; None (default) uses LoRA shared-session reference.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank; 0 for full-param. Default 64 matches the qwen3-4b-minimum-lora shape.",
    )
    parser.add_argument(
        "--deployment-id",
        default=None,
        help="Reuse a specific deployment ID (e.g. one left over by an earlier "
             "PER_DEPLOYMENT run); omit to auto-generate.",
    )
    parser.add_argument(
        "--keep-resources",
        action="store_true",
        help="Skip trainer cancellation + deployment scale-to-zero on exit.",
    )
    args = parser.parse_args()

    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("FIREWORKS_API_KEY not set")
    if not os.path.exists(_DEEPMATH_DATASET):
        sys.exit(f"deepmath dataset missing at {_DEEPMATH_DATASET}")

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://dev.api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(
        api_key=api_key, base_url=base_url, hotload_api_url=base_url,
    )

    deployment_id = args.deployment_id or f"depfirst-manual-{uuid.uuid4().hex[:8]}"
    logger.info("=== Manual PER_DEPLOYMENT test ===")
    logger.info(
        "deployment_id=%s  base_model=%s  training_shape=%s",
        deployment_id, args.base_model, args.training_shape,
    )

    config = rl_loop.Config(
        log_path=tempfile.mkdtemp(prefix="depfirst_manual_"),
        base_model=args.base_model,
        dataset=_DEEPMATH_DATASET,
        learning_rate=1e-5,
        kl_beta=0.001,
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        max_completion_tokens=128,
        max_rows=5,
        epochs=1,
        lora_rank=args.lora_rank,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            ref_training_shape_id=args.ref_training_shape,
        ),
        deployment=DeployConfig(
            deployment_id=deployment_id,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=600,
            weight_sync_scope=WeightSyncScope.PER_DEPLOYMENT,  # <-- the scope under test
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=1,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
        wandb=WandBConfig(run_name=f"depfirst-{deployment_id}"),
    )

    # Lazy import: train_deepmath reads FIREWORKS_API_KEY at import time.
    from training.examples.rl.deepmath.train_deepmath import deepmath_reward

    rl_loop.reward_fn = deepmath_reward
    rl_loop.should_accept = lambda _: True  # avoid zero-variance filter on tiny runs

    metrics = rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=not args.keep_resources,
    )
    logger.info("Run metrics: %s", metrics)

    # Invariant: PER_DEPLOYMENT never sets hot_load_trainer_job on the
    # deployment — the coupling is on the trainer side.
    assert config.deployment.hot_load_trainer_job is None, (
        "PER_DEPLOYMENT regression: hot_load_trainer_job was set on the deployment"
    )
    assert metrics.get("policy_job_id"), f"no trainer job created: {metrics}"

    if args.keep_resources:
        logger.info("keep-resources set — deployment %s left running", deployment_id)

    time.sleep(2)
    logger.info("=== PASS === PER_DEPLOYMENT exercised; trainer=%s", metrics.get("policy_job_id"))


if __name__ == "__main__":
    main()
