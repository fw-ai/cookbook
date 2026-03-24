#!/usr/bin/env python3
"""E2E RLOR test with adaptive concurrency on deepmath dataset.

Runs a full GRPO training loop (100 rows, no dynamic filtering) using
the AdaptiveConcurrencyController instead of fixed concurrency.

Reuses the existing deployment e2e-pressure-chengxili-v2 if alive,
otherwise creates one.

Usage:
    FIREWORKS_API_KEY=... python training/tests/e2e/run_rlor_adaptive.py
"""

from __future__ import annotations

import os
import sys
import logging
import time

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.utils import (
    ConcurrencyConfig,
    InfraConfig,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
)
from training.utils.rl import TISConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FIREWORKS_API_KEY", "fw_58efLjimG74e2zwAf69iqS")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = "e2e-pressure-chengxili-v2"
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-30b-a3b-instruct-2507-128k-b200"
REF_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-30b-a3b-instruct-2507-128k-b200-ref"
REGION = "US_OHIO_1"


def main():
    os.environ["FIREWORKS_API_KEY"] = API_KEY
    os.environ["FIREWORKS_BASE_URL"] = BASE_URL

    from fireworks.training.sdk import DeploymentManager, TrainerJobManager
    import training.recipes.rl_loop as rl_loop

    # Import deepmath reward function.
    sys.path.insert(0, os.path.join(_SRC, "training", "examples", "deepmath_rl"))
    from train_deepmath import deepmath_reward

    rlor_mgr = TrainerJobManager(api_key=API_KEY, base_url=BASE_URL)
    deploy_mgr = DeploymentManager(
        api_key=API_KEY,
        base_url=BASE_URL,
        hotload_api_url=BASE_URL,
    )

    dataset_path = os.path.join(
        _SRC, "training", "examples", "deepmath_rl", "dataset.jsonl"
    )

    config = rl_loop.Config(
        log_path=f"./rlor_adaptive_test_{int(time.time()) % 100000}",
        base_model="accounts/fireworks/models/qwen3-30b-a3b-instruct-2507",
        dataset=dataset_path,
        learning_rate=1e-5,
        kl_beta=0.001,
        completions_per_prompt=8,
        max_completion_tokens=30 * 1024,
        temperature=1.0,
        epochs=1,
        max_rows=100,
        prompt_groups_per_step=32,
        tis=TISConfig(cap=2.0),
        concurrency=ConcurrencyConfig(mode="adaptive"),
        infra=InfraConfig(
            training_shape_id=TRAINING_SHAPE,
            region=REGION,
        ),
        deployment=DeployConfig(
            deployment_id=DEPLOYMENT_ID,
            deployment_region=REGION,
            tokenizer_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            sample_timeout=1200,
            replica_count=2,
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=1,
            dcp_save_interval=0,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
        wandb=WandBConfig(
            project="grpo-adaptive-test",
            run_name=f"adaptive-{int(time.time()) % 100000}",
        ),
    )

    # Override reward function.
    rl_loop.reward_fn = deepmath_reward

    # Disable dynamic filtering -- always accept.
    rl_loop.should_accept = lambda pg: True

    logger.info("Starting RLOR with adaptive concurrency, 100 rows, no filter")
    logger.info("Deployment: %s, Training shape: %s", DEPLOYMENT_ID, TRAINING_SHAPE)

    metrics = rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=False,
    )

    logger.info("RLOR COMPLETE. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
