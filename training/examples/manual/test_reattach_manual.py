#!/usr/bin/env python3
"""Manual test: PER_TRAINER re-attach flow.

Drives ``rl_loop.main`` twice against the same ``deployment_id``:

1. First invocation creates a fresh trainer + deployment
   (``hot_load_trainer_job`` baked in at creation).
2. Second invocation is what is under test: ``setup_infra`` sees the
   existing deployment, creates a new trainer, and **PATCHes**
   ``hot_load_trainer_job`` to point at it, triggering a serving-pod
   rolling restart and a ``_wait_for_reattach_settled`` wait. After
   settle, the recipe performs a hotload against the new pod.

Minimal run: 1 prompt group per step, 2 completions, 5 rows, 1 epoch
on qwen3-4b + the 1xGPU qwen3-4b-minimum training shape. Reference
https://github.com/fw-ai/fireworks/actions/runs/24703610604 is the CI
that exercises the single-run path — this script exercises the
re-attach path the CI does not cover.

Usage:
    export FIREWORKS_API_KEY=<pyroworks key>
    python training/examples/manual/test_reattach_manual.py

What to look for in the second run's logs:
    Re-attached deployment <dep_id> to trainer <new_trainer> (prev_pod=<old>) —
    settling in parallel with trainer waits
    Re-attach settled: new pod <new> replaced <old>
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
logger = logging.getLogger("manual.reattach")

_DEEPMATH_DATASET = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "rl", "deepmath", "dataset.jsonl",
    )
)


def _build_config(
    *,
    deployment_id: str,
    base_model: str,
    tokenizer_model: str,
    training_shape: str,
    ref_training_shape: str | None,
    lora_rank: int,
    log_path: str,
    run_label: str,
) -> rl_loop.Config:
    return rl_loop.Config(
        log_path=log_path,
        base_model=base_model,
        dataset=_DEEPMATH_DATASET,
        learning_rate=1e-5,
        kl_beta=0.001,
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        max_completion_tokens=128,
        max_rows=5,
        epochs=1,
        lora_rank=lora_rank,
        infra=InfraConfig(
            training_shape_id=training_shape,
            ref_training_shape_id=ref_training_shape,
        ),
        deployment=DeployConfig(
            deployment_id=deployment_id,
            tokenizer_model=tokenizer_model,
            sample_timeout=600,
            weight_sync_scope=WeightSyncScope.PER_TRAINER,
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=1,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
        wandb=WandBConfig(run_name=run_label),
    )


def _run(cfg: rl_loop.Config, rlor_mgr, deploy_mgr, cleanup_on_exit: bool) -> dict:
    # Lazy import: train_deepmath reads FIREWORKS_API_KEY at import time, so
    # we defer until main() has already validated the env.
    from training.examples.rl.deepmath.train_deepmath import deepmath_reward

    # Patch the reward fn into the recipe module so the same rl_loop.main path
    # used by CI fires with deepmath scoring.
    rl_loop.reward_fn = deepmath_reward
    rl_loop.should_accept = lambda _: True  # avoid zero-variance filter on tiny runs
    return rl_loop.main(
        cfg, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr, cleanup_on_exit=cleanup_on_exit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default="accounts/fireworks/models/qwen3-4b",
    )
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
        help="Reuse a specific deployment ID across both runs; omit to auto-generate.",
    )
    parser.add_argument(
        "--keep-resources",
        action="store_true",
        help="Skip trainer cancellation + deployment scale-to-zero between / after runs.",
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

    deployment_id = args.deployment_id or f"reattach-manual-{uuid.uuid4().hex[:8]}"
    logger.info("=== Manual re-attach test ===")
    logger.info("deployment_id=%s (shared across both runs)", deployment_id)
    logger.info("base_model=%s  training_shape=%s", args.base_model, args.training_shape)

    def _make_cfg(label: str) -> rl_loop.Config:
        return _build_config(
            deployment_id=deployment_id,
            base_model=args.base_model,
            tokenizer_model=args.tokenizer_model,
            training_shape=args.training_shape,
            ref_training_shape=args.ref_training_shape,
            lora_rank=args.lora_rank,
            log_path=tempfile.mkdtemp(prefix=f"reattach_{label}_"),
            run_label=f"reattach-{label}-{deployment_id}",
        )

    # Run 1: cold start. setup_infra creates both trainer + deployment.
    # cleanup_on_exit=False so the deployment survives for run 2.
    logger.info("--- Run 1: cold start ---")
    cfg1 = _make_cfg("r1")
    metrics1 = _run(cfg1, rlor_mgr, deploy_mgr, cleanup_on_exit=False)
    logger.info("Run 1 metrics: %s", metrics1)
    trainer_r1 = metrics1.get("policy_job_id")

    # Best-effort: cancel the first trainer so the second run definitely
    # provisions a fresh one (the control plane is free to reuse, but for
    # a re-attach test we want a different job_name).
    if trainer_r1:
        try:
            logger.info("Cancelling run-1 trainer %s so run-2 creates a fresh one", trainer_r1)
            rlor_mgr.cancel(trainer_r1)
        except Exception as e:
            logger.warning("Could not cancel run-1 trainer (continuing): %s", e)

    # Run 2: re-attach. setup_infra sees the existing deployment, creates a
    # fresh trainer, PATCHes hot_load_trainer_job, waits for the pod roll.
    logger.info("--- Run 2: re-attach ---")
    cfg2 = _make_cfg("r2")
    metrics2 = _run(cfg2, rlor_mgr, deploy_mgr, cleanup_on_exit=not args.keep_resources)
    logger.info("Run 2 metrics: %s", metrics2)

    trainer_r2 = metrics2.get("policy_job_id")
    assert trainer_r1 != trainer_r2, (
        f"re-attach invariant broken: same trainer across both runs ({trainer_r1})"
    )

    # setup_infra is documented "never mutates caller's config" so we can't
    # check cfg2.deployment — the PATCH is on a local copy. Verify server-side
    # by reading the deployment's current hot_load_trainer_job directly.
    resp = deploy_mgr._get(
        f"/v1/accounts/{deploy_mgr.account_id}/deployments/{deployment_id}",
        timeout=30,
    )
    resp.raise_for_status()
    dep_trainer = resp.json().get("hotLoadTrainerJob", "")
    assert trainer_r2 in dep_trainer, (
        f"PER_TRAINER re-attach regression: deployment {deployment_id} "
        f"hotLoadTrainerJob={dep_trainer!r} does not reference run-2 trainer {trainer_r2}"
    )

    if args.keep_resources:
        logger.info("keep-resources set — deployment %s left running", deployment_id)

    # Sleep briefly so post-run cleanup log lines flush before the script exits.
    time.sleep(2)
    logger.info("=== PASS === re-attach exercised; trainers r1=%s, r2=%s", trainer_r1, trainer_r2)


if __name__ == "__main__":
    main()
