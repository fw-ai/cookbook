#!/usr/bin/env python3
"""Minimal hotload re-attach smoke test.

Validates the fix in fw-ai/fireworks#21731 (control-plane re-attach for
shape-based deployments) together with cookbook PR #301 and SDK PR #116
(``DeploymentManager.update`` + ``WeightSyncer.reset_delta_chain``).

Flow (single run):
    1. Create policy trainer T1 (qwen3-4b-minimum, 1xH200, 65k ctx).
    2. Create or reuse a deployment D, with hot_load_trainer_job=T1.
    3. Hotload an initial *base* checkpoint from T1 -> D.
    4. Delete T1.
    5. Create a new policy trainer T2 with the same training shape.
    6. PATCH D to point at T2 via ``setup_or_reattach_deployment``.
       The shared WeightSyncer's delta-chain state is reset so the next
       save is a fresh base, not a delta against T1's bucket.
    7. Hotload a base checkpoint from T2 -> D. Success means re-attach
       worked end-to-end.
    8. Skip cleanup. Print --deployment-id and the surviving trainer ID
       so the next run can pin them and skip the slow setup.

Re-use across runs:
    First run:
        python test_reattach.py
    Next runs (fast):
        python test_reattach.py \
            --deployment-id <id-from-prev-run> \
            --initial-job-id <T2-from-prev-run>
    The script always creates one fresh trainer per run (it has to --
    that is the thing under test). Re-using the deployment + the previous
    surviving trainer cuts the slowest two boots out of the loop.

Required env (use the shared dev gateway -- it has all the feature
flags pre-configured for hotload):
    FIREWORKS_API_KEY  e.g. fw_3ZkNBrXgLw1EJ4y77kqSMBU5
    FIREWORKS_BASE_URL https://dev.api.fireworks.ai

Run from the cookbook root:
    cd ~/workspace_fw/cookbook
    PYTHONPATH=. python training/examples/hotload_reattach/test_reattach.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make ``training.utils...`` importable when invoked directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.weight_syncer import WeightSyncer

from training.utils import (
    DeployConfig,
    InfraConfig,
    ReconnectableClient,
    ShapeSelectionRequest,
    create_trainer_job,
    materialize_profile_infra,
    select_validated_launch_shapes,
    setup_or_reattach_deployment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hotload-reattach-test")

# ----------------------------------------------------------------------------
# Defaults: qwen3-4b-minimum (1xH200, 65k ctx) on the shared dev account.
# ----------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"
DEFAULT_TRAINING_SHAPE = (
    "accounts/fireworks/trainingShapes/qwen3-4b-minimum"
)
# qwen3-4b-minimum is 1xB200, 65k ctx; pinned to dep-shape rft-qwen3-4b/az2rbxop
# (image 4.146.1) which has the chart fix from PR #20756.
# B200 lives in US_UTAH_1, US_OHIO_1, US_WASHINGTON_4 on prod.
DEFAULT_REGION = "US_OHIO_1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER)
    p.add_argument("--training-shape", default=DEFAULT_TRAINING_SHAPE)
    p.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Both training and deployment region (1xH200 -> US_VIRGINIA_1).",
    )
    p.add_argument(
        "--deployment-id",
        default=None,
        help="Reuse this deployment across runs. Omit to auto-create.",
    )
    p.add_argument(
        "--initial-job-id",
        default=None,
        help=(
            "Pre-created policy trainer to use as the *initial* trainer "
            "(T1). Typically the surviving trainer printed by the previous "
            "run. Omit to create a fresh one."
        ),
    )
    return p.parse_args()


def _build_managers() -> tuple[TrainerJobManager, DeploymentManager, str]:
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(
        api_key=api_key,
        base_url=base_url,
        hotload_api_url=base_url,
    )
    return rlor_mgr, deploy_mgr, api_key


def _hotload(syncer: WeightSyncer, name: str, label: str) -> None:
    logger.info("=== %s hotload ===", label)
    snapshot = syncer.save_and_hotload(name)
    if not snapshot:
        raise RuntimeError(f"{label} hotload returned no snapshot")
    logger.info(
        "%s hotload OK: snapshot=%s timing=%s",
        label,
        snapshot,
        syncer.last_timing,
    )


def main() -> None:
    args = parse_args()
    rlor_mgr, deploy_mgr, api_key = _build_managers()

    # 1. Resolve the training shape (and the deployment shape it pins).
    selection = select_validated_launch_shapes(
        rlor_mgr,
        deploy_mgr=deploy_mgr,
        request=ShapeSelectionRequest(
            base_model=args.base_model,
            trainer_role="policy",
            needs_deployment=True,
            explicit_training_shape_id=args.training_shape,
        ),
    )
    profile = selection.training_profile
    if profile is None:
        raise RuntimeError(
            f"No training profile resolved for {args.training_shape}"
        )
    infra = InfraConfig(
        training_shape_id=selection.training_shape_id,
        region=args.region,
    )
    policy_infra = materialize_profile_infra(infra, profile)

    deploy_cfg = DeployConfig(
        deployment_id=args.deployment_id,
        deployment_shape=selection.deployment_shape,
        deployment_region=args.region,
        tokenizer_model=args.tokenizer_model,
        replica_count=1,
        sample_timeout=600,
    )
    logger.info(
        "Resolved | training_shape=%s | deployment_shape=%s | region=%s",
        selection.training_shape_id,
        selection.deployment_shape,
        args.region,
    )

    # 2. Bring up the initial trainer (T1).
    initial_ep = create_trainer_job(
        rlor_mgr,
        base_model=args.base_model,
        infra=policy_infra,
        profile=profile,
        max_seq_len=profile.max_supported_context_length,
        display_name="reattach-test-T1",
        job_id=args.initial_job_id,
    )
    logger.info("Initial trainer T1 ready: %s", initial_ep.job_id)

    # 3. Setup or re-attach the deployment to T1.
    dep_info = setup_or_reattach_deployment(
        deploy_mgr,
        deploy_cfg,
        args.base_model,
        infra,
        initial_ep.job_name,
    )
    logger.info(
        "Deployment %s ready (state=%s, model=%s)",
        deploy_cfg.deployment_id,
        dep_info.state,
        dep_info.inference_model,
    )

    # 4. Wire up the WeightSyncer and run the initial hotload.
    policy = ReconnectableClient(
        rlor_mgr,
        initial_ep.job_id,
        args.base_model,
        lora_rank=0,
        fw_api_key=api_key,
    )
    syncer = WeightSyncer(
        policy_client=policy.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=deploy_cfg.deployment_id,
        base_model=args.base_model,
        hotload_timeout=1200,
        first_checkpoint_type="base",
    )
    _hotload(syncer, "reattach-test-initial", "Initial (T1)")

    # 5. Tear down T1. The deployment is now pointing at a deleted trainer's
    #    bucket -- exactly the broken state PR 21731 fixes.
    try:
        policy.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to close T1 client cleanly: %s", e)
    logger.info("Deleting initial trainer T1=%s ...", initial_ep.job_id)
    rlor_mgr.delete(initial_ep.job_id)

    # 6. Bring up a fresh trainer (T2).
    new_ep = create_trainer_job(
        rlor_mgr,
        base_model=args.base_model,
        infra=policy_infra,
        profile=profile,
        max_seq_len=profile.max_supported_context_length,
        display_name="reattach-test-T2",
    )
    logger.info("New trainer T2 ready: %s", new_ep.job_id)

    # 7. PATCH the deployment's hot_load_trainer_job to T2 and reset the
    #    delta-chain state so the next checkpoint is a base.
    setup_or_reattach_deployment(
        deploy_mgr,
        deploy_cfg,
        args.base_model,
        infra,
        new_ep.job_name,
        weight_syncer=syncer,
    )

    # 8. Re-wire the syncer to T2 and verify the next hotload succeeds.
    new_policy = ReconnectableClient(
        rlor_mgr,
        new_ep.job_id,
        args.base_model,
        lora_rank=0,
        fw_api_key=api_key,
    )
    syncer.policy_client = new_policy.inner
    _hotload(syncer, "reattach-test-post", "Post-reattach (T2)")

    # 9. Skip cleanup. Print everything needed to re-run quickly.
    logger.info(
        "\n=== SUCCESS ===\n"
        "Re-use these on the next run for fast iteration:\n"
        "  --deployment-id %s \\\n"
        "  --initial-job-id %s\n"
        "(Both deployment and the surviving trainer are intentionally left "
        "running. Delete manually when you are done iterating.)",
        deploy_cfg.deployment_id,
        new_ep.job_id,
    )


if __name__ == "__main__":
    main()
