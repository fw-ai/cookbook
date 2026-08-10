"""Unit tests for ``training.examples.tools.audit_gpu_usage``.

The fixtures mirror real control-plane payloads: a trainer job pinning a shape
*version*, an SDK-managed rollout deployment attached to that job, an orphaned
rollout whose trainer was deleted, and a plain inference deployment.
"""

from __future__ import annotations

from training.examples.tools.audit_gpu_usage import (
    KIND_INFERENCE,
    KIND_MANAGED_ROLLOUT,
    KIND_UNATTACHED_ROLLOUT,
    MANAGED_ROLLOUT_ANNOTATION,
    build_audit,
    classify_deployment,
    deployment_charge,
    quota_suffix,
    rollout_shape_names,
    rollout_trainer_id,
    trainer_charge,
)

ACCOUNT = "acme"
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k-lora"
ROLLOUT_SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3p6-35b-a3b"

SHAPE_ROWS = [
    {
        "name": TRAINING_SHAPE,
        "acceleratorType": "NVIDIA_B200_180GB",
        "acceleratorCount": 4,
        "deploymentShapeVersion": f"{ROLLOUT_SHAPE}/versions/iwj1k21w",
    }
]
SHAPES = {row["name"]: row for row in SHAPE_ROWS}
ROLLOUT_SHAPES = rollout_shape_names(SHAPE_ROWS)


def _trainer(job_id: str, *, state: str, replicas: int, owner: str) -> dict:
    return {
        "name": f"accounts/{ACCOUNT}/rlorTrainerJobs/{job_id}",
        "state": state,
        "trainerReplicaCount": replicas,
        # Jobs pin a shape version; the catalogue is keyed unversioned.
        "trainingShapeVersion": f"{TRAINING_SHAPE}/versions/p87t11ah",
        "createdBy": owner,
    }


def _rollout(dep_id: str, *, replicas: int, attached_to: str | None, managed: bool) -> dict:
    dep = {
        "name": f"accounts/{ACCOUNT}/deployments/{dep_id}",
        "state": "READY",
        "acceleratorType": "NVIDIA_B200_180GB",
        "acceleratorCount": 2,
        "minReplicaCount": replicas,
        "maxReplicaCount": replicas,
        "deploymentShape": f"{ROLLOUT_SHAPE}/versions/iwj1k21w",
        "replicaStats": {"readyReplicaCount": replicas},
    }
    if managed:
        dep["annotations"] = {MANAGED_ROLLOUT_ANNOTATION: "true"}
    if attached_to:
        dep["hotLoadTrainerJob"] = f"accounts/{ACCOUNT}/rlorTrainerJobs/{attached_to}"
    return dep


def test_quota_suffix_maps_accelerator_names_to_quota_rows():
    assert quota_suffix("NVIDIA_B200_180GB") == "b200"
    assert quota_suffix("NVIDIA_H100_80GB") == "h100"
    assert quota_suffix("NVIDIA_GB300") == "gb300"


def test_deployment_charge_uses_the_autoscaling_ceiling_not_running_replicas():
    """Quota reserves ``maxReplicaCount``, so an idle or unschedulable
    deployment still holds GPUs. This is why "it scaled down" does not mean
    "quota was released"."""
    dep = {
        "acceleratorCount": 4,
        "minReplicaCount": 0,
        "maxReplicaCount": 2,
        "replicaStats": {"readyReplicaCount": 0},
    }
    assert deployment_charge(dep) == 8

    parked = {"acceleratorCount": 4, "minReplicaCount": 0, "maxReplicaCount": 0}
    assert deployment_charge(parked) == 0


def test_trainer_charge_multiplies_replicas_by_the_shape_width():
    """The per-replica GPU width lives on the shape, not the job -- a job that
    reads as "8 GPUs" on the dashboard can be much wider."""
    job = _trainer("j-trainer", state="JOB_STATE_RUNNING", replicas=2, owner="a@x.com")
    assert trainer_charge(job, SHAPES) == 8

    wide = _trainer("j2-trainer", state="JOB_STATE_RUNNING", replicas=8, owner="a@x.com")
    assert trainer_charge(wide, SHAPES) == 32


def test_trainer_charge_resolves_a_pinned_shape_version():
    job = _trainer("j-trainer", state="JOB_STATE_RUNNING", replicas=1, owner="a@x.com")
    job["trainingShapeVersion"] = TRAINING_SHAPE  # unversioned reference
    assert trainer_charge(job, SHAPES) == 4


def test_terminal_and_tombstoned_jobs_hold_no_gpus():
    for state in ("JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_DELETED"):
        job = _trainer("j-trainer", state=state, replicas=2, owner="a@x.com")
        assert trainer_charge(job, SHAPES) == 0


def test_classify_separates_attached_rollouts_orphans_and_inference():
    attached = _rollout("run-rollout", replicas=4, attached_to="run-trainer", managed=True)
    assert classify_deployment(attached, ROLLOUT_SHAPES) == KIND_MANAGED_ROLLOUT

    detached = _rollout("run-rollout", replicas=4, attached_to=None, managed=True)
    assert classify_deployment(detached, ROLLOUT_SHAPES) == KIND_UNATTACHED_ROLLOUT

    inference = {
        "name": f"accounts/{ACCOUNT}/deployments/prod",
        "acceleratorType": "NVIDIA_B200_180GB",
        "acceleratorCount": 8,
        "maxReplicaCount": 1,
        "deploymentShape": "accounts/fireworks/deploymentShapes/kimi-k2-thinking-throughput/versions/x",
    }
    assert classify_deployment(inference, ROLLOUT_SHAPES) == KIND_INFERENCE


def test_orphan_on_a_rollout_shape_is_caught_without_annotation_or_naming():
    """An orphan created by an older SDK has no annotation, no ``-rollout``
    suffix, and no trainer link. The rollout-shape catalogue is the only signal
    left, and without it the deployment is misread as production inference."""
    orphan = _rollout("goyeyidq", replicas=8, attached_to=None, managed=False)
    orphan["name"] = f"accounts/{ACCOUNT}/deployments/goyeyidq"

    assert rollout_trainer_id(orphan) is None
    assert classify_deployment(orphan, frozenset()) == KIND_INFERENCE
    assert classify_deployment(orphan, ROLLOUT_SHAPES) == KIND_UNATTACHED_ROLLOUT


def test_rollout_trainer_id_falls_back_to_the_naming_convention():
    attached = _rollout("run-rollout", replicas=2, attached_to="other-trainer", managed=True)
    assert rollout_trainer_id(attached) == "other-trainer"

    detached = _rollout("run-rollout", replicas=2, attached_to=None, managed=True)
    assert rollout_trainer_id(detached) == "run-trainer"


def test_an_rl_job_charges_training_quota_for_trainer_plus_rollout():
    """The dashboard's trainer list is half the job: the reconciliation only
    balances when the rollout deployment is counted too."""
    job = _trainer("run-trainer", state="JOB_STATE_RUNNING", replicas=2, owner="a@x.com")
    rollout = _rollout("run-rollout", replicas=4, attached_to="run-trainer", managed=True)
    quotas = {"training-b200-count": {"usage": 16, "value": "256"}}

    audit = build_audit([job], [rollout], SHAPES, quotas, rollout_shapes=ROLLOUT_SHAPES)

    assert audit.by_owner() == {"a@x.com": {"NVIDIA_B200_180GB": 16}}
    row = next(r for r in audit.reconcile() if r["quota"] == "training-b200-count")
    assert row == {
        "quota": "training-b200-count",
        "computed": 16,
        "reported": 16,
        "limit": 256,
        "unexplained": 0,
    }


def test_orphaned_rollout_is_attributed_to_its_deleted_trainers_owner():
    """The trainer row is tombstoned and absent from list responses, so the
    owner arrives via a direct GET passed in as ``owner_lookup``. The GPUs move
    to the ``global--`` bucket because nothing is attached."""
    orphan = _rollout("gone-rollout", replicas=8, attached_to=None, managed=True)
    quotas = {
        "training-b200-count": {"usage": 0, "value": "256"},
        "global--b200-count": {"usage": 16, "value": "80"},
    }

    audit = build_audit(
        [],
        [orphan],
        SHAPES,
        quotas,
        owner_lookup={"gone-trainer": "deleter@x.com"},
        rollout_shapes=ROLLOUT_SHAPES,
    )

    assert [h.resource for h in audit.orphans()] == ["gone-rollout"]
    assert audit.by_owner() == {"deleter@x.com": {"NVIDIA_B200_180GB": 16}}
    row = next(r for r in audit.reconcile() if r["quota"] == "global--b200-count")
    assert row["computed"] == 16 and row["unexplained"] == 0


def test_reconcile_surfaces_quota_usage_no_resource_explains():
    """A quota row reporting usage with nothing behind it is a stale
    reservation: invisible in every listing, and not releasable by the account
    owner. It must still appear in the report."""
    quotas = {"training-b300-count": {"usage": 16, "value": "256"}}

    audit = build_audit([], [], SHAPES, quotas, rollout_shapes=ROLLOUT_SHAPES)

    assert audit.reconcile() == [
        {
            "quota": "training-b300-count",
            "computed": 0,
            "reported": 16,
            "limit": 256,
            "unexplained": 16,
        }
    ]


def test_zero_usage_quota_rows_are_omitted():
    quotas = {"training-h100-count": {"usage": 0, "value": "256"}}
    assert build_audit([], [], SHAPES, quotas).reconcile() == []
