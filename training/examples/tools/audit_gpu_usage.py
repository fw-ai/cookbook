#!/usr/bin/env python3
"""Audit which GPUs an account is holding, and who is holding them.

Answers the question the fine-tuning dashboard cannot: *quota says N GPUs are
used, but I only count M on the dashboard -- where are the other N-M?*

The gap is almost always one of three things, and this tool separates them:

1. **The dashboard's trainer list is only half of an RL job.** Every RL job is
   a trainer *plus* a rollout (sampler) deployment, and both charge the same
   ``training-<gpu>-count`` quota. The fine-tuning page lists the trainer side;
   the rollout side lives under Deployments.
2. **A rollout deployment can outlive its trainer.** Deleting a trainer job
   tombstones it (``ListRlorTrainerJobs`` filters ``JOB_STATE_DELETED``, so it
   vanishes from the dashboard immediately) but does *not* tear down the paired
   deployment. Cookbook rollout deployments are pinned to
   ``min_replica_count == max_replica_count``, so an orphan can never scale to
   zero on its own -- it holds its GPUs until someone deletes it.
3. **Quota is charged on the autoscaling ceiling, not on running replicas.**
   A deployment is billed against quota for
   ``acceleratorCount * maxReplicaCount`` even when zero replicas are up
   (unschedulable, cold, or scaled down via ``minReplicaCount`` alone). Only
   ``maxReplicaCount == 0`` -- or deletion -- releases quota.

Deployments carry no ``createdBy`` field, so per-user attribution comes from the
owning trainer job. A rollout is matched to its trainer via
``hotLoadTrainerJob`` (or the ``<trainer-id>-rollout`` naming convention) and
attributed to that job's ``createdBy``. A tombstoned trainer is still readable
by direct GET, so an orphan whose trainer was deleted can still be traced back
to its owner. Anything that cannot be traced is reported as unattributed rather
than silently dropped.

Read-only: issues GET requests only, and never mutates or deletes anything. It
prints the ``firectl`` commands to run for cleanup instead.

Usage:
    export FIREWORKS_API_KEY=...

    # Audit the account the key resolves to:
    python audit_gpu_usage.py

    # Audit a specific account, and only show B200:
    python audit_gpu_usage.py --account-id <account-id> --accelerator-type NVIDIA_B200_180GB

    # Just the leaks:
    python audit_gpu_usage.py --orphans-only

    # Machine-readable:
    python audit_gpu_usage.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.fireworks.ai"

#: Trainer job states that no longer hold GPUs. Everything else is charged.
TERMINAL_JOB_STATES = frozenset(
    {
        "JOB_STATE_COMPLETED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_DELETED",
        "JOB_STATE_EXPIRED",
    }
)

#: Annotation the SDK stamps on rollout deployments it provisions.
MANAGED_ROLLOUT_ANNOTATION = "fireworks-training-sdk/managed-rollout"

KIND_MANAGED_ROLLOUT = "managed-rollout"
KIND_UNATTACHED_ROLLOUT = "unattached-rollout"
KIND_INFERENCE = "inference"


def quota_suffix(accelerator_type: str) -> str:
    """``NVIDIA_B200_180GB`` -> ``b200``, matching the quota resource names.

    Quotas are named ``training-<suffix>-count`` and ``global--<suffix>-count``.
    """
    return accelerator_type.removeprefix("NVIDIA_").split("_")[0].lower()


def strip_version(resource: str) -> str:
    """``.../trainingShapes/foo/versions/abc123`` -> ``.../trainingShapes/foo``."""
    return (resource or "").rsplit("/versions/", 1)[0]


def deployment_charge(deployment: dict) -> int:
    """GPUs a deployment holds against quota.

    The autoscaling *ceiling*, not the running replica count: quota is reserved
    for ``maxReplicaCount`` so that a scale-up never has to fail on quota. A
    deployment with zero ready replicas still charges its ceiling.
    """
    return int(deployment.get("acceleratorCount") or 0) * int(
        deployment.get("maxReplicaCount") or 0
    )


def shape_for(job: dict, shapes: dict[str, dict]) -> dict:
    """Resolve a job's training shape. Jobs pin a shape *version*; shapes are
    catalogued unversioned, so fall back to the version-stripped name."""
    ref = job.get("trainingShapeVersion") or ""
    return shapes.get(ref) or shapes.get(strip_version(ref)) or {}


def trainer_charge(job: dict, shapes: dict[str, dict]) -> int:
    """GPUs a trainer job holds against quota.

    ``trainerReplicaCount`` replicas of the training shape, each of which is
    ``shape.acceleratorCount`` GPUs. The per-replica width lives on the shape,
    not the job, which is why a job that looks like "8 GPUs" on the dashboard
    can be far wider.
    """
    if job.get("state") in TERMINAL_JOB_STATES:
        return 0
    per_replica = int(shape_for(job, shapes).get("acceleratorCount") or 0)
    return int(job.get("trainerReplicaCount") or 1) * per_replica


def trainer_accelerator_type(job: dict, shapes: dict[str, dict]) -> str:
    return shape_for(job, shapes).get("acceleratorType") or "UNKNOWN"


def rollout_shape_names(training_shapes: list[dict]) -> set[str]:
    """Deployment shapes that serve rollouts, taken from the training-shape catalogue.

    Each training shape names the deployment shape its rollout fleet runs on, so
    the catalogue identifies rollout-shaped deployments without hard-coding a
    naming prefix. This is what lets the audit recognise a rollout deployment
    that has lost both its annotation and its trainer link.
    """
    return {
        strip_version(shape["deploymentShapeVersion"])
        for shape in training_shapes
        if shape.get("deploymentShapeVersion")
    }


def rollout_trainer_id(deployment: dict) -> str | None:
    """The trainer job a rollout deployment belongs to, if it can be determined.

    ``hotLoadTrainerJob`` is authoritative. Detached rollouts lose it, so fall
    back to the ``<trainer-id>-rollout`` naming convention the cookbook uses.
    """
    attached = deployment.get("hotLoadTrainerJob") or ""
    if attached:
        return attached.rsplit("/", 1)[-1]
    short_name = (deployment.get("name") or "").rsplit("/", 1)[-1]
    if short_name.endswith("-rollout"):
        return f"{short_name[: -len('-rollout')]}-trainer"
    return None


def classify_deployment(
    deployment: dict, rollout_shapes: frozenset[str] | set[str] = frozenset()
) -> str:
    """Split deployments into SDK-managed rollouts, orphans, and inference.

    An SDK-managed rollout is charged to ``training-<gpu>-count``; everything
    else is charged to ``global--<gpu>-count``. A rollout that loses its trainer
    attachment therefore moves quota buckets, which is why "just ignore
    ``global--*``" hides leaked training capacity.

    A rollout is recognised by its annotation, its ``<trainer>-rollout`` name, or
    its deployment shape appearing in the rollout-shape catalogue -- an orphan
    created by an older SDK has none of the first two.
    """
    annotations = deployment.get("annotations") or {}
    attached = bool(deployment.get("hotLoadTrainerJob"))
    is_managed = annotations.get(MANAGED_ROLLOUT_ANNOTATION) == "true"
    if is_managed and attached:
        return KIND_MANAGED_ROLLOUT
    on_rollout_shape = strip_version(deployment.get("deploymentShape") or "") in (
        rollout_shapes or frozenset()
    )
    if is_managed or on_rollout_shape or rollout_trainer_id(deployment):
        return KIND_UNATTACHED_ROLLOUT
    return KIND_INFERENCE


@dataclass
class Holding:
    """One resource holding GPUs."""

    resource: str
    kind: str
    accelerator_type: str
    gpus: int
    owner: str
    state: str
    detail: str = ""


@dataclass
class Audit:
    holdings: list[Holding] = field(default_factory=list)
    quotas: dict[str, dict] = field(default_factory=dict)

    def by_owner(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for h in self.holdings:
            out.setdefault(h.owner, {}).setdefault(h.accelerator_type, 0)
            out[h.owner][h.accelerator_type] += h.gpus
        return out

    def orphans(self) -> list[Holding]:
        return [h for h in self.holdings if h.kind == KIND_UNATTACHED_ROLLOUT]

    def reconcile(self) -> list[dict]:
        """Compare computed charges against what the quota API reports.

        A positive ``unexplained`` means quota is holding capacity that no
        visible resource accounts for -- a control-plane-side leak worth
        escalating, not something the account owner can clean up. Quota rows
        reporting usage are always included, even when nothing is found behind
        them, since that is precisely the case worth seeing.
        """
        computed: dict[str, int] = {}
        for h in self.holdings:
            if h.accelerator_type == "UNKNOWN":
                continue
            bucket = "training" if h.kind in (KIND_MANAGED_ROLLOUT, "trainer") else "global"
            key = (
                f"training-{quota_suffix(h.accelerator_type)}-count"
                if bucket == "training"
                else f"global--{quota_suffix(h.accelerator_type)}-count"
            )
            computed[key] = computed.get(key, 0) + h.gpus

        buckets = set(computed) | {
            name
            for name, row in self.quotas.items()
            if (name.startswith("training-") or name.startswith("global--"))
            and name.endswith("-count")
            and int(row.get("usage") or 0) > 0
        }

        rows: list[dict] = []
        for name in sorted(buckets):
            quota = self.quotas.get(name)
            reported = int(quota.get("usage") or 0) if quota else 0
            mine = computed.get(name, 0)
            rows.append(
                {
                    "quota": name,
                    "computed": mine,
                    "reported": reported,
                    "limit": int(quota.get("value") or 0) if quota else None,
                    "unexplained": reported - mine,
                }
            )
        return rows


class _Api:
    def __init__(self, api_key: str, base_url: str, account_id: str | None) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self.account_id = account_id or self._resolve_account_id()

    def _resolve_account_id(self) -> str:
        accounts = self.get("/v1/accounts").get("accounts") or []
        if not accounts:
            raise SystemExit(
                "Could not resolve an account from the API key; pass --account-id."
            )
        return accounts[0]["name"].rsplit("/", 1)[-1]

    def get(self, path: str, **params) -> dict:
        resp = self._client.get(path, params=params or None)
        resp.raise_for_status()
        return resp.json()

    def paginate(self, path: str, key: str, **params) -> list[dict]:
        rows: list[dict] = []
        token = ""
        while True:
            page = self.get(path, pageSize=200, pageToken=token or None, **params)
            rows.extend(page.get(key) or [])
            token = page.get("nextPageToken") or ""
            if not token:
                return rows

    def get_job(self, account_id: str, job_id: str) -> dict | None:
        """Read one trainer job, including tombstoned (``JOB_STATE_DELETED``) rows.

        List responses filter tombstones out; a direct GET still returns them
        for 30 days, which is how an orphan outliving its trainer is traced back
        to an owner.
        """
        try:
            return self.get(
                f"/v1/accounts/{account_id}/rlorTrainerJobs/{job_id}",
            )
        except httpx.HTTPStatusError:
            return None

    def close(self) -> None:
        self._client.close()


def build_audit(
    jobs: list[dict],
    deployments: list[dict],
    shapes: dict[str, dict],
    quotas: dict[str, dict],
    owner_lookup: dict[str, str] | None = None,
    rollout_shapes: frozenset[str] | set[str] = frozenset(),
) -> Audit:
    """Assemble the audit from already-fetched control-plane rows.

    Pure: no I/O, so the accounting rules are unit-testable.
    """
    owners = dict(owner_lookup or {})
    audit = Audit(quotas=quotas)

    for job in jobs:
        job_id = (job.get("name") or "").rsplit("/", 1)[-1]
        owner = job.get("createdBy") or "unknown"
        owners.setdefault(job_id, owner)
        gpus = trainer_charge(job, shapes)
        if gpus <= 0:
            continue
        audit.holdings.append(
            Holding(
                resource=job_id,
                kind="trainer",
                accelerator_type=trainer_accelerator_type(job, shapes),
                gpus=gpus,
                owner=owner,
                state=job.get("state") or "",
                detail=(
                    f"{job.get('trainerReplicaCount') or 1} replica(s) x "
                    f"{gpus // max(int(job.get('trainerReplicaCount') or 1), 1)} GPU"
                ),
            )
        )

    for dep in deployments:
        gpus = deployment_charge(dep)
        if gpus <= 0:
            continue
        dep_id = (dep.get("name") or "").rsplit("/", 1)[-1]
        kind = classify_deployment(dep, rollout_shapes)
        trainer_id = rollout_trainer_id(dep)
        owner = owners.get(trainer_id or "", "unattributed")
        ready = int((dep.get("replicaStats") or {}).get("readyReplicaCount") or 0)
        detail = (
            f"{dep.get('maxReplicaCount') or 0} max replica(s) x "
            f"{dep.get('acceleratorCount') or 0} GPU, {ready} ready"
        )
        if kind == KIND_UNATTACHED_ROLLOUT:
            detail += "; no trainer attached"
        audit.holdings.append(
            Holding(
                resource=dep_id,
                kind=kind,
                accelerator_type=dep.get("acceleratorType") or "UNKNOWN",
                gpus=gpus,
                owner=owner,
                state=dep.get("state") or "",
                detail=detail,
            )
        )

    return audit


def fetch_audit(api: _Api, accelerator_type: str | None) -> Audit:
    account = api.account_id
    jobs = api.paginate(f"/v1/accounts/{account}/rlorTrainerJobs", "rlorTrainerJobs")
    deployments = api.paginate(f"/v1/accounts/{account}/deployments", "deployments")
    shape_rows = api.paginate("/v1/accounts/fireworks/trainingShapes", "trainingShapes")
    quota_rows = api.paginate(f"/v1/accounts/{account}/quotas", "quotas")

    shapes = {row["name"]: row for row in shape_rows}
    quotas = {row["name"].rsplit("/", 1)[-1]: row for row in quota_rows}
    rollout_shapes = rollout_shape_names(shape_rows)

    # Trace rollouts whose trainer was deleted (and so is absent from the list).
    listed = {(j.get("name") or "").rsplit("/", 1)[-1] for j in jobs}
    owner_lookup: dict[str, str] = {}
    for dep in deployments:
        trainer_id = rollout_trainer_id(dep)
        if not trainer_id or trainer_id in listed:
            continue
        tombstone = api.get_job(account, trainer_id)
        if tombstone:
            owner_lookup[trainer_id] = tombstone.get("createdBy") or "unknown"

    if accelerator_type:
        deployments = [
            d for d in deployments if d.get("acceleratorType") == accelerator_type
        ]
        jobs = [
            j for j in jobs if trainer_accelerator_type(j, shapes) == accelerator_type
        ]
        # Drop the other types' quota rows too, or they reconcile against zero
        # holdings and report the whole account as unexplained.
        suffix = quota_suffix(accelerator_type)
        quotas = {
            name: row
            for name, row in quotas.items()
            if name in (f"training-{suffix}-count", f"global--{suffix}-count")
        }

    return build_audit(
        jobs, deployments, shapes, quotas, owner_lookup, rollout_shapes=rollout_shapes
    )


def _print_report(audit: Audit, account_id: str, orphans_only: bool) -> None:
    orphans = audit.orphans()

    if not orphans_only:
        print(f"\nGPUs held by account {account_id}\n")
        header = f"{'resource':<44}  {'kind':<20}  {'accelerator':<20}  {'gpus':>5}  {'owner':<28}  detail"
        print(header)
        print("-" * len(header))
        for h in sorted(audit.holdings, key=lambda h: (-h.gpus, h.resource)):
            print(
                f"{h.resource:<44}  {h.kind:<20}  {h.accelerator_type:<20}  "
                f"{h.gpus:>5}  {h.owner:<28}  {h.detail}"
            )

        print("\nPer owner\n")
        for owner, per_type in sorted(audit.by_owner().items()):
            totals = ", ".join(
                f"{gpus} x {accel}" for accel, gpus in sorted(per_type.items())
            )
            print(f"  {owner:<30} {totals}")

        print("\nReconciliation against reported quota usage\n")
        head = f"{'quota':<28}  {'computed':>8}  {'reported':>8}  {'limit':>6}  {'unexplained':>11}"
        print(head)
        print("-" * len(head))
        for row in audit.reconcile():
            limit = "-" if row["limit"] is None else row["limit"]
            print(
                f"{row['quota']:<28}  {row['computed']:>8}  {row['reported']:>8}  "
                f"{limit:>6}  {row['unexplained']:>11}"
            )
        print(
            "\n  unexplained > 0: quota holds capacity no visible resource accounts for."
            "\n                   A stale reservation -- escalate to Fireworks; an"
            "\n                   account owner cannot release it."
            "\n  unexplained < 0: a resource was created moments ago and the quota row"
            "\n                   has not caught up. Re-run shortly."
        )

    if not orphans:
        if orphans_only:
            print("No unattached rollout deployments found.")
        return

    print("\nUnattached rollout deployments (holding GPUs with no trainer)\n")
    for h in orphans:
        print(f"  {h.resource}  {h.gpus} x {h.accelerator_type}  owner={h.owner}")
    print(
        "\nConfirm with the owner before removing. To release the GPUs:\n"
        f"  firectl deployment update <ID> -a {account_id} --min-replica-count 0 --max-replica-count 0\n"
        f"  firectl deployment delete <ID> -a {account_id}\n"
        "Setting only --min-replica-count 0 does NOT release quota; the ceiling is charged."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GPU holdings and quota attribution for a Fireworks account.",
    )
    parser.add_argument(
        "--account-id",
        default=os.environ.get("FIREWORKS_ACCOUNT_ID"),
        help="Account to audit. Defaults to the account the API key resolves to.",
    )
    parser.add_argument(
        "--accelerator-type",
        help="Restrict to one accelerator type, e.g. NVIDIA_B200_180GB.",
    )
    parser.add_argument(
        "--orphans-only",
        action="store_true",
        help="Only report rollout deployments holding GPUs with no trainer attached.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit raw JSON to stdout instead of human tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY is not set.")
    base_url = os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL)

    api = _Api(api_key=api_key, base_url=base_url, account_id=args.account_id)
    try:
        audit = fetch_audit(api, args.accelerator_type)
    finally:
        api.close()

    if args.as_json:
        json.dump(
            {
                "account": api.account_id,
                "holdings": [h.__dict__ for h in audit.holdings],
                "byOwner": audit.by_owner(),
                "orphans": [h.__dict__ for h in audit.orphans()],
                "reconciliation": audit.reconcile(),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    _print_report(audit, api.account_id, args.orphans_only)


if __name__ == "__main__":
    main()
