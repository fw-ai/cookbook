"""Minimal Fireworks deployment spinup/teardown for the verifier probe.

This is a deliberately small wrapper around
``fireworks.training.sdk.deployment.{DeploymentManager,DeploymentConfig}``
plus ``TrainerJobManager.resolve_training_profile`` for shape resolution.
It mirrors the path that ``training/recipes/rl_loop.py`` uses to
provision a serving deployment; the only purpose here is to hand the
probe a ``--model`` identifier. Nothing in the verifier package depends
on it directly.

Usage::

    # Create (or reuse) a deployment for GLM5 using a versioned deployment shape
    python -m training.renderer.verifier.spinup_deployment up \\
        --base-model accounts/fireworks/models/glm-5p1 \\
        --shape accounts/fireworks/deploymentShapes/glm-5p1-b300/versions/jqami1br \\
        --deployment-id my-glm5-probe

    # Or pass a training shape — its pinned deployment_shape_version is used:
    python -m training.renderer.verifier.spinup_deployment up \\
        --base-model accounts/fireworks/models/glm-5p1 \\
        --shape accounts/fireworks/trainingShapes/ts-glm-5p1-policy

    # → prints, on the last line:
    #   accounts/<account>/deployments/my-glm5-probe
    #
    # Pass that string as --model to ``python -m training.renderer.verifier render``.

    # Tear it down when probing is done
    python -m training.renderer.verifier.spinup_deployment down --deployment-id my-glm5-probe

Shape resolution rules (matching the rest of the cookbook):

* ``accounts/.../deploymentShapes/<id>/versions/<v>`` → used as-is.
* ``accounts/.../deploymentShapes/<id>`` → latest validated version.
* ``accounts/.../trainingShapes/<id>/versions/<v>`` → that version's
  pinned ``deployment_shape_version``.
* ``accounts/.../trainingShapes/<id>`` → latest-validated training-shape
  version's pinned ``deployment_shape_version``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

# Documented example for GLM5; override with --shape and --base-model for other renderers.
DEFAULT_GLM5_BASE_MODEL = "accounts/fireworks/models/glm-5p1"
DEFAULT_GLM5_SHAPE = (
    "accounts/fireworks/deploymentShapes/glm-5p1-b300/versions/jqami1br"
)


def _build_deployment_manager(api_key: str | None, base_url: str | None):
    api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit(
            "FIREWORKS_API_KEY not set. Pass --api-key or export FIREWORKS_API_KEY."
        )
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    from fireworks.training.sdk.deployment import DeploymentManager

    return DeploymentManager(api_key=api_key, base_url=base_url)


def _build_trainer_manager(api_key: str | None, base_url: str | None):
    api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    base_url = base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    from fireworks.training.sdk.trainer import TrainerJobManager

    return TrainerJobManager(api_key=api_key, base_url=base_url)


def _resolve_deployment_shape_version(
    deploy_mgr,
    deployment_shape: str,
) -> str:
    """Return a fully-versioned deployment shape resource string.

    Accepts versioned input untouched; falls back to the latest validated
    version when the input is unversioned. This duplicates the small
    request the cookbook's ``infra._get_deployment_shape_version`` runs;
    we keep it inline here so this script has no internal cookbook
    dependency aside from public SDK classes.
    """
    if "/versions/" in deployment_shape:
        return deployment_shape
    from urllib.parse import urlencode

    path = (
        f"/v1/{deployment_shape}/versions?"
        f"{urlencode({'filter': 'latest_validated=true', 'pageSize': 1})}"
    )
    resp = deploy_mgr._get(path, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    versions = data.get("deploymentShapeVersions", []) or []
    if not versions:
        raise RuntimeError(
            f"No latest-validated version found for deployment shape {deployment_shape!r}"
        )
    name = versions[0].get("name")
    if not name:
        raise RuntimeError(
            f"Deployment-shape version response missing 'name' for {deployment_shape!r}"
        )
    return name


def _resolve_shape(
    *,
    api_key: str | None,
    base_url: str | None,
    shape: str,
) -> tuple[str, str | None]:
    """Resolve --shape (deployment or training) into a deployment shape version.

    Returns ``(deployment_shape_version, training_shape_version_or_None)``.
    """
    deploy_mgr = _build_deployment_manager(api_key, base_url)

    if "/deploymentShapes/" in shape:
        return _resolve_deployment_shape_version(deploy_mgr, shape), None

    if "/trainingShapes/" in shape:
        trainer_mgr = _build_trainer_manager(api_key, base_url)
        # ``resolve_training_profile`` natively handles versioned vs unversioned
        # input — it picks the latest validated version when unversioned.
        profile = trainer_mgr.resolve_training_profile(shape)
        deploy_shape_version = getattr(profile, "deployment_shape_version", "") or ""
        if not deploy_shape_version:
            raise RuntimeError(
                f"Training shape {shape!r} has no pinned deployment_shape_version. "
                "The shape is broken and should be fixed; alternatively, pass an "
                "explicit deployment shape via --shape."
            )
        return deploy_shape_version, profile.training_shape_version

    raise SystemExit(
        f"--shape {shape!r} is not a deployment-shape or training-shape "
        "resource. Expected accounts/<acct>/deploymentShapes/... or "
        "accounts/<acct>/trainingShapes/..."
    )


def _default_deployment_id(base_model: str) -> str:
    short = base_model.rsplit("/", 1)[-1]
    return f"{short}-verifier-{int(time.time())}"


def _cmd_up(args: argparse.Namespace) -> int:
    from fireworks.training.sdk.deployment import DeploymentConfig

    deployment_shape_version, training_shape_version = _resolve_shape(
        api_key=args.api_key,
        base_url=args.base_url,
        shape=args.shape,
    )
    if training_shape_version:
        logger.info(
            "training shape %s pins deployment shape %s",
            training_shape_version,
            deployment_shape_version,
        )
    else:
        logger.info("using deployment shape %s", deployment_shape_version)

    deploy_mgr = _build_deployment_manager(args.api_key, args.base_url)
    deployment_id = args.deployment_id or _default_deployment_id(args.base_model)

    cfg = DeploymentConfig(
        deployment_id=deployment_id,
        base_model=args.base_model,
        deployment_shape=deployment_shape_version,
        min_replica_count=args.min_replicas,
        max_replica_count=args.max_replicas,
        skip_shape_validation=args.skip_shape_validation,
    )
    if args.region:
        cfg.region = args.region

    logger.info(
        "creating-or-getting deployment %s (base_model=%s, shape=%s)",
        cfg.deployment_id,
        cfg.base_model,
        cfg.deployment_shape,
    )
    info = deploy_mgr.create_or_get(cfg)

    if info.state not in ("READY", "UPDATING"):
        logger.info(
            "waiting for deployment %s (current state=%s, timeout=%ds)",
            info.deployment_id,
            info.state,
            args.timeout_s,
        )
        info = deploy_mgr.wait_for_ready(info.deployment_id, timeout_s=args.timeout_s)

    if info.inference_model is None:
        info.inference_model = (
            f"accounts/{deploy_mgr.account_id}/deployments/{info.deployment_id}"
        )

    logger.info("deployment ready: state=%s", info.state)
    # Final stdout line is the identifier the probe consumes; keep stdout
    # clean so `MODEL=$(... up ... | tail -1)` works.
    print(info.inference_model)
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    deploy_mgr = _build_deployment_manager(args.api_key, args.base_url)
    logger.info("deleting deployment %s", args.deployment_id)
    try:
        deploy_mgr.delete(args.deployment_id)
    except AttributeError:
        # Older SDKs only expose the private hook.
        deploy_mgr._delete_deployment(args.deployment_id)  # type: ignore[attr-defined]
    logger.info("delete request submitted for %s", args.deployment_id)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.renderer.verifier.spinup_deployment",
        description="Minimal spinup/teardown helper for verifier probe deployments.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Fireworks API key. Falls back to FIREWORKS_API_KEY env var.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Fireworks control-plane URL. Falls back to FIREWORKS_BASE_URL "
        "env var, then https://api.fireworks.ai.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="Create or reuse a deployment and wait until READY.")
    up.add_argument(
        "--base-model",
        default=DEFAULT_GLM5_BASE_MODEL,
        help=f"Fireworks base model. Default: {DEFAULT_GLM5_BASE_MODEL}.",
    )
    up.add_argument(
        "--shape",
        default=DEFAULT_GLM5_SHAPE,
        help="Deployment-shape OR training-shape resource. Versioned input "
        "is used as-is; unversioned input resolves to the latest validated "
        "version. Training-shape input is resolved to its pinned "
        f"deployment_shape_version. Default: {DEFAULT_GLM5_SHAPE}.",
    )
    up.add_argument(
        "--deployment-id",
        default=None,
        help="Reuse-or-create id. Auto-generated from base model if omitted.",
    )
    up.add_argument(
        "--min-replicas", type=int, default=0,
        help="Min replica count. Default 0 (scales to zero when idle).",
    )
    up.add_argument(
        "--max-replicas", type=int, default=1,
        help="Max replica count. Default 1; the probe is single-stream.",
    )
    up.add_argument(
        "--region", default=None,
        help="Optional region override (e.g. NA_BRITISHCOLUMBIA_1 for B300). "
        "Inferred from accelerator type when omitted.",
    )
    up.add_argument(
        "--timeout-s", type=int, default=1800,
        help="Wait-for-ready timeout. Default 1800s (30 min).",
    )
    up.add_argument(
        "--skip-shape-validation", action="store_true",
        help="Pass skipShapeValidation=true. Use only if the shape version is unsupported.",
    )

    down = sub.add_parser("down", help="Delete a deployment by id.")
    down.add_argument("--deployment-id", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if args.cmd == "up":
        return _cmd_up(args)
    if args.cmd == "down":
        return _cmd_down(args)
    return 2  # pragma: no cover - argparse enforces required=True


if __name__ == "__main__":
    raise SystemExit(main())
