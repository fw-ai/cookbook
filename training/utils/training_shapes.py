"""Validated training/deployment shape selection for cookbook recipes."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol
from urllib.parse import urlencode

from fireworks.training.sdk.deployment import DeploymentManager
from fireworks.training.sdk.trainer import TrainerJobManager
from fireworks.training.sdk.fireworks_client import TrainingShapeProfile

from training.utils.config import InfraConfig

logger = logging.getLogger(__name__)

TRAINING_SHAPES_DOCS_URL = (
    "https://docs.fireworks.ai/fine-tuning/training-sdk/training-shapes"
)

_TRAINING_SHAPE_VERSION_PARENT = "accounts/-/trainingShapes/-"
_DEPLOYMENT_SHAPE_VERSION_PARENT = "accounts/-/deploymentShapes/-"
_ORDER_BY_CREATE_TIME_DESC = "create_time desc"
_TRAINING_SHAPE_VERSION_RE = re.compile(r"/versions/[^/]+$")
_TRAINING_SHAPE_RESOURCE_RE = re.compile(
    r"^accounts/[^/]+/trainingShapes/[^/]+(?:/versions/[^/]+)?$"
)
_SHORT_TRAINING_SHAPE_ID_RE = re.compile(r"^ts-[^/]+$")

_TRAINER_MODE_BY_CODE = {
    1: "POLICY_TRAINER",
    2: "FORWARD_ONLY",
    3: "LORA_TRAINER",
}


class _RestCapable(Protocol):
    def _get(self, path: str, **kwargs): ...


@dataclass(frozen=True)
class ShapeSelectionRequest:
    """Inputs for the unified validated-shape selector."""

    base_model: str
    max_seq_len: int | None = None
    trainer_role: Literal["policy", "reference"] = "policy"
    needs_deployment: bool = False
    lora_rank: int = 0
    explicit_training_shape_id: str | None = None
    explicit_deployment_shape: str | None = None


@dataclass(frozen=True)
class ShapeSelectionResult:
    """Resolved shape resources and training profile for one trainer role."""

    request: ShapeSelectionRequest
    training_shape_id: str | None
    training_profile: TrainingShapeProfile | None
    deployment_shape: str | None
    inferred_training_shape: bool
    inferred_deployment_shape: bool


@dataclass(frozen=True)
class _ModelSelectionContext:
    model_type: str
    parameter_count: int
    supports_fireattention: bool


@dataclass(frozen=True)
class _TrainingShapeCandidate:
    training_shape_version: str
    training_shape: str
    base_model: str | None
    model_type: str | None
    parameter_count: int | None
    trainer_mode: str | None
    trainer_image_tag: str
    max_supported_context_length: int
    node_count: int
    deployment_shape_version: str
    deployment_image_tag: str
    accelerator_type: str
    accelerator_count: int
    base_model_weight_precision: str
    pipeline_parallelism: int

    def to_profile(self) -> TrainingShapeProfile:
        return TrainingShapeProfile(
            training_shape_version=self.training_shape_version,
            trainer_image_tag=self.trainer_image_tag,
            max_supported_context_length=self.max_supported_context_length,
            node_count=self.node_count,
            deployment_shape_version=self.deployment_shape_version,
            deployment_image_tag=self.deployment_image_tag,
            accelerator_type=self.accelerator_type,
            accelerator_count=self.accelerator_count,
            base_model_weight_precision=self.base_model_weight_precision,
            pipeline_parallelism=self.pipeline_parallelism,
        )


@dataclass(frozen=True)
class _DeploymentShapeCandidate:
    deployment_shape_version: str
    deployment_shape: str
    base_model: str | None
    model_type: str | None
    parameter_count: int | None
    accelerator_type: str
    accelerator_count: int
    engine: str | None


def canonical_base_model(base_model: str) -> str:
    """Return the model identifier used for exact shape lookups."""
    return base_model


def _canonicalize_training_shape_id(
    training_shape_id: str,
    *,
    default_account: str | None = None,
) -> str:
    """Accept cookbook-friendly shorthand and SDK-level resource names.

    The cookbook historically documented bare ``ts-...`` IDs, while newer SDK
    versions require full ``accounts/<account>/trainingShapes/<shape>``
    resource names. Preserve the shorthand UX here so existing examples and
    tests keep working after SDK upgrades.
    """
    if _TRAINING_SHAPE_RESOURCE_RE.match(training_shape_id):
        return _TRAINING_SHAPE_VERSION_RE.sub("", training_shape_id)
    if "/" in training_shape_id or not _SHORT_TRAINING_SHAPE_ID_RE.match(training_shape_id):
        return training_shape_id
    account = default_account or os.environ.get("FIREWORKS_ACCOUNT_ID") or "fireworks"
    return f"accounts/{account}/trainingShapes/{training_shape_id}"


def materialize_profile_infra(
    infra: InfraConfig,
    profile: TrainingShapeProfile,
) -> InfraConfig:
    """Return an InfraConfig copy populated from a resolved training shape."""
    return replace(
        infra,
        custom_image_tag=getattr(profile, "trainer_image_tag", None) or infra.custom_image_tag,
        accelerator_type=getattr(profile, "accelerator_type", None) or infra.accelerator_type,
        accelerator_count=getattr(profile, "accelerator_count", None) or infra.accelerator_count,
        node_count=getattr(profile, "node_count", None) or infra.node_count,
    )


def prepare_training_shape_launch(
    infra: InfraConfig,
    profile: TrainingShapeProfile | None,
    *,
    client_managed: bool,
) -> tuple[InfraConfig, TrainingShapeProfile | None]:
    """Choose manual-vs-shape launch config for a resolved profile."""
    if not client_managed or profile is None:
        return infra, profile
    return materialize_profile_infra(infra, profile), None


def select_validated_launch_shapes(
    trainer_mgr: TrainerJobManager,
    *,
    request: ShapeSelectionRequest,
    deploy_mgr: DeploymentManager | None = None,
) -> ShapeSelectionResult:
    """Resolve validated trainer/deployment shapes for one trainer role."""
    if request.trainer_role not in {"policy", "reference"}:
        raise ValueError(
            f"Unsupported trainer_role={request.trainer_role!r}; expected 'policy' or 'reference'"
        )
    if (
        request.needs_deployment
        and request.explicit_training_shape_id is None
        and request.max_seq_len is None
    ):
        raise ValueError(
            "max_seq_len is required for auto-selecting deployment-backed training shapes"
        )

    if request.explicit_training_shape_id:
        training_shape_id = _canonicalize_training_shape_id(
            request.explicit_training_shape_id,
            default_account=getattr(trainer_mgr, "_account_id", None),
        )
        profile = trainer_mgr.resolve_training_profile(training_shape_id)
        inferred_training_shape = False
    else:
        candidate = _select_training_shape_candidate(trainer_mgr, request=request)
        training_shape_id = candidate.training_shape
        profile = candidate.to_profile()
        inferred_training_shape = True

    deployment_shape = request.explicit_deployment_shape
    inferred_deployment_shape = False
    if request.needs_deployment and not deployment_shape:
        deployment_shape = (
            getattr(profile, "deployment_shape_version", None)
            or getattr(profile, "deployment_shape", None)
            or _select_deployment_shape_candidate(
                deploy_mgr or trainer_mgr,
                base_model=request.base_model,
            ).deployment_shape_version
        )
        inferred_deployment_shape = bool(deployment_shape)

    return ShapeSelectionResult(
        request=request,
        training_shape_id=training_shape_id,
        training_profile=profile,
        deployment_shape=deployment_shape,
        inferred_training_shape=inferred_training_shape,
        inferred_deployment_shape=inferred_deployment_shape,
    )


def _select_training_shape_candidate(
    client: TrainerJobManager,
    *,
    request: ShapeSelectionRequest,
) -> _TrainingShapeCandidate:
    base_model = canonical_base_model(request.base_model)
    expected_mode = _expected_trainer_mode(request.trainer_role, request.lora_rank)
    deployment_filter = request.explicit_deployment_shape

    exact_candidates = _compatible_training_shape_candidates(
        _list_training_shape_candidates(
            client,
            _build_latest_validated_training_shape_filter(
                base_model=base_model,
                trainer_mode=expected_mode,
                deployment_shape=deployment_filter,
            ),
        ),
        request=request,
        expected_mode=expected_mode,
    )
    if exact_candidates:
        return _choose_training_shape_candidate(exact_candidates, request)

    model_ctx = _fetch_model_selection_context(client, base_model)
    compat_candidates = _compatible_training_shape_candidates(
        _list_training_shape_candidates(
            client,
            _build_compatible_training_shape_filter(
                model_ctx=model_ctx,
                trainer_mode=expected_mode,
                deployment_shape=deployment_filter,
            ),
        ),
        request=request,
        expected_mode=expected_mode,
    )
    if compat_candidates:
        return _choose_training_shape_candidate(compat_candidates, request)

    raise ValueError(_format_training_shape_selection_error(request, expected_mode))


def _select_deployment_shape_candidate(
    client: _RestCapable,
    *,
    base_model: str,
) -> _DeploymentShapeCandidate:
    base_model = canonical_base_model(base_model)
    model_ctx = _fetch_model_selection_context(client, base_model)

    exact_candidates = _list_deployment_shape_candidates(
        client,
        _build_latest_validated_deployment_shape_filter(
            base_model=base_model,
            supports_fireattention=model_ctx.supports_fireattention,
        ),
    )
    if exact_candidates:
        return exact_candidates[0]

    compat_candidates = _list_deployment_shape_candidates(
        client,
        _build_compatible_deployment_shape_filter(model_ctx),
    )
    if compat_candidates:
        return compat_candidates[0]

    raise ValueError(
        "No validated deployment shape is available for "
        f"base_model={base_model!r}. Check deployment-shape docs or provide "
        "DeployConfig.deployment_shape explicitly."
    )


def _compatible_training_shape_candidates(
    candidates: list[_TrainingShapeCandidate],
    *,
    request: ShapeSelectionRequest,
    expected_mode: str,
) -> list[_TrainingShapeCandidate]:
    compatible: list[_TrainingShapeCandidate] = []
    requested_deployment = request.explicit_deployment_shape
    normalized_requested_deployment = _strip_version_suffix(requested_deployment)
    for candidate in candidates:
        if candidate.trainer_mode != expected_mode:
            continue
        if request.max_seq_len is not None:
            if candidate.max_supported_context_length < request.max_seq_len:
                continue
        if requested_deployment:
            candidate_deployment = candidate.deployment_shape_version
            if not candidate_deployment:
                continue
            if candidate_deployment != requested_deployment and (
                _strip_version_suffix(candidate_deployment) != normalized_requested_deployment
            ):
                continue
        compatible.append(candidate)
    return compatible


def _choose_training_shape_candidate(
    candidates: list[_TrainingShapeCandidate],
    request: ShapeSelectionRequest,
) -> _TrainingShapeCandidate:
    if request.max_seq_len is None:
        return candidates[0]
    indexed = list(enumerate(candidates))
    index, candidate = min(
        indexed,
        key=lambda item: (
            item[1].max_supported_context_length,
            item[0],
        ),
    )
    _ = index
    return candidate


def _fetch_model_selection_context(
    client: _RestCapable,
    base_model: str,
) -> _ModelSelectionContext:
    resp = client._get(f"/v1/{base_model}", timeout=30)
    if not resp.is_success:
        raise RuntimeError(
            f"Failed to fetch base model details for {base_model!r} "
            f"(HTTP {resp.status_code})"
        )
    data = resp.json() or {}
    details = _get_mapping(data, "baseModelDetails", "base_model_details") or {}

    model_type = _get_str(details, "modelType", "model_type") or _get_str(
        data, "modelType", "model_type"
    )
    parameter_count = _get_int(details, "parameterCount", "parameter_count")
    if parameter_count == 0:
        parameter_count = _get_int(data, "parameterCount", "parameter_count")
    supports_fireattention = _get_bool(
        details,
        "supportsFireattention",
        "supports_fireattention",
    )
    if model_type is None or parameter_count <= 0:
        raise ValueError(
            f"Base model {base_model!r} is missing model_type or parameter_count"
        )
    return _ModelSelectionContext(
        model_type=model_type,
        parameter_count=parameter_count,
        supports_fireattention=supports_fireattention,
    )


def _list_training_shape_candidates(
    client: _RestCapable,
    filter_expr: str,
) -> list[_TrainingShapeCandidate]:
    versions = _list_paginated_resources(
        client,
        parent=_TRAINING_SHAPE_VERSION_PARENT,
        collection_key="trainingShapeVersions",
        filter_expr=filter_expr,
    )
    return [_parse_training_shape_candidate(version) for version in versions]


def _list_deployment_shape_candidates(
    client: _RestCapable,
    filter_expr: str,
) -> list[_DeploymentShapeCandidate]:
    versions = _list_paginated_resources(
        client,
        parent=_DEPLOYMENT_SHAPE_VERSION_PARENT,
        collection_key="deploymentShapeVersions",
        filter_expr=filter_expr,
    )
    return [_parse_deployment_shape_candidate(version) for version in versions]


def _list_paginated_resources(
    client: _RestCapable,
    *,
    parent: str,
    collection_key: str,
    filter_expr: str,
) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "filter": filter_expr,
            "orderBy": _ORDER_BY_CREATE_TIME_DESC,
            "pageSize": 200,
        }
        if page_token:
            params["pageToken"] = page_token
        path = f"/v1/{parent}/versions?{urlencode(params)}"
        resp = client._get(path, timeout=30)
        if not resp.is_success:
            raise RuntimeError(
                f"Failed to list shape versions for parent={parent!r} "
                f"(HTTP {resp.status_code})"
            )
        data = resp.json() or {}
        page_items = data.get(collection_key, []) or data.get(
            _snake_case(collection_key), []
        ) or []
        resources.extend(page_items)
        page_token = _get_str(data, "nextPageToken", "next_page_token")
        if not page_token:
            break
    return resources


def _parse_training_shape_candidate(version: dict[str, Any]) -> _TrainingShapeCandidate:
    snapshot = _get_mapping(version, "snapshot") or {}
    sharding = _get_mapping(
        snapshot,
        "trainerShardingScheme",
        "trainer_sharding_scheme",
    ) or {}
    training_shape_version = _get_str(version, "name") or ""
    return _TrainingShapeCandidate(
        training_shape_version=training_shape_version,
        training_shape=_strip_version_suffix(training_shape_version) or training_shape_version,
        base_model=_get_str(snapshot, "baseModel", "base_model"),
        model_type=_get_str(snapshot, "modelType", "model_type"),
        parameter_count=_get_optional_int(snapshot, "parameterCount", "parameter_count"),
        trainer_mode=_normalize_trainer_mode(_get_value(snapshot, "trainerMode", "trainer_mode")),
        trainer_image_tag=_get_str(snapshot, "trainerImageTag", "trainer_image_tag") or "",
        max_supported_context_length=_get_int(
            snapshot,
            "maxSupportedContextLength",
            "max_supported_context_length",
        ),
        node_count=max(_get_int(snapshot, "nodeCount", "node_count"), 1),
        deployment_shape_version=_get_str(
            snapshot,
            "deploymentShapeVersion",
            "deployment_shape_version",
        ) or "",
        deployment_image_tag=_get_str(
            snapshot,
            "deploymentImageTag",
            "deployment_image_tag",
        ) or "",
        accelerator_type=_get_str(snapshot, "acceleratorType", "accelerator_type") or "",
        accelerator_count=_get_int(snapshot, "acceleratorCount", "accelerator_count"),
        base_model_weight_precision=_get_str(
            snapshot,
            "baseModelWeightPrecision",
            "base_model_weight_precision",
        ) or "",
        pipeline_parallelism=max(
            _get_int(sharding, "pipelineParallelism", "pipeline_parallelism"),
            1,
        ),
    )


def _parse_deployment_shape_candidate(version: dict[str, Any]) -> _DeploymentShapeCandidate:
    snapshot = _get_mapping(version, "snapshot") or {}
    deployment_shape_version = _get_str(version, "name") or ""
    return _DeploymentShapeCandidate(
        deployment_shape_version=deployment_shape_version,
        deployment_shape=_strip_version_suffix(deployment_shape_version) or deployment_shape_version,
        base_model=_get_str(snapshot, "baseModel", "base_model"),
        model_type=_get_str(snapshot, "modelType", "model_type"),
        parameter_count=_get_optional_int(snapshot, "parameterCount", "parameter_count"),
        accelerator_type=_get_str(snapshot, "acceleratorType", "accelerator_type") or "",
        accelerator_count=_get_int(snapshot, "acceleratorCount", "accelerator_count"),
        engine=_get_str(snapshot, "engine"),
    )


def _expected_trainer_mode(
    trainer_role: Literal["policy", "reference"],
    lora_rank: int,
) -> str:
    if trainer_role == "reference":
        return "FORWARD_ONLY"
    return "LORA_TRAINER" if lora_rank > 0 else "POLICY_TRAINER"


def _build_latest_validated_training_shape_filter(
    *,
    base_model: str,
    trainer_mode: str,
    deployment_shape: str | None,
) -> str:
    extras = [f'snapshot.trainer_mode="{trainer_mode}"']
    if deployment_shape:
        extras.extend(_deployment_shape_filters(deployment_shape))
    return _combine_filters(
        f'snapshot.base_model="{base_model}"',
        "latest_validated=true",
        *extras,
    )


def _build_compatible_training_shape_filter(
    *,
    model_ctx: _ModelSelectionContext,
    trainer_mode: str,
    deployment_shape: str | None,
) -> str:
    lower_bound, upper_bound = _get_parameter_count_bucket_bounds(model_ctx.parameter_count)
    extras = [f'snapshot.trainer_mode="{trainer_mode}"']
    if deployment_shape:
        extras.extend(_deployment_shape_filters(deployment_shape))
    return _combine_filters(
        f'snapshot.model_type="{model_ctx.model_type}"',
        f"snapshot.parameter_count>={lower_bound}",
        f"snapshot.parameter_count<={upper_bound}",
        "latest_validated=true",
        *extras,
    )


def _build_latest_validated_deployment_shape_filter(
    *,
    base_model: str,
    supports_fireattention: bool,
) -> str:
    return _combine_filters(
        f'snapshot.base_model="{base_model}"',
        "latest_validated=true",
        _deployment_engine_filter(supports_fireattention),
    )


def _build_compatible_deployment_shape_filter(
    model_ctx: _ModelSelectionContext,
) -> str:
    lower_bound, upper_bound = _get_parameter_count_bucket_bounds(model_ctx.parameter_count)
    return _combine_filters(
        f'snapshot.model_type="{model_ctx.model_type}"',
        f"snapshot.parameter_count>={lower_bound}",
        f"snapshot.parameter_count<={upper_bound}",
        "latest_validated=true",
        _deployment_engine_filter(model_ctx.supports_fireattention),
    )


def _deployment_shape_filters(deployment_shape: str) -> tuple[str, ...]:
    if "/versions/" in deployment_shape:
        return (f'snapshot.deployment_shape_version="{deployment_shape}"',)
    return ()


def _deployment_engine_filter(supports_fireattention: bool) -> str:
    if supports_fireattention:
        return 'snapshot.engine="FIREATTENTION"'
    return 'snapshot.engine!="FIREATTENTION"'


def _combine_filters(*parts: str) -> str:
    return " AND ".join(part for part in parts if part)


def _get_parameter_count_bucket_bounds(parameter_count: int) -> tuple[int, int]:
    one_b = 1_000_000_000
    ten_b = 10 * one_b
    bucket_size = one_b if parameter_count < ten_b else ten_b
    lower = (parameter_count // bucket_size) * bucket_size
    upper = lower + bucket_size
    return lower, upper


def _strip_version_suffix(resource_name: str | None) -> str | None:
    if not resource_name:
        return resource_name
    return _TRAINING_SHAPE_VERSION_RE.sub("", resource_name)


def _normalize_trainer_mode(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return _TRAINER_MODE_BY_CODE.get(int(value))


def _format_training_shape_selection_error(
    request: ShapeSelectionRequest,
    trainer_mode: str,
) -> str:
    mode_label = "LoRA" if trainer_mode == "LORA_TRAINER" else (
        "reference" if trainer_mode == "FORWARD_ONLY" else "full-tune"
    )
    suffix = (
        " Provide explicit shape overrides or lower max_seq_len."
        if request.max_seq_len is not None
        else " Provide explicit shape overrides."
    )
    return (
        "No validated training shape matched "
        f"base_model={request.base_model!r}, "
        f"trainer_role={request.trainer_role!r}, "
        f"mode={mode_label!r}, "
        f"max_seq_len={request.max_seq_len!r}, "
        f"needs_deployment={request.needs_deployment!r}."
        + suffix
    )


def _get_mapping(data: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    value = _get_value(data, *keys)
    return value if isinstance(value, dict) else None


def _get_str(data: dict[str, Any], *keys: str) -> str | None:
    value = _get_value(data, *keys)
    if value is None:
        return None
    return str(value)


def _get_int(data: dict[str, Any], *keys: str) -> int:
    value = _get_value(data, *keys)
    if value in (None, ""):
        return 0
    return int(value)


def _get_optional_int(data: dict[str, Any], *keys: str) -> int | None:
    value = _get_value(data, *keys)
    if value in (None, ""):
        return None
    return int(value)


def _get_bool(data: dict[str, Any], *keys: str) -> bool:
    value = _get_value(data, *keys)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _get_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _snake_case(name: str) -> str:
    chars: list[str] = []
    for char in name:
        if char.isupper():
            chars.append("_")
            chars.append(char.lower())
        else:
            chars.append(char)
    return "".join(chars)
