"""Capabilities for rendering historical assistant thinking traces.

The public API expresses one model-independent semantic dimension:

* ``INTERLEAVED`` removes thinking across user-turn boundaries while retaining
  the assistant -> tool -> assistant trajectory within the current turn.
* ``PRESERVED`` retains thinking from every prior turn.

Vendor-specific switches such as ``clear_thinking`` and
``preserve_thinking`` are implemented by the concrete renderer variants.  A
caller therefore selects a semantic mode here and never forwards a shared
boolean to arbitrary renderers.

Only explicitly onboarded aliases are classified.  Unknown/legacy models keep
their existing renderer path for ``UNSPECIFIED`` and reject explicit modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ThinkingTraceHistoryMode(StrEnum):
    """How earlier assistant thinking appears in later training contexts."""

    UNSPECIFIED = "unspecified"
    INTERLEAVED = "interleaved"
    PRESERVED = "preserved"


class UnsupportedThinkingTraceHistoryModeError(ValueError):
    """The requested semantic mode is not onboarded for this model alias."""


@dataclass(frozen=True)
class ReviewedTokenizerPlan:
    """Immutable tokenizer implementation reviewed for one model family."""

    repo: str
    revision: str
    trust_remote_code: bool


@dataclass(frozen=True)
class ThinkingTraceRendererPlan:
    """One semantic history mode and its concrete renderer implementation."""

    mode: ThinkingTraceHistoryMode
    renderer_name: str
    is_default: bool
    # Static fallback for control-plane display/error paths.  The concrete
    # renderer's ``has_extension_property`` remains authoritative at runtime.
    unrolls_multi_turn: bool | None


@dataclass(frozen=True)
class ThinkingTraceModelCapability:
    """Manually onboarded capability contract for one logical model family."""

    canonical_family: str
    aliases: frozenset[str]
    plans: tuple[ThinkingTraceRendererPlan, ...]
    tokenizer: ReviewedTokenizerPlan


@dataclass(frozen=True)
class ResolvedThinkingTraceRendererPlan:
    """Result of resolving a request against the immutable registry."""

    requested_mode: ThinkingTraceHistoryMode
    effective_mode: ThinkingTraceHistoryMode
    renderer_name: str
    unrolls_multi_turn: bool | None
    canonical_family: str | None


@dataclass(frozen=True)
class ResolvedTokenizerPlan:
    """Tokenizer identity materialized once for one training workflow run.

    ``trust_remote_code`` is ``None`` for an unregistered model so existing
    cookbook callers retain their legacy loading policy. Registered models
    always carry an explicit reviewed policy and immutable revision.
    """

    tokenizer_model: str
    tokenizer_revision: str
    trust_remote_code: bool | None
    canonical_family: str | None


def _plan(
    mode: ThinkingTraceHistoryMode,
    renderer_name: str,
    *,
    is_default: bool,
    unrolls_multi_turn: bool,
) -> ThinkingTraceRendererPlan:
    return ThinkingTraceRendererPlan(
        mode=mode,
        renderer_name=renderer_name,
        is_default=is_default,
        unrolls_multi_turn=unrolls_multi_turn,
    )


def _tokenizer(
    repo: str,
    revision: str,
    *,
    trust_remote_code: bool,
) -> ReviewedTokenizerPlan:
    return ReviewedTokenizerPlan(
        repo=repo,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


_CAPABILITIES: tuple[ThinkingTraceModelCapability, ...] = (
    ThinkingTraceModelCapability(
        canonical_family="glm5.1",
        aliases=frozenset(
            {
                "zai-org/glm-5.1",
                "zai-org/glm-5.1-fp8",
                "accounts/fireworks/models/glm-5p1",
                "accounts/fireworks/models/glm-5p1-bf16",
            }
        ),
        plans=(
            # GLM exposes ``clear_thinking``: INTERLEAVED -> True.
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "glm5_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
            # PRESERVED -> clear_thinking=False. GLM 5.1 remains non-extending.
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "glm5_preserve_thinking",
                is_default=False,
                unrolls_multi_turn=True,
            ),
        ),
        tokenizer=_tokenizer(
            "zai-org/GLM-5.1",
            "26e1bd6e011feb778d25ae34b09b07074139d92d",
            trust_remote_code=False,
        ),
    ),
    ThinkingTraceModelCapability(
        canonical_family="glm5.2",
        aliases=frozenset(
            {
                "zai-org/glm-5.2",
                "zai-org/glm-5.2-fp8",
                "accounts/fireworks/models/glm-5p2",
                "accounts/fireworks/models/glm-5p2-fp8",
            }
        ),
        plans=(
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "glm_moe_dsa_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "glm_moe_dsa_preserve_thinking",
                is_default=False,
                unrolls_multi_turn=False,
            ),
        ),
        tokenizer=_tokenizer(
            "zai-org/GLM-5.2",
            "b4734de4facf877f85769a911abafc5283eab3d9",
            trust_remote_code=False,
        ),
    ),
    ThinkingTraceModelCapability(
        canonical_family="qwen3.5",
        aliases=frozenset(
            {
                "qwen/qwen3.5-2b",
                "qwen/qwen3.5-4b",
                "qwen/qwen3.5-9b",
                "qwen/qwen3.5-27b",
                "qwen/qwen3.5-35b-a3b",
                "qwen/qwen3.5-397b-a17b",
                "qwen/qwen3.5-vl-8b-instruct",
                "accounts/fireworks/models/qwen3p5-4b",
                "accounts/fireworks/models/qwen3p5-9b",
                "accounts/fireworks/models/qwen3p5-27b",
                "accounts/fireworks/models/qwen3p5-35b-a3b",
                "accounts/fireworks/models/qwen3p5-397b-a17b",
            }
        ),
        plans=(
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "qwen3_5_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
        ),
        tokenizer=_tokenizer(
            "Qwen/Qwen3.5-35B-A3B",
            "59d61f3ce65a6d9863b86d2e96597125219dc754",
            trust_remote_code=False,
        ),
    ),
    ThinkingTraceModelCapability(
        canonical_family="qwen3.6",
        aliases=frozenset(
            {
                "qwen/qwen3.6-9b",
                "qwen/qwen3.6-27b",
                "qwen/qwen3.6-35b-a3b",
                "qwen/qwen3.6-vl-8b-instruct",
                "accounts/fireworks/models/qwen3p6-27b",
                "accounts/fireworks/models/qwen3p6-35b-a3b",
            }
        ),
        plans=(
            # Qwen 3.6 exposes ``preserve_thinking``: INTERLEAVED -> False.
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "qwen3_6_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
            # PRESERVED -> preserve_thinking=True.
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "qwen3_6_preserved",
                is_default=False,
                unrolls_multi_turn=False,
            ),
        ),
        tokenizer=_tokenizer(
            "Qwen/Qwen3.6-27B",
            "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            trust_remote_code=False,
        ),
    ),
    ThinkingTraceModelCapability(
        canonical_family="kimi-k2.5",
        aliases=frozenset(
            {
                "moonshotai/kimi-k2.5",
                "accounts/fireworks/models/kimi-k2p5",
                "accounts/fireworks/models/kimi-k2p5-vl",
            }
        ),
        plans=(
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "kimi_k25_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
        ),
        tokenizer=_tokenizer(
            "moonshotai/Kimi-K2.5",
            "4d01dfe0332d63057c186e0b262165819efb6611",
            trust_remote_code=True,
        ),
    ),
    ThinkingTraceModelCapability(
        canonical_family="kimi-k2.6",
        aliases=frozenset(
            {
                "moonshotai/kimi-k2.6",
                "accounts/fireworks/models/kimi-k2p6",
            }
        ),
        plans=(
            # Kimi K2.6 exposes ``preserve_thinking``: INTERLEAVED -> False.
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "kimi_k26_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
            # PRESERVED -> preserve_thinking=True.
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "kimi_k26_preserve_thinking",
                is_default=False,
                unrolls_multi_turn=False,
            ),
        ),
        tokenizer=_tokenizer(
            "moonshotai/Kimi-K2.6",
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
            trust_remote_code=True,
        ),
    ),
    ThinkingTraceModelCapability(
        canonical_family="kimi-k2.7-code",
        aliases=frozenset(
            {
                "moonshotai/kimi-k2.7-code",
                "accounts/fireworks/models/kimi-k2p7-code",
            }
        ),
        plans=(
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "kimi_k27_code_preserved",
                is_default=True,
                unrolls_multi_turn=False,
            ),
        ),
        tokenizer=_tokenizer(
            "moonshotai/Kimi-K2.7-Code",
            "74797c9c62378b951a1f6fcf5c4631024e9b8bef",
            trust_remote_code=True,
        ),
    ),
)


def _normalize_alias(value: str) -> str:
    return value.strip().rstrip("/").lower()


def _build_alias_index() -> dict[str, ThinkingTraceModelCapability]:
    index: dict[str, ThinkingTraceModelCapability] = {}
    for capability in _CAPABILITIES:
        if not capability.plans:
            raise ValueError(f"{capability.canonical_family} has no renderer plans")
        defaults = [plan for plan in capability.plans if plan.is_default]
        if len(defaults) != 1:
            raise ValueError(
                f"{capability.canonical_family} must have exactly one default plan"
            )
        tokenizer = capability.tokenizer
        if not tokenizer.repo or len(tokenizer.revision) != 40:
            raise ValueError(
                f"{capability.canonical_family} must pin one immutable tokenizer"
            )
        try:
            int(tokenizer.revision, 16)
        except ValueError as exc:
            raise ValueError(
                f"{capability.canonical_family} has an invalid tokenizer revision"
            ) from exc
        modes = [plan.mode for plan in capability.plans]
        if ThinkingTraceHistoryMode.UNSPECIFIED in modes or len(modes) != len(
            set(modes)
        ):
            raise ValueError(
                f"{capability.canonical_family} has invalid or duplicate modes"
            )
        for alias in capability.aliases:
            normalized = _normalize_alias(alias)
            previous = index.get(normalized)
            if previous is not None:
                raise ValueError(
                    f"Model alias {alias!r} belongs to both "
                    f"{previous.canonical_family} and {capability.canonical_family}"
                )
            index[normalized] = capability
        normalized_tokenizer_repo = _normalize_alias(tokenizer.repo)
        if normalized_tokenizer_repo not in {
            _normalize_alias(alias) for alias in capability.aliases
        }:
            raise ValueError(
                f"{capability.canonical_family} tokenizer repo must be a registered alias"
            )
    return index


_CAPABILITIES_BY_ALIAS = _build_alias_index()


def _build_renderer_index() -> dict[str, ThinkingTraceModelCapability]:
    """Index concrete registry plans by their persisted renderer names.

    Renderer names are durable Managed Training state, so ownership must be
    exact and unambiguous. In particular, a resume must not infer tokenizer
    ownership from a model alias that may now point at a different family.
    Legacy concrete names remain registered with the renderer implementation,
    but are deliberately absent here unless they are an active capability
    plan: an unknown/retired name safely falls back to the legacy tokenizer
    source instead of being guessed from a substring.
    """

    index: dict[str, ThinkingTraceModelCapability] = {}
    for capability in _CAPABILITIES:
        for plan in capability.plans:
            name = plan.renderer_name.strip()
            if not name:
                raise ValueError(
                    f"{capability.canonical_family} has an empty renderer name"
                )
            previous = index.get(name)
            if previous is not None:
                raise ValueError(
                    f"Renderer {name!r} belongs to both "
                    f"{previous.canonical_family} and {capability.canonical_family}"
                )
            index[name] = capability
    return index


_CAPABILITIES_BY_RENDERER = _build_renderer_index()


def normalize_thinking_trace_history_mode(
    value: str | ThinkingTraceHistoryMode | None,
) -> ThinkingTraceHistoryMode:
    """Normalize cookbook strings and generated Protobuf enum names."""

    if value is None or value == "":
        return ThinkingTraceHistoryMode.UNSPECIFIED
    if isinstance(value, ThinkingTraceHistoryMode):
        return value
    normalized = value.strip().lower()
    prefix = "thinking_trace_history_mode_"
    if normalized.startswith(prefix):
        normalized = normalized.removeprefix(prefix)
    try:
        return ThinkingTraceHistoryMode(normalized)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in ThinkingTraceHistoryMode)
        raise ValueError(
            f"Unknown thinking_trace_history_mode={value!r}; expected: {allowed}."
        ) from exc


def get_thinking_trace_model_capability(
    tokenizer_model: str,
) -> ThinkingTraceModelCapability | None:
    """Return an explicitly onboarded family; never infer from substrings."""

    return _CAPABILITIES_BY_ALIAS.get(_normalize_alias(tokenizer_model))


def get_thinking_trace_model_capability_for_renderer(
    renderer_name: str,
) -> ThinkingTraceModelCapability | None:
    """Return the family that owns an exact, persistable renderer name."""

    return _CAPABILITIES_BY_RENDERER.get(renderer_name.strip())


def _resolved_tokenizer_plan(
    capability: ThinkingTraceModelCapability | None,
    resolved_tokenizer_model: str,
) -> ResolvedTokenizerPlan:
    source = resolved_tokenizer_model.strip()
    if not source:
        raise ValueError("resolved_tokenizer_model is required")
    if capability is None:
        return ResolvedTokenizerPlan(
            tokenizer_model=source,
            tokenizer_revision="",
            trust_remote_code=None,
            canonical_family=None,
        )

    tokenizer = capability.tokenizer
    return ResolvedTokenizerPlan(
        tokenizer_model=tokenizer.repo,
        tokenizer_revision=tokenizer.revision,
        trust_remote_code=tokenizer.trust_remote_code,
        canonical_family=capability.canonical_family,
    )


def resolve_tokenizer_plan(
    model_alias: str,
    resolved_tokenizer_model: str,
) -> ResolvedTokenizerPlan:
    """Materialize one safe tokenizer identity for a workflow execution.

    The Go control plane remains authoritative for resolving legacy/customer
    model metadata. Once a model family is explicitly registered here, however,
    its tokenizer is part of the reviewed renderer contract. Catalog storage may
    legitimately resolve to a staged GCS directory, so registered aliases always
    materialize the canonical reviewed Hugging Face repo and immutable revision.
    Unregistered aliases preserve the control-plane source and legacy policy.
    """

    return _resolved_tokenizer_plan(
        get_thinking_trace_model_capability(model_alias),
        resolved_tokenizer_model,
    )


def resolve_tokenizer_plan_for_renderer(
    renderer_name: str,
    resolved_tokenizer_model: str,
) -> ResolvedTokenizerPlan:
    """Materialize a tokenizer from the owner of a persisted renderer.

    A concrete renderer is authoritative on retry/resume. If its name still
    belongs to an onboarded capability, use that family's reviewed tokenizer
    even when the current model alias resolves elsewhere. Unknown legacy
    renderer names retain the control-plane tokenizer source and loading
    policy.
    """

    name = renderer_name.strip()
    if not name:
        raise ValueError("renderer_name is required")
    return _resolved_tokenizer_plan(
        get_thinking_trace_model_capability_for_renderer(name),
        resolved_tokenizer_model,
    )


def thinking_trace_renderer_plans(
    tokenizer_model: str,
    *,
    default_renderer_name: str,
) -> tuple[ThinkingTraceRendererPlan, ...]:
    """Return selectable plans, or one legacy UNSPECIFIED fallback plan."""

    capability = get_thinking_trace_model_capability(tokenizer_model)
    if capability is not None:
        return capability.plans
    return (
        ThinkingTraceRendererPlan(
            mode=ThinkingTraceHistoryMode.UNSPECIFIED,
            renderer_name=default_renderer_name,
            is_default=True,
            unrolls_multi_turn=None,
        ),
    )


def resolve_thinking_trace_renderer_plan(
    tokenizer_model: str,
    *,
    requested_mode: str | ThinkingTraceHistoryMode | None,
    default_renderer_name: str,
) -> ResolvedThinkingTraceRendererPlan:
    """Resolve a semantic request or reject an unsupported explicit mode."""

    requested = normalize_thinking_trace_history_mode(requested_mode)
    capability = get_thinking_trace_model_capability(tokenizer_model)
    plans = thinking_trace_renderer_plans(
        tokenizer_model,
        default_renderer_name=default_renderer_name,
    )
    if requested is ThinkingTraceHistoryMode.UNSPECIFIED:
        plan = next(plan for plan in plans if plan.is_default)
    else:
        plan = next((plan for plan in plans if plan.mode is requested), None)
        if plan is None:
            supported = [
                candidate.mode.value
                for candidate in plans
                if candidate.mode is not ThinkingTraceHistoryMode.UNSPECIFIED
            ]
            supported_text = ", ".join(supported) if supported else "default only"
            raise UnsupportedThinkingTraceHistoryModeError(
                f"thinking_trace_history_mode={requested.value!r} is not supported "
                f"for tokenizer_model={tokenizer_model!r}; supported modes: "
                f"{supported_text}."
            )
    return ResolvedThinkingTraceRendererPlan(
        requested_mode=requested,
        effective_mode=plan.mode,
        renderer_name=plan.renderer_name,
        unrolls_multi_turn=plan.unrolls_multi_turn,
        canonical_family=(
            capability.canonical_family if capability is not None else None
        ),
    )


def renderer_unrolls_multi_turn(renderer: object) -> bool:
    """Whether ALL_ASSISTANT_MESSAGES needs one datum per user turn."""

    return not bool(getattr(renderer, "has_extension_property", False))


def iter_thinking_trace_capabilities() -> tuple[ThinkingTraceModelCapability, ...]:
    """Expose immutable registry contents to validation and preview callers."""

    return _CAPABILITIES
