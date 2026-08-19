"""Capabilities for rendering historical assistant thinking traces.

The public API expresses one model-independent semantic dimension:

* ``INTERLEAVED`` removes thinking across user-turn boundaries while retaining
  the assistant -> tool -> assistant trajectory within the current turn.
* ``PRESERVED`` retains thinking from every prior turn.

Vendor-specific switches such as ``clear_thinking`` and
``preserve_thinking`` are implemented by the concrete renderer variants.  A
caller therefore selects a semantic mode here and never forwards a shared
boolean to arbitrary renderers.

Only explicitly onboarded aliases are classified. An onboarded model may expose
only ``UNSPECIFIED`` when its model-native behavior is automatic rather than a
stable user-selectable history mode. Unknown/legacy models also keep their
existing renderer path for ``UNSPECIFIED`` and reject explicit modes.
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


@dataclass(frozen=True)
class ResolvedThinkingTraceRendererPlan:
    """Result of resolving a request against the immutable registry."""

    requested_mode: ThinkingTraceHistoryMode
    effective_mode: ThinkingTraceHistoryMode
    renderer_name: str
    unrolls_multi_turn: bool | None
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


_CAPABILITIES: tuple[ThinkingTraceModelCapability, ...] = (
    ThinkingTraceModelCapability(
        canonical_family="deepseek-v4",
        aliases=frozenset(
            {
                "deepseek-ai/deepseek-v4-flash",
                "deepseek-ai/deepseek-v4-flash-0731",
                "deepseek-ai/deepseek-v4-pro",
                "sgl-project/deepseek-v4-flash-fp8",
                "sgl-project/deepseek-v4-pro-fp8",
                "accounts/fireworks/models/deepseek-v4-flash",
                "accounts/fireworks/models/deepseek-v4-flash-0731",
                "accounts/fireworks/models/deepseek-v4-pro",
                "accounts/fireworks/models/deepseek-v4-flash-fp8",
                "accounts/fireworks/models/deepseek-v4-pro-fp8",
            }
        ),
        plans=(
            # DeepSeek V4's encoder chooses per conversation: no tools strips
            # thinking across user turns; declaring tools preserves all
            # thinking. Neither explicit semantic mode describes both cases.
            _plan(
                ThinkingTraceHistoryMode.UNSPECIFIED,
                "deepseek_v4",
                is_default=True,
                # Conservatively advertise that AUTO may unroll. At runtime
                # tool-bearing rows take the one-sequence fast path.
                unrolls_multi_turn=True,
            ),
        ),
    ),
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
    ),
    ThinkingTraceModelCapability(
        canonical_family="qwen3.8",
        aliases=frozenset(
            {
                "qwen/qwen3.8-27b",
                "accounts/fireworks/models/qwen3p8-27b",
            }
        ),
        plans=(
            # Qwen3.8 HF default is preserve_thinking=true (undefined counts as
            # true). Keep PRESERVED as the cookbook default so SFT/RL match
            # serving with --conversation-style=qwen3p8.
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "qwen3_8_preserved",
                is_default=True,
                unrolls_multi_turn=False,
            ),
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "qwen3_8_interleaved",
                is_default=False,
                unrolls_multi_turn=True,
            ),
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
    ),
    ThinkingTraceModelCapability(
        canonical_family="nemotron3",
        aliases=frozenset(
            {
                "nvidia/nvidia-nemotron-3-super-120b-a12b-bf16",
                "nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16",
            }
        ),
        plans=(
            # HF default truncate_history_thinking=True → INTERLEAVED.
            _plan(
                ThinkingTraceHistoryMode.INTERLEAVED,
                "nemotron3_interleaved",
                is_default=True,
                unrolls_multi_turn=True,
            ),
            # PRESERVED → truncate_history_thinking=False (serving #40028).
            # Canonical registry name matches the Qwen/Kimi ``*_preserved``
            # pattern; ``nemotron3_preserve_thinking`` remains a registered
            # get_renderer alias for direct callers.
            _plan(
                ThinkingTraceHistoryMode.PRESERVED,
                "nemotron3_preserved",
                is_default=False,
                unrolls_multi_turn=False,
            ),
        ),
    ),
)


def _normalize_alias(value: str) -> str:
    return value.strip().rstrip("/").lower()


def _build_alias_index() -> dict[str, ThinkingTraceModelCapability]:
    index: dict[str, ThinkingTraceModelCapability] = {}
    renderer_names: set[str] = set()
    for capability in _CAPABILITIES:
        if not capability.plans:
            raise ValueError(f"{capability.canonical_family} has no renderer plans")
        defaults = [plan for plan in capability.plans if plan.is_default]
        if len(defaults) != 1:
            raise ValueError(
                f"{capability.canonical_family} must have exactly one default plan"
            )
        modes = [plan.mode for plan in capability.plans]
        if len(modes) != len(set(modes)):
            raise ValueError(
                f"{capability.canonical_family} has invalid or duplicate modes"
            )
        if ThinkingTraceHistoryMode.UNSPECIFIED in modes and modes != [
            ThinkingTraceHistoryMode.UNSPECIFIED
        ]:
            raise ValueError(
                f"{capability.canonical_family} must expose UNSPECIFIED alone"
            )
        for plan in capability.plans:
            renderer_name = plan.renderer_name.strip()
            if not renderer_name or renderer_name in renderer_names:
                raise ValueError(
                    f"{capability.canonical_family} has an empty or duplicate renderer"
                )
            renderer_names.add(renderer_name)
        for alias in capability.aliases:
            normalized = _normalize_alias(alias)
            previous = index.get(normalized)
            if previous is not None:
                raise ValueError(
                    f"Model alias {alias!r} belongs to both "
                    f"{previous.canonical_family} and {capability.canonical_family}"
                )
            index[normalized] = capability
    return index


_CAPABILITIES_BY_ALIAS = _build_alias_index()


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
