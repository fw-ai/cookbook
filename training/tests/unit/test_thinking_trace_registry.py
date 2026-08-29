from __future__ import annotations

from typing import Any

import pytest
import training.renderer  # noqa: F401  (installs local registrations)
from tinker_cookbook.renderers import get_renderer
from training.renderer.thinking_trace import (
    ThinkingTraceHistoryMode,
    get_thinking_trace_model_capability,
    iter_thinking_trace_capabilities,
    normalize_thinking_trace_history_mode,
    renderer_unrolls_multi_turn,
    resolve_thinking_trace_renderer_plan,
    thinking_trace_renderer_plans,
)
from training.utils.supervised import resolve_renderer_plan


class _StringTokenizer:
    """Minimal reversible tokenizer for renderer capability construction."""

    name_or_path = "test-tokenizer"

    def __init__(self) -> None:
        self._text_to_id: dict[str, int] = {}
        self._id_to_text: dict[int, str] = {}

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        if text not in self._text_to_id:
            token_id = len(self._text_to_id) + 1
            self._text_to_id[text] = token_id
            self._id_to_text[token_id] = text
        return [self._text_to_id[text]]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self._id_to_text[token_id] for token_id in token_ids)


@pytest.mark.parametrize(
    ("model", "family", "modes", "default_renderer", "unrolls"),
    [
        (
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "deepseek-v4",
            ["unspecified"],
            "deepseek_v4",
            [True],
        ),
        (
            "zai-org/GLM-5.1",
            "glm5.1",
            ["interleaved", "preserved"],
            "glm5_interleaved",
            [True, True],
        ),
        (
            "zai-org/GLM-5.2",
            "glm5.2",
            ["interleaved", "preserved"],
            "glm_moe_dsa_interleaved",
            [True, False],
        ),
        (
            "zai-org/GLM-5.3",
            "glm5.3",
            ["preserved", "interleaved"],
            "glm53_preserve_thinking",
            [False, True],
        ),
        (
            "Qwen/Qwen3.5-35B-A3B",
            "qwen3.5",
            ["interleaved"],
            "qwen3_5_interleaved",
            [True],
        ),
        (
            "Qwen/Qwen3.6-27B",
            "qwen3.6",
            ["interleaved", "preserved"],
            "qwen3_6_interleaved",
            [True, False],
        ),
        (
            "Qwen/Qwen3.8-27B",
            "qwen3.8",
            ["preserved", "interleaved"],
            "qwen3_8_preserved",
            [False, True],
        ),
        (
            "moonshotai/Kimi-K2.5",
            "kimi-k2.5",
            ["interleaved"],
            "kimi_k25_interleaved",
            [True],
        ),
        (
            "moonshotai/Kimi-K2.6",
            "kimi-k2.6",
            ["interleaved", "preserved"],
            "kimi_k26_interleaved",
            [True, False],
        ),
        (
            "moonshotai/Kimi-K2.7-Code",
            "kimi-k2.7-code",
            ["preserved"],
            "kimi_k27_code_preserved",
            [False],
        ),
        (
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            "nemotron3-ultra",
            ["interleaved", "preserved"],
            "nemotron3_ultra_interleaved",
            [True, False],
        ),
    ],
)
def test_registered_model_matrix(
    model: str,
    family: str,
    modes: list[str],
    default_renderer: str,
    unrolls: list[bool],
) -> None:
    capability = get_thinking_trace_model_capability(model)
    assert capability is not None
    assert capability.canonical_family == family

    plans = thinking_trace_renderer_plans(
        model,
        default_renderer_name="unused-legacy-renderer",
    )
    assert [plan.mode.value for plan in plans] == modes
    assert [plan.renderer_name for plan in plans if plan.is_default] == [
        default_renderer
    ]
    assert [plan.unrolls_multi_turn for plan in plans] == unrolls


def test_every_declared_renderer_matches_live_extension_property() -> None:
    tokenizer = _StringTokenizer()
    for capability in iter_thinking_trace_capabilities():
        for plan in capability.plans:
            renderer = get_renderer(plan.renderer_name, tokenizer)
            assert plan.unrolls_multi_turn == renderer_unrolls_multi_turn(renderer), (
                capability.canonical_family,
                plan.mode,
                plan.renderer_name,
            )


def test_legacy_concrete_names_are_not_rebound_to_corrected_adapters() -> None:
    tokenizer = _StringTokenizer()
    expected_types = {
        # ``*SplitRenderer`` keeps the legacy name's rendering and only adds
        # weight-aware per-user-turn unrolling, the same shape ``qwen3_5`` has
        # had. A rebind to a corrected adapter would change the rendered tokens;
        # these do not.
        "kimi_k25": "KimiK25SplitRenderer",
        "qwen3_5": "Qwen3_5SplitRenderer",
        "qwen3_6": "Qwen3_6SplitRenderer",
        "qwen3_6_preserve_thinking": "Qwen3_6PreserveThinkingSplitRenderer",
        "qwen3_8": "Qwen3_8PreservedRenderer",
        "qwen3_8_preserved": "Qwen3_8PreservedRenderer",
        "qwen3_8_interleaved": "Qwen3_8InterleavedRenderer",
        "qwen3_8_disable_thinking_interleaved": (
            "Qwen3_8DisableThinkingInterleavedRenderer"
        ),
        "glm5": "GLM5Renderer",
        "glm_moe_dsa": "GLMMoeDsaRenderer",
        "glm53": "GLM53Renderer",
        "kimi_k27_code": "KimiK27CodeRenderer",
    }

    assert {
        name: type(get_renderer(name, tokenizer)).__name__ for name in expected_types
    } == expected_types

    assert get_renderer("glm5", tokenizer)._honor_source_reasoning_fields is False
    assert (
        get_renderer("glm5_interleaved", tokenizer)._honor_source_reasoning_fields
        is True
    )
    assert (
        get_renderer("glm_moe_dsa", tokenizer)._honor_source_reasoning_fields is False
    )
    assert (
        get_renderer(
            "glm_moe_dsa_interleaved", tokenizer
        )._honor_source_reasoning_fields
        is True
    )
    assert get_renderer("glm53", tokenizer)._clear_thinking is False
    assert get_renderer("glm53", tokenizer)._honor_source_reasoning_fields is True


def test_registered_aliases_are_exact_and_case_insensitive() -> None:
    assert (
        get_thinking_trace_model_capability(
            "  ACCOUNTS/FIREWORKS/MODELS/QWEN3P6-27B/  "
        ).canonical_family
        == "qwen3.6"
    )
    assert (
        get_thinking_trace_model_capability(
            "accounts/fireworks/models/qwen3p8-27b"
        ).canonical_family
        == "qwen3.8"
    )
    assert (
        get_thinking_trace_model_capability("accounts/fireworks/models/glm-5p3")
        .canonical_family
        == "glm5.3"
    )

    # The legacy default resolver may understand custom/fine-tuned names, but
    # they are deliberately default-only until explicitly onboarded.
    assert get_thinking_trace_model_capability("custom/qwen3_6-finetune") is None


def test_unspecified_resolves_registered_default() -> None:
    glm = resolve_thinking_trace_renderer_plan(
        "zai-org/GLM-5.1",
        requested_mode=ThinkingTraceHistoryMode.UNSPECIFIED,
        default_renderer_name="wrong-default",
    )
    assert glm.requested_mode is ThinkingTraceHistoryMode.UNSPECIFIED
    assert glm.effective_mode is ThinkingTraceHistoryMode.INTERLEAVED
    assert glm.renderer_name == "glm5_interleaved"

    kimi = resolve_thinking_trace_renderer_plan(
        "moonshotai/Kimi-K2.7-Code",
        requested_mode=None,
        default_renderer_name="wrong-default",
    )
    assert kimi.effective_mode is ThinkingTraceHistoryMode.PRESERVED
    assert kimi.renderer_name == "kimi_k27_code_preserved"

    qwen38 = resolve_thinking_trace_renderer_plan(
        "Qwen/Qwen3.8-27B",
        requested_mode=None,
        default_renderer_name="wrong-default",
    )
    assert qwen38.effective_mode is ThinkingTraceHistoryMode.PRESERVED
    assert qwen38.renderer_name == "qwen3_8_preserved"


def test_unregistered_model_is_default_only() -> None:
    default = resolve_thinking_trace_renderer_plan(
        "meta-llama/Llama-3.1-8B-Instruct",
        requested_mode=None,
        default_renderer_name="llama-3",
    )
    assert default.effective_mode is ThinkingTraceHistoryMode.UNSPECIFIED
    assert default.renderer_name == "llama-3"
    assert default.canonical_family is None

    with pytest.raises(ValueError, match="supported modes: default only"):
        resolve_thinking_trace_renderer_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            requested_mode="preserved",
            default_renderer_name="llama-3",
        )


@pytest.mark.parametrize("mode", ["interleaved", "preserved"])
def test_deepseek_v4_rejects_explicit_history_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="supported modes: default only"):
        resolve_thinking_trace_renderer_plan(
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            requested_mode=mode,
            default_renderer_name="wrong-default",
        )


@pytest.mark.parametrize(
    ("model", "unsupported"),
    [
        ("Qwen/Qwen3.5-35B-A3B", "preserved"),
        ("moonshotai/Kimi-K2.5", "preserved"),
        ("moonshotai/Kimi-K2.7-Code", "interleaved"),
    ],
)
def test_fixed_mode_models_reject_the_other_mode(
    model: str,
    unsupported: str,
) -> None:
    with pytest.raises(ValueError, match="is not supported"):
        resolve_thinking_trace_renderer_plan(
            model,
            requested_mode=unsupported,
            default_renderer_name="unused",
        )


def test_public_semantics_select_vendor_specific_renderer_adapters() -> None:
    # GLM's vendor flag is inverted: PRESERVED means clear_thinking=False.
    glm_interleaved = resolve_renderer_plan(
        "zai-org/GLM-5.1",
        thinking_trace_history_mode="interleaved",
    )
    glm = resolve_renderer_plan(
        "zai-org/GLM-5.1",
        thinking_trace_history_mode="preserved",
    )
    assert glm_interleaved.renderer_name == "glm5_interleaved"
    assert glm.renderer_name == "glm5_preserve_thinking"

    # Qwen 3.6 and Kimi K2.6 use preserve_thinking directly:
    # INTERLEAVED -> False, PRESERVED -> True.
    qwen_interleaved = resolve_renderer_plan(
        "Qwen/Qwen3.6-27B",
        thinking_trace_history_mode="interleaved",
    )
    qwen = resolve_renderer_plan(
        "Qwen/Qwen3.6-27B",
        thinking_trace_history_mode="preserved",
    )
    kimi_interleaved = resolve_renderer_plan(
        "moonshotai/Kimi-K2.6",
        thinking_trace_history_mode="interleaved",
    )
    kimi = resolve_renderer_plan(
        "moonshotai/Kimi-K2.6",
        thinking_trace_history_mode="preserved",
    )
    assert qwen_interleaved.renderer_name == "qwen3_6_interleaved"
    assert qwen.renderer_name == "qwen3_6_preserved"
    assert kimi_interleaved.renderer_name == "kimi_k26_interleaved"
    assert kimi.renderer_name == "kimi_k26_preserve_thinking"

    # Nemotron-3: INTERLEAVED strips via truncate_history_thinking=True;
    # PRESERVED maps to truncate_history_thinking=False (serving #40028).
    nemotron_interleaved = resolve_renderer_plan(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        thinking_trace_history_mode="interleaved",
    )
    nemotron = resolve_renderer_plan(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        thinking_trace_history_mode="preserved",
    )
    ultra = resolve_renderer_plan(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        thinking_trace_history_mode="preserved",
    )
    assert nemotron_interleaved.renderer_name == "nemotron3_interleaved"
    assert nemotron.renderer_name == "nemotron3_preserved"
    assert ultra.renderer_name == "nemotron3_ultra_preserved"

    qwen38_interleaved = resolve_renderer_plan(
        "Qwen/Qwen3.8-27B",
        thinking_trace_history_mode="interleaved",
    )
    qwen38 = resolve_renderer_plan(
        "Qwen/Qwen3.8-27B",
        thinking_trace_history_mode="preserved",
    )
    assert qwen38_interleaved.renderer_name == "qwen3_8_interleaved"
    assert qwen38.renderer_name == "qwen3_8_preserved"


def test_unspecified_preserves_legacy_explicit_renderer_override() -> None:
    disabled = resolve_renderer_plan(
        "Qwen/Qwen3.6-27B",
        renderer_name="qwen3_6_disable_thinking",
    )
    assert disabled.requested_mode is ThinkingTraceHistoryMode.UNSPECIFIED
    assert disabled.effective_mode is ThinkingTraceHistoryMode.UNSPECIFIED
    assert disabled.renderer_name == "qwen3_6_disable_thinking"

    preserve = resolve_renderer_plan(
        "Qwen/Qwen3.6-27B",
        renderer_name="qwen3_6_preserve_thinking",
    )
    # Legacy concrete names are no longer reclassified as current registry
    # plans; direct callers still get exactly the requested renderer.
    assert preserve.effective_mode is ThinkingTraceHistoryMode.UNSPECIFIED


def test_explicit_mode_rejects_conflicting_explicit_renderer() -> None:
    with pytest.raises(ValueError, match="conflicts with"):
        resolve_renderer_plan(
            "Qwen/Qwen3.6-27B",
            renderer_name="qwen3_6_disable_thinking",
            thinking_trace_history_mode="preserved",
        )


def test_materialized_renderer_snapshot_remains_authoritative() -> None:
    from training.utils.supervised import resolve_renderer_snapshot

    assert (
        resolve_renderer_snapshot(
            tokenizer_model="Qwen/Qwen3.6-27B",
            renderer_name="qwen3_6_preserved",
            thinking_trace_history_mode="interleaved",
            renderer_name_is_resolved=True,
        )
        == "qwen3_6_preserved"
    )


@pytest.mark.parametrize(
    "raw_mode",
    [
        "",
        "unspecified",
        "UNSPECIFIED",
        "  unspecified  ",
        "THINKING_TRACE_HISTORY_MODE_UNSPECIFIED",
    ],
)
def test_snapshot_treats_equivalent_unspecified_spellings_as_legacy_default(
    raw_mode: str,
) -> None:
    from training.utils.supervised import resolve_renderer_snapshot

    assert (
        resolve_renderer_snapshot(
            tokenizer_model="Qwen/Qwen3.6-27B",
            renderer_name="",
            thinking_trace_history_mode=raw_mode,
        )
        == "qwen3_6"
    )


def test_direct_renderer_snapshot_rejects_conflicting_mode() -> None:
    from training.utils.supervised import resolve_renderer_snapshot

    with pytest.raises(ValueError, match="conflicts with"):
        resolve_renderer_snapshot(
            tokenizer_model="Qwen/Qwen3.6-27B",
            renderer_name="qwen3_6_preserved",
            thinking_trace_history_mode="interleaved",
        )


def test_resolved_provenance_requires_a_concrete_name() -> None:
    from training.utils.supervised import resolve_renderer_snapshot

    with pytest.raises(ValueError, match="requires a non-empty renderer_name"):
        resolve_renderer_snapshot(
            tokenizer_model="Qwen/Qwen3.6-27B",
            renderer_name="",
            thinking_trace_history_mode="preserved",
            renderer_name_is_resolved=True,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ThinkingTraceHistoryMode.UNSPECIFIED),
        ("", ThinkingTraceHistoryMode.UNSPECIFIED),
        ("INTERLEAVED", ThinkingTraceHistoryMode.INTERLEAVED),
        (
            "THINKING_TRACE_HISTORY_MODE_PRESERVED",
            ThinkingTraceHistoryMode.PRESERVED,
        ),
    ],
)
def test_mode_normalization(
    raw: str | None, expected: ThinkingTraceHistoryMode
) -> None:
    assert normalize_thinking_trace_history_mode(raw) is expected


@pytest.mark.parametrize("removed_mode", ["model_default", "clear", "preserve"])
def test_removed_mode_names_are_rejected(removed_mode: str) -> None:
    with pytest.raises(ValueError, match="Unknown thinking_trace_history_mode"):
        normalize_thinking_trace_history_mode(removed_mode)
