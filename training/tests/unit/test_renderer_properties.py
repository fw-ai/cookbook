"""Unified property-based QA harness for cookbook model renderers.

This is the CPU CI gate for renderer correctness. It mirrors the four
invariants upstream tinker asserts per renderer, but factors the renderer
set (:mod:`renderer_matrix`) and the conversation bank
(:mod:`renderer_scenarios`) into shared data so a new renderer joins CI by
adding a single :class:`~renderer_matrix.RendererCase` row.

The four invariants:

1. HF generation parity — ``build_generation_prompt`` tokens equal
   ``apply_chat_template(add_generation_prompt=True)`` for every
   generation-shaped scenario (the renderer<->upstream tokenization
   contract). Tool declarations are byte-compared too: the HF reference
   gets ``tools=`` and the renderer gets its own tool block (there is no
   ``hf_safe`` opt-out for tool or thinking scenarios).
2. HF supervised parity — ``build_supervised_example`` tokens equal
   ``apply_chat_template(add_generation_prompt=False)`` for renderers whose
   supervised header matches the generation header.
3. supervised<->generation<->parse consistency — the supervised weight mask
   is ``000...111``, its observation equals the generation prompt of the
   prefix (for renderers with ``observation_equals_generation``), and the
   trained action tokens parse back to a clean assistant turn whose text
   matches the scenario.
4. sequence extension — for renderers that claim it, each turn's full
   sequence is a prefix of the next turn's observation (KV-cache-safe).

Every test skips cleanly when the upstream tokenizer cannot be loaded
(network outage, gated repo, missing local checkpoint) so CI without HF Hub
access completes; the moment a tokenizer is present the invariant runs and
asserts. Known divergences are tracked per ``(renderer, scenario)`` in
:mod:`renderer_expected_divergences` and applied as
``pytest.mark.xfail(strict=True)``. Because the marks are strict, fixing a
divergence turns the xfail into an XPASS and fails the suite until the stale
entry is removed — the divergence maps can never silently rot. Set
``RENDERER_QA_STRICT=1`` (CI) to also fail when a REQUIRED public renderer's
tokenizer silently stops loading.
"""

from __future__ import annotations

import os
from functools import cache

import pytest

# Importing the cookbook renderer package registers cookbook-local renderer
# names (glm5, gemma4, minimax_m2, nemotron, deepseek_v4, mistral,
# kimi_k27_code, ...) under training._vendor.tinker_cookbook_0_4_3.renderers.get_renderer.
import training.renderer  # noqa: F401
from training.renderer import Renderer, get_renderer
from training.renderer.tokenizer import Tokenizer, get_tokenizer
from training.renderer.verifier.utils.hf_parity import (
    compare_renderer_to_hf,
    format_divergence,
)
from training.utils.tokenizers import load_tokenizer

from training.tests.unit.renderer_expected_divergences import (
    EXTENSION_EXPECTED_DIVERGENCES,
    HF_EXPECTED_DIVERGENCES,
    HISTORICAL_PARSE_EXPECTED_DIVERGENCES,
    OBSERVATION_EXPECTED_DIVERGENCES,
    PARSE_EXPECTED_DIVERGENCES,
    TEXT_EXPECTED_DIVERGENCES,
)
from training.tests.unit.renderer_matrix import (
    REQUIRED_RENDERERS,
    RENDERER_MATRIX,
    RendererCase,
)
from training.tests.unit.renderer_scenarios import ALL_SCENARIOS, Scenario
from training.tests.unit.renderer_contract import RendererContractTests, _thinking_parts

# Exceptions raised when an upstream tokenizer / renderer cannot be
# materialized (offline, gated repo, missing chat_template, or a tokenizer
# registration issue). These map to a clean skip, not a failure.
_UNAVAILABLE = (AttributeError, OSError, ValueError, RuntimeError)


def _shard_renderer_cases(
    cases: list[RendererCase], shard_index: int, shard_count: int
) -> list[RendererCase]:
    """Partition by tokenizer so variants share one load within a shard."""
    groups: dict[tuple[str, str | None, bool | None], list[RendererCase]] = {}
    for case in cases:
        key = (
            case.resolved_tokenizer_model(),
            case.resolved_tokenizer_revision(),
            case.tokenizer_trust_remote_code,
        )
        groups.setdefault(key, []).append(case)
    return [
        case
        for group_index, group in enumerate(groups.values())
        if group_index % shard_count == shard_index
        for case in group
    ]


def _selected_renderer_cases() -> list[RendererCase]:
    """Select an optional CI shard while keeping local runs exhaustive."""
    shard_index = int(os.environ.get("RENDERER_QA_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("RENDERER_QA_SHARD_COUNT", "1"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"invalid renderer QA shard {shard_index}; expected 0 <= index < {shard_count}"
        )

    # Assign every future matrix row automatically; adding a renderer still
    # requires only one matrix row.
    return _shard_renderer_cases(RENDERER_MATRIX, shard_index, shard_count)


_QA_CASES = _selected_renderer_cases()


@cache
def _load_tokenizer_result(
    model: str,
    revision: str | None,
    trust_remote_code: bool | None,
    explicit: bool,
) -> tuple[Tokenizer | None, BaseException | None]:
    try:
        tokenizer = (
            load_tokenizer(model, revision, trust_remote_code)
            if explicit
            else get_tokenizer(model)
        )
    except _UNAVAILABLE as exc:
        return None, exc
    return tokenizer, None


def _load_case_tokenizer(case: RendererCase) -> Tokenizer:
    """Cache both successful loads and unavailability for this QA process."""
    model = case.resolved_tokenizer_model()
    revision = case.resolved_tokenizer_revision()
    explicit = (
        case.tokenizer_revision is not None
        or case.tokenizer_trust_remote_code is not None
    )
    tokenizer, error = _load_tokenizer_result(
        model,
        revision,
        case.tokenizer_trust_remote_code,
        explicit,
    )
    if error is not None:
        raise error
    assert tokenizer is not None
    return tokenizer


def _load_renderer(case: RendererCase) -> tuple[Tokenizer, Renderer]:
    """Load the case's tokenizer and renderer, or raise to trigger a skip."""
    tokenizer = _load_case_tokenizer(case)
    renderer = get_renderer(case.renderer, tokenizer)
    return tokenizer, renderer


def test_hf_case_loads_pinned_public_repo_by_default(monkeypatch) -> None:
    case = next(case for case in RENDERER_MATRIX if case.renderer == "kimi_k3")
    monkeypatch.delenv("KIMI_K3_MODEL_PATH", raising=False)
    tokenizer = object()
    renderer = object()
    loaded: list[tuple[str, str | None, bool | None]] = []

    def fake_load_tokenizer(model, revision, trust):
        loaded.append((model, revision, trust))
        return tokenizer

    monkeypatch.setitem(
        globals(),
        "load_tokenizer",
        fake_load_tokenizer,
    )
    monkeypatch.setitem(globals(), "get_renderer", lambda _name, _tokenizer: renderer)

    _load_tokenizer_result.cache_clear()
    try:
        assert _load_renderer(case) == (tokenizer, renderer)
        assert loaded == [
            (
                "moonshotai/Kimi-K3",
                "301be1b88c89c0d3a763da6301352cb8fe399e90",
                True,
            )
        ]
    finally:
        _load_tokenizer_result.cache_clear()


def test_local_fixture_path_resolves_at_load_time(monkeypatch) -> None:
    case = next(case for case in RENDERER_MATRIX if case.renderer == "kimi_k3")
    fixture_path = "/tmp/runtime-kimi-k3-fixture"
    tokenizer = object()
    renderer = object()
    loaded: list[tuple[str, str | None, bool | None]] = []

    def fake_load_tokenizer(model, revision, trust):
        loaded.append((model, revision, trust))
        return tokenizer

    monkeypatch.setenv("KIMI_K3_MODEL_PATH", fixture_path)
    monkeypatch.setitem(
        globals(),
        "load_tokenizer",
        fake_load_tokenizer,
    )
    monkeypatch.setitem(globals(), "get_renderer", lambda _name, _tokenizer: renderer)

    _load_tokenizer_result.cache_clear()
    try:
        assert _load_renderer(case) == (tokenizer, renderer)
        assert loaded == [(fixture_path, None, True)]
    finally:
        _load_tokenizer_result.cache_clear()


def _hf_xfail_reason(case: RendererCase, scenario: Scenario) -> str | None:
    """Resolve a documented HF-parity divergence for this pair.

    Precedence (all resolve to a documented xfail reason, or ``None`` when the
    pair is expected to match):

    1. a matrix-wide ``case.xfail_hf`` divergence;
    2. a scenario-wide ``scenario.xfail_reason`` divergence;
    3. the empirically-built per-(renderer, scenario) HF divergence map
       map — the common case, since almost every real divergence is
       renderer-specific rather than scenario-wide.
    """
    return (
        case.xfail_hf
        or scenario.xfail_reason
        or HF_EXPECTED_DIVERGENCES.get((case.renderer, scenario.id))
    )


# ---------------------------------------------------------------------------
# Harness integrity — prevent stale ids, flags, and divergence entries
# ---------------------------------------------------------------------------
def test_scenario_metadata_is_self_consistent() -> None:
    scenario_ids = [scenario.id for scenario in ALL_SCENARIOS]
    assert len(scenario_ids) == len(set(scenario_ids)), "scenario ids must be unique"

    for scenario in ALL_SCENARIOS:
        assert scenario.messages, f"{scenario.id}: messages must not be empty"
        assert scenario.ends_with_assistant == (
            scenario.messages[-1].get("role") == "assistant"
        ), f"{scenario.id}: ends_with_assistant is stale"

        # A scenario is tool-shaped if it carries tool calls / tool results OR
        # declares tools (a declaration-only scenario still needs a renderer
        # that can serialize the tool block).
        has_tools = bool(scenario.tools) or any(
            message.get("tool_calls") or message.get("role") == "tool"
            for message in scenario.messages
        )
        assert scenario.requires_tools == has_tools, (
            f"{scenario.id}: requires_tools={scenario.requires_tools}, "
            f"but tool-shaped (messages or tools=)={has_tools}"
        )

        has_thinking = any(
            message.get("reasoning_content") or _thinking_parts(message.get("content"))
            for message in scenario.messages
        )
        assert scenario.requires_thinking == has_thinking, (
            f"{scenario.id}: requires_thinking={scenario.requires_thinking}, "
            f"but thinking-shaped messages={has_thinking}"
        )


def test_renderer_qa_shards_partition_matrix() -> None:
    partitions = [
        _shard_renderer_cases(RENDERER_MATRIX, index, 4) for index in range(4)
    ]
    flattened = [case for partition in partitions for case in partition]
    assert {case.renderer for case in flattened} == {
        case.renderer for case in RENDERER_MATRIX
    }
    assert len(flattened) == len(RENDERER_MATRIX)


def test_expected_divergence_entries_reference_the_matrix() -> None:
    renderer_names = {case.renderer for case in RENDERER_MATRIX}
    scenario_ids = {scenario.id for scenario in ALL_SCENARIOS}
    # Maps are intentionally NOT disjoint: a pair can diverge on more than one
    # invariant (e.g. a renderer whose tool declaration diverges from HF *and*
    # whose parser does not read the tool call back). Each map still gates a
    # distinct invariant, so the only integrity requirement is that every entry
    # references a real (renderer, scenario) in the matrix/bank.
    divergence_maps = (
        HF_EXPECTED_DIVERGENCES,
        PARSE_EXPECTED_DIVERGENCES,
        HISTORICAL_PARSE_EXPECTED_DIVERGENCES,
        TEXT_EXPECTED_DIVERGENCES,
        OBSERVATION_EXPECTED_DIVERGENCES,
        EXTENSION_EXPECTED_DIVERGENCES,
    )
    for renderer_name, scenario_id in set().union(
        *(set(divergences) for divergences in divergence_maps)
    ):
        assert renderer_name in renderer_names, f"unknown renderer {renderer_name!r}"
        assert scenario_id in scenario_ids, f"unknown scenario {scenario_id!r}"


# ---------------------------------------------------------------------------
# Coverage guard — public renderers must not silently drop to skip
# ---------------------------------------------------------------------------
# By default the invariant tests skip when a tokenizer can't load (offline dev
# boxes stay green). That resilience hides a real failure mode: if a *public*
# model's tokenizer stops loading (HF outage, renamed repo, tokenizer-class
# break), its coverage silently vanishes while CI stays green. This guard turns
# that into a loud failure — but only in strict mode, so it never flakes on an
# offline dev box.
#
# Enable in CI with RENDERER_QA_STRICT=1 (any of 1/true/yes). Preview renderers
# (kimi_k27_code and deepseek_v4) are intentionally not REQUIRED and are exempt.
_STRICT_MODE = os.environ.get("RENDERER_QA_STRICT", "").lower() in ("1", "true", "yes")

_REQUIRED_CASES = [c for c in _QA_CASES if c.renderer in REQUIRED_RENDERERS]


@pytest.mark.skipif(
    not _STRICT_MODE,
    reason="coverage guard runs only under RENDERER_QA_STRICT=1 (set in CI)",
)
@pytest.mark.parametrize(
    "case", _REQUIRED_CASES, ids=[c.renderer for c in _REQUIRED_CASES]
)
@pytest.mark.timeout(180)
def test_required_renderers_load_in_strict_mode(case: RendererCase) -> None:
    """A REQUIRED (public, ungated) renderer's tokenizer MUST load in strict
    mode. Failing here means a public model lost coverage — fix the tokenizer
    resolution rather than letting the invariant tests skip it into a false
    green.
    """
    tokenizer, renderer = _load_renderer(case)
    assert tokenizer is not None
    assert renderer is not None
    assert renderer.has_extension_property == case.has_extension_property, (
        f"{case.renderer}: matrix has_extension_property="
        f"{case.has_extension_property}, renderer reports "
        f"{renderer.has_extension_property}"
    )

    canonical = next(
        scenario for scenario in ALL_SCENARIOS if scenario.id == "system_user"
    )
    result = compare_renderer_to_hf(
        renderer_name=case.renderer,
        tokenizer_model=case.resolved_tokenizer_model(),
        tokenizer_revision=case.resolved_tokenizer_revision(),
        tokenizer_trust_remote_code=case.tokenizer_trust_remote_code,
        messages=canonical.messages,
        add_generation_prompt=True,
        apply_chat_template_kwargs=case.hf_kwargs,
        tools=canonical.tools,
    )
    expected_gap = _hf_xfail_reason(case, canonical)
    if expected_gap:
        assert not result.match, (
            f"{case.renderer}'s strict smoke test now passes; remove its stale "
            f"expected divergence: {expected_gap}"
        )
        return
    assert result.match, (
        f"{case.renderer} loaded but failed the canonical HF parity smoke test:\n"
        f"{format_divergence(result)}"
    )


class TestPublicRendererContracts(RendererContractTests):
    cases = _QA_CASES
    divergences = {
        "hf": HF_EXPECTED_DIVERGENCES,
        "parse": PARSE_EXPECTED_DIVERGENCES,
        "historical_parse": HISTORICAL_PARSE_EXPECTED_DIVERGENCES,
        "text": TEXT_EXPECTED_DIVERGENCES,
        "observation": OBSERVATION_EXPECTED_DIVERGENCES,
        "extension": EXTENSION_EXPECTED_DIVERGENCES,
    }

    def load_renderer(self, case):
        try:
            return _load_renderer(case)
        except _UNAVAILABLE as exc:
            pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    def compare_reference(self, case, scenario, *, add_generation_prompt):
        self.load_renderer(case)
        return compare_renderer_to_hf(
            renderer_name=case.renderer,
            tokenizer_model=case.resolved_tokenizer_model(),
            tokenizer_revision=case.resolved_tokenizer_revision(),
            tokenizer_trust_remote_code=case.tokenizer_trust_remote_code,
            messages=scenario.messages,
            add_generation_prompt=add_generation_prompt,
            apply_chat_template_kwargs=case.hf_kwargs,
            tools=scenario.tools,
        )
