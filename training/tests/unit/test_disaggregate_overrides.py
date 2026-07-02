"""Parity tests for the local Split renderers that override upstream
``tinker_cookbook`` registrations to add multi-turn ALL_ASSISTANT_MESSAGES
disaggregation.

Coverage matrix (run for every renderer name in ``_PARITY_CASES``):

1. ``test_split_override_registered_and_has_disaggregate`` — the
   registered factory returns the local Split subclass, the mixin's
   ``build_supervised_examples`` shadows the upstream/base default, and
   ``has_extension_property`` reports False (so the dispatcher routes
   to the plural path instead of fast-pathing).
2. ``test_disaggregate_produces_n_examples`` — N-user-turn input ⇒ N
   datums, each containing the matching assistant answer in its
   trained-token slice.
3. ``test_disaggregate_strips_history_thinking`` — multi-turn input
   carrying ``reasoning_content`` produces datums where the *current*
   turn's reasoning is preserved while *historical* turns' reasoning is
   stripped to preserve inference parity.
4. ``test_disaggregate_per_datum_weight_mask_only_last_assistant`` —
   only the final assistant span in each datum has nonzero weight; the
   rest of the prompt context is loss-masked.
5. ``test_disaggregate_mixed_thinking_exact_prefix_tokenization_and_masking`` —
   intermittent thinking traces in intermediate assistant turns produce
   exactly the same tokens/masks as rendering each prefix as
   ``LAST_ASSISTANT_TURN``.
6. ``test_single_turn_skips_disaggregate`` — one-user-turn input bypasses
   the per-prefix loop and returns a single datum equal to the
   ``LAST_ASSISTANT_TURN`` rendering.
7. ``test_last_assistant_mode_short_circuits`` — even with multi-turn
   data, ``LAST_ASSISTANT_MESSAGE`` / ``LAST_ASSISTANT_TURN`` modes
   short-circuit the disaggregate loop (one datum trained on the final
   assistant only).
8. ``test_non_assistant_split_mode_warns_and_splits`` —
   ``ALL_TOKENS`` / ``ALL_MESSAGES`` etc. fire the renderer-extension
   warning and still produce N per-prefix datums.

Network-dependent: tokenizers load via ``transformers.AutoTokenizer``;
tests skip cleanly when a model can't be loaded (no network, gated
repo, ...).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import transformers

import training.renderer  # noqa: F401  — registers Split overrides
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Renderer, TrainOnWhat

# (registered name, HF model id, override class name).
_PARITY_CASES = [
    ("qwen3", "Qwen/Qwen3-8B", "Qwen3SplitRenderer"),
    (
        "qwen3_disable_thinking",
        "Qwen/Qwen3-8B",
        "Qwen3DisableThinkingSplitRenderer",
    ),
    ("qwen3_5", "Qwen/Qwen3.5-9B", "Qwen3_5SplitRenderer"),
    (
        "qwen3_5_disable_thinking",
        "Qwen/Qwen3.5-9B",
        "Qwen3_5DisableThinkingSplitRenderer",
    ),
    (
        "deepseekv3_thinking",
        "deepseek-ai/DeepSeek-V3.1",
        "DeepSeekV3ThinkingSplitRenderer",
    ),
    (
        "nemotron3",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "Nemotron3SplitRenderer",
    ),
    (
        "gpt_oss_high_reasoning",
        "openai/gpt-oss-120b",
        "GptOssSplitRenderer",
    ),
    (
        "gemma4",
        "google/gemma-4-12b-it",
        "Gemma4SplitRenderer",
    ),
    (
        "gemma4_thinking",
        "google/gemma-4-12b-it",
        "Gemma4ThinkingSplitRenderer",
    ),
]


# Renderers whose ``has_extension_property`` is True for *disable-thinking*
# variants — for these the dispatcher fast-paths to singular and the mixin's
# disaggregate loop never runs. We still want to register the Split subclass
# (defensive, no-cost mixin) but the per-turn-split tests don't apply.
_NON_THINKING_VARIANTS_NAMES: set[str] = set()


def _load_tokenizer(model_id: str):
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
    except Exception:  # noqa: BLE001 — network / gated repo / config drift
        return None


def _decoded(tok, token_ids: list[int]) -> str:
    return tok.decode(token_ids)


def _trained_text(tok, token_ids: list[int], weights: list[float]) -> str:
    return _decoded(tok, [t for t, w in zip(token_ids, weights) if w > 0])


def _token_ids_and_weights(example: tuple[Any, Any]) -> tuple[list[int], list[float]]:
    token_ids = list(example[0].to_ints())
    weights = [float(weight) for weight in example[1].tolist()]
    return token_ids, weights


def _resolve_renderer(name: str, model_id: str, split_classname: str):
    tok = _load_tokenizer(model_id)
    if tok is None:
        pytest.skip(f"tokenizer for {model_id!r} not available")
    renderer = get_renderer(name, tok)
    if type(renderer).__name__ != split_classname:
        pytest.fail(
            f"{name!r} should resolve to local {split_classname}, got "
            f"{type(renderer).__name__}"
        )
    return tok, renderer


def _multi_turn_messages(n: int = 3) -> list[dict[str, Any]]:
    """Return a [user, assistant] × ``n`` plain-content conversation."""
    msgs: list[dict[str, Any]] = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"Q{i + 1}"})
        msgs.append({"role": "assistant", "content": f"A{i + 1}"})
    return msgs


def _multi_turn_messages_with_reasoning(n: int = 3) -> list[dict[str, Any]]:
    """Return a multi-turn conversation where each assistant turn carries
    a unique reasoning sentinel inside a structured-content thinking
    part — the cookbook renderer's input convention for reasoning that
    flows through to ``<think>...</think>`` rendering. We use
    distinguishable strings so any test can grep-check whether a given
    turn's reasoning survived per-prefix rendering."""
    msgs: list[dict[str, Any]] = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"Q{i + 1}"})
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": f"REASON_TURN_{i + 1}"},
                    {"type": "text", "text": f"A{i + 1}"},
                ],
            }
        )
    return msgs


def _multi_turn_messages_with_intermittent_reasoning() -> list[dict[str, Any]]:
    """Return alternating plain/reasoning assistant turns.

    The second assistant is intentionally both intermediate and thinking-bearing:
    it is the turn that historically loses CoT if splitting/masking is wrong.
    """
    return [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "ANSWER_ONLY_TURN_1"},
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "INTERMITTENT_REASON_TURN_2"},
                {"type": "text", "text": "ANSWER_WITH_REASON_TURN_2"},
            ],
        },
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "ANSWER_ONLY_TURN_3"},
        {"role": "user", "content": "Q4"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "INTERMITTENT_REASON_TURN_4"},
                {"type": "text", "text": "ANSWER_WITH_REASON_TURN_4"},
            ],
        },
    ]


def _intermittent_answer_for_turn(turn_idx: int) -> str:
    if turn_idx == 1:
        return "ANSWER_WITH_REASON_TURN_2"
    if turn_idx == 3:
        return "ANSWER_WITH_REASON_TURN_4"
    return f"ANSWER_ONLY_TURN_{turn_idx + 1}"


# ─────────────────────────────────────────────────────────────────────────
# 1. Override registration
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_split_override_registered_and_has_disaggregate(
    name: str, model_id: str, split_classname: str
):
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    # Disaggregate mixin's ``build_supervised_examples`` shadows the base.
    assert (
        type(renderer).build_supervised_examples
        is not Renderer.build_supervised_examples
    ), f"{split_classname} must override build_supervised_examples"
    assert renderer.has_extension_property is False, (
        f"{name!r}: has_extension_property must be False so dispatcher routes "
        "multi-turn ALL_ASSISTANT_MESSAGES to the plural disaggregate path"
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. N-user-turn → N datums
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_produces_n_examples(
    name: str, model_id: str, split_classname: str
):
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = _multi_turn_messages(n=3)
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    assert len(examples) == len(user_idxs), (
        f"{name!r}: expected {len(user_idxs)} per-turn datums, " f"got {len(examples)}"
    )

    for i, example in enumerate(examples):
        token_ids = list(example[0].to_ints())
        weights = example[1].tolist()
        assert len(token_ids) == len(weights)
        trained = _trained_text(tok, token_ids, weights)
        expected_answer = messages[user_idxs[i] + 1]["content"]
        assert expected_answer in trained, (
            f"{name!r} datum {i}: trained tokens should contain {expected_answer!r}; "
            f"got {trained!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# 3. The actual bug: history reasoning stripped, current turn preserved
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_strips_history_thinking(
    name: str, model_id: str, split_classname: str
):
    """The headline correctness invariant: in datum ``i``, the *current*
    assistant turn's reasoning_content survives rendering, while reasoning
    from any *earlier* assistant turn is stripped (matches what HF
    apply_chat_template emits at inference)."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = _multi_turn_messages_with_reasoning(n=3)
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    # Some renderers (e.g. plain ``qwen3_disable_thinking``,
    # ``gpt_oss_*``) do not consume ``reasoning_content`` at all because
    # their chat template doesn't include a thinking phase. Detect this
    # at the singular level: if no datum carries any sentinel, the
    # assertion below is vacuous and we skip cleanly. This guards us
    # against false negatives without weakening the strip check.
    if not any(
        f"REASON_TURN_{i + 1}" in _decoded(tok, list(ex[0].to_ints()))
        for i, ex in enumerate(examples)
    ):
        pytest.skip(
            f"{name!r}: chat template does not surface reasoning_content "
            "(non-thinking renderer family); strip-from-history check N/A"
        )

    for i, example in enumerate(examples):
        decoded = _decoded(tok, list(example[0].to_ints()))
        # Current turn's reasoning MUST survive.
        current_marker = f"REASON_TURN_{i + 1}"
        assert current_marker in decoded, (
            f"{name!r} datum {i}: expected current-turn reasoning "
            f"{current_marker!r} in rendered prompt; got: {decoded!r}"
        )
        # Earlier turns' reasoning MUST be stripped.
        for earlier in range(i):
            stripped_marker = f"REASON_TURN_{earlier + 1}"
            assert stripped_marker not in decoded, (
                f"{name!r} datum {i}: history-turn reasoning "
                f"{stripped_marker!r} should be stripped (HF default), "
                f"but appears in rendered prompt: {decoded!r}"
            )


# ─────────────────────────────────────────────────────────────────────────
# 4. Intermittent thinking traces: exact token/mask prefix parity
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_mixed_thinking_exact_prefix_tokenization_and_masking(
    name: str, model_id: str, split_classname: str
):
    """Mixed plain/thinking assistant turns should disaggregate to datums
    whose token IDs and per-token weights exactly match independent
    ``LAST_ASSISTANT_TURN`` renders of each user-turn prefix."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = _multi_turn_messages_with_intermittent_reasoning()
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    user_idxs = [
        idx for idx, message in enumerate(messages) if message["role"] == "user"
    ]
    expected_prefixes = [
        messages[:next_user_idx] for next_user_idx in [*user_idxs[1:], len(messages)]
    ]

    assert len(examples) == len(expected_prefixes)

    for idx, (actual, prefix) in enumerate(zip(examples, expected_prefixes)):
        actual_token_ids, actual_weights = _token_ids_and_weights(actual)
        expected = renderer.build_supervised_example(
            prefix, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
        )
        expected_token_ids, expected_weights = _token_ids_and_weights(expected)

        assert actual_token_ids == expected_token_ids, (
            f"{name!r} datum {idx}: disaggregated tokenization must byte-for-byte "
            "match the independent LAST_ASSISTANT_TURN prefix render"
        )
        assert actual_weights == expected_weights, (
            f"{name!r} datum {idx}: disaggregated mask must exactly match the "
            "independent LAST_ASSISTANT_TURN prefix render"
        )

    trained_slices = [
        _trained_text(tok, *_token_ids_and_weights(example)) for example in examples
    ]
    for idx, trained in enumerate(trained_slices):
        expected_answer = _intermittent_answer_for_turn(idx)
        assert expected_answer in trained, (
            f"{name!r} datum {idx}: current assistant answer must be trained; "
            f"got trained slice {trained!r}"
        )
        for earlier in range(idx):
            earlier_answer = _intermittent_answer_for_turn(earlier)
            assert earlier_answer not in trained, (
                f"{name!r} datum {idx}: historical assistant answer "
                f"{earlier_answer!r} must remain masked; got {trained!r}"
            )

    decoded_datums = [
        _decoded(tok, _token_ids_and_weights(example)[0]) for example in examples
    ]
    if not any("INTERMITTENT_REASON_TURN_" in decoded for decoded in decoded_datums):
        pytest.skip(
            f"{name!r}: chat template does not surface intermittent reasoning "
            "(non-thinking renderer family); reasoning-token mask check N/A"
        )

    assert "INTERMITTENT_REASON_TURN_2" in trained_slices[1], (
        f"{name!r}: intermediate current-turn reasoning must be trained in datum 1; "
        f"got {trained_slices[1]!r}"
    )
    assert "INTERMITTENT_REASON_TURN_2" not in decoded_datums[2], (
        f"{name!r}: once turn 2 becomes history, its reasoning must be stripped "
        f"from datum 2; got {decoded_datums[2]!r}"
    )
    assert "INTERMITTENT_REASON_TURN_2" not in decoded_datums[3], (
        f"{name!r}: turn 2 reasoning must stay stripped from later history; "
        f"got {decoded_datums[3]!r}"
    )
    assert "INTERMITTENT_REASON_TURN_4" in trained_slices[3], (
        f"{name!r}: final current-turn reasoning must be trained in datum 3; "
        f"got {trained_slices[3]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# 5. Per-datum weight mask: only last assistant trained
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_per_datum_weight_mask_only_last_assistant(
    name: str, model_id: str, split_classname: str
):
    """Each per-turn datum is rendered as ``LAST_ASSISTANT_TURN``; the
    weight mask must therefore train ONLY the last assistant turn's
    answer, not earlier assistant answers in the same prefix."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = _multi_turn_messages(n=3)
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]

    for i, example in enumerate(examples):
        token_ids = list(example[0].to_ints())
        weights = example[1].tolist()
        trained = _trained_text(tok, token_ids, weights)

        # Current turn's answer trained.
        current_answer = messages[user_idxs[i] + 1]["content"]
        assert current_answer in trained, (
            f"{name!r} datum {i}: expected {current_answer!r} in trained "
            f"slice; got {trained!r}"
        )
        # Earlier turns' answers must NOT be trained — they are part of
        # the prefix context, weighted zero.
        for earlier in range(i):
            earlier_answer = messages[user_idxs[earlier] + 1]["content"]
            assert earlier_answer not in trained, (
                f"{name!r} datum {i}: prefix answer {earlier_answer!r} should "
                f"not appear in trained slice; got {trained!r}"
            )


# ─────────────────────────────────────────────────────────────────────────
# 6. Single-turn fast-path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_single_turn_skips_disaggregate(name: str, model_id: str, split_classname: str):
    """One-user-turn conversation: the mixin must return exactly one
    datum (no disaggregate loop). The dispatcher would already
    short-circuit at ``user_count > 1`` but the mixin's loop also
    handles this case correctly."""
    _tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = [
        {"role": "user", "content": "single Q"},
        {"role": "assistant", "content": "single A"},
    ]
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert (
        len(examples) == 1
    ), f"{name!r}: single-user-turn should produce 1 datum, got {len(examples)}"


# ─────────────────────────────────────────────────────────────────────────
# 7. LAST_ASSISTANT_* fast-path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode", [TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.LAST_ASSISTANT_TURN]
)
@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_last_assistant_mode_short_circuits(
    name: str, model_id: str, split_classname: str, mode: TrainOnWhat
):
    """Even with multi-turn data, ``LAST_ASSISTANT_*`` modes must produce
    a single datum — the mixin's second fast-path covers this. Trained
    tokens should match only the final assistant answer."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = _multi_turn_messages(n=3)
    examples = renderer.build_supervised_examples(messages, train_on_what=mode)
    assert len(examples) == 1, (
        f"{name!r} mode={mode}: must produce 1 datum (fast-path), "
        f"got {len(examples)}"
    )
    token_ids = list(examples[0][0].to_ints())
    weights = examples[0][1].tolist()
    trained = _trained_text(tok, token_ids, weights)
    assert "A3" in trained, (
        f"{name!r} mode={mode}: final answer 'A3' missing from trained "
        f"slice; got {trained!r}"
    )
    # Earlier answers must be in context (zero-weighted) but not in trained.
    assert "A1" not in trained
    assert "A2" not in trained


# ─────────────────────────────────────────────────────────────────────────
# 8. Non-ALL_ASSISTANT split mode warns + splits
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_non_assistant_split_mode_warns_and_splits(
    name: str, model_id: str, split_classname: str
):
    """``ALL_TOKENS`` / ``ALL_MESSAGES`` / ``ALL_USER_AND_SYSTEM_MESSAGES``
    on a non-extension renderer should fire the
    "does not satisfy the extension property" warning and still produce
    one datum per user turn (each rendered with the requested mode)."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = _multi_turn_messages(n=3)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        examples = renderer.build_supervised_examples(
            messages, train_on_what=TrainOnWhat.ALL_MESSAGES
        )
    assert any(
        "does not satisfy the extension property" in str(w.message) for w in captured
    ), (
        f"{name!r}: expected extension-property warning for ALL_MESSAGES; "
        f"captured: {[str(w.message) for w in captured]!r}"
    )
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    assert len(examples) == len(user_idxs)


# ─────────────────────────────────────────────────────────────────────────
# 9. Non-trainable round filter
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_skips_non_trainable_round(
    name: str, model_id: str, split_classname: str
):
    """A round whose terminal assistant is marked non-trainable
    (``weight=0`` or ``trainable=False``) must NOT generate a datum.
    The non-trainable assistant remains in the prefix of any later
    trainable round as context. Mirrors V1
    ``_split_at_thinking_boundaries``'s skip behavior."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "weight": 0},  # ← skip me
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    # 3 user turns → 3 candidate rounds, but the middle one's terminal
    # assistant (A2) is non-trainable → 2 datums.
    assert len(examples) == 2, (
        f"{name!r}: expected 2 datums (A1 round, A3 round) — middle A2 "
        f"round skipped because weight=0; got {len(examples)}"
    )

    # Datum 0 trains A1, not A2 / A3.
    trained0 = _trained_text(
        tok, list(examples[0][0].to_ints()), examples[0][1].tolist()
    )
    assert "A1" in trained0
    assert "A2" not in trained0
    assert "A3" not in trained0

    # Datum 1 trains A3 only. A2 is in the prefix as context (decoded
    # tokens contain A2) but its tokens are weight=0 (not in trained slice).
    decoded1 = _decoded(tok, list(examples[1][0].to_ints()))
    trained1 = _trained_text(
        tok, list(examples[1][0].to_ints()), examples[1][1].tolist()
    )
    assert "A2" in decoded1, (
        f"{name!r}: non-trainable A2 should still appear in datum 1's "
        f"prompt context as conversation history; got: {decoded1!r}"
    )
    assert "A2" not in trained1, (
        f"{name!r}: non-trainable A2 must not appear in trained tokens; "
        f"got trained slice: {trained1!r}"
    )
    assert "A3" in trained1


@pytest.mark.parametrize(
    "name,model_id,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_skips_first_round_with_weight_zero(
    name: str, model_id: str, split_classname: str
):
    """Skip filter applies to the *first* round too — not just middle ones.
    A first-turn assistant with weight=0 leaves no datum at all unless a
    later trainable round exists."""
    tok, renderer = _resolve_renderer(name, model_id, split_classname)

    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "weight": 0},  # ← skip first round
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert len(examples) == 1, (
        f"{name!r}: first-round terminal A1 weight=0 → skip; A2 round → 1 "
        f"datum. Got {len(examples)}"
    )
    decoded = _decoded(tok, list(examples[0][0].to_ints()))
    trained = _trained_text(
        tok, list(examples[0][0].to_ints()), examples[0][1].tolist()
    )
    # A1 still in prefix as context, just not trained.
    assert "A1" in decoded
    assert "A1" not in trained
    assert "A2" in trained
