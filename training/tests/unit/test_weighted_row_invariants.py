"""What must hold when an SFT row carries per-message weights.

A Fireworks SFT row marks which turns carry loss with a per-message ``weight``.
The production entry point (:func:`render_messages_to_datums`, also behind
``RenderDatasetPreview`` / Render Samples) resolves those flags and, for a
renderer whose chat template strips historical thinking, unrolls the row into
one datum per user turn. Two layers therefore have to agree about loss
placement, and the seam between them is where masking has broken before:

* the unroll picks the loss mode per prefix,
* the renderer may synthesize template messages while rendering that prefix —
  an empty system block, a thinking marker, a reasoning-effort preamble.

The properties pinned here, in the order they were established:

1. A weighted row renders at all, and the template context the renderer
   invented for itself carries no loss.
2. Each unrolled datum trains only its own terminal turn, so masking one
   message cannot move loss in a split that does not even contain it, and
   flags that restate the default change nothing at all.
3. Weights only ever REMOVE loss. For the same row rendered with and without
   weights, the token sequences are identical and every position the weighted
   render trains is also trained by the unweighted one. Since an unweighted
   render assigns loss exclusively to the terminal turn's assistant output,
   that single subset relation subsumes 1 and 2 and is the contract worth
   remembering.

Network-dependent: tokenizers load via ``transformers.AutoTokenizer`` and a
case skips cleanly when its model cannot be loaded.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from functools import cache
from typing import Any

import pytest
import tinker
import transformers

import training.renderer  # noqa: F401  — registers the cookbook renderers
from tinker_cookbook.renderers import get_registered_renderer_names, get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat
from training.renderer.deepseek_v4 import _merge_tool_messages
from training.renderer.message_weights import (
    _rendered_positions,
    stable_chunk_sentinel,
    untrained_synthesized_context,
)
from training.tests.unit.renderer_matrix import RENDERER_MATRIX
from training.utils.supervised import render_messages_to_datums

# The QA matrix already binds most renderers to their canonical tokenizer.
_MATRIX_TOKENIZERS = {
    case.renderer: case.resolved_tokenizer_model() for case in RENDERER_MATRIX
}

# Renderers the QA matrix does not carry a row for yet. The invariant below
# needs only a tokenizer, not the matrix's capability flags, so bind them here
# instead of enrolling them in every harness invariant. Each of these families
# synthesizes template context, which is exactly what this module guards.
# Renderers the QA matrix does not carry a row for. Several are the LEGACY
# concrete names, which resolve to their own classes rather than to the
# corrected ``*_interleaved`` adapters, so they need their own row here — a
# legacy name is where a weighted row is most likely to land in practice.
_EXTRA_TOKENIZERS = {
    "qwen3_5": "Qwen/Qwen3.5-9B",
    "qwen3_5_interleaved": "Qwen/Qwen3.5-9B",
    "qwen3_5_disable_thinking": "Qwen/Qwen3.5-9B",
    "qwen3_5_disable_thinking_interleaved": "Qwen/Qwen3.5-9B",
    "qwen3_6": "Qwen/Qwen3.6-27B",
    "qwen3_6_disable_thinking": "Qwen/Qwen3.6-27B",
    "qwen3_6_preserve_thinking": "Qwen/Qwen3.6-27B",
    "qwen3_8": "Qwen/Qwen3.8-27B",
    "deepseekv3_thinking": "deepseek-ai/DeepSeek-V3.1",
    "nemotron3_disable_thinking": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    # Same tokenizer family as the QA-matrix ``nemotron3`` row. Preserve /
    # interleaved aliases are distinct registered names (not covered by the
    # matrix) and must be enrolled so the coverage guard stays honest.
    "nemotron3_interleaved": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron3_low_thinking": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron3_preserve_thinking": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron3_preserved": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron3_ultra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nemotron3_ultra_disable_thinking": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nemotron3_ultra_interleaved": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nemotron3_ultra_medium_thinking": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nemotron3_ultra_preserve_thinking": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nemotron3_ultra_preserved": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "gpt_oss_high_reasoning": "openai/gpt-oss-120b",
    "gpt_oss_no_sysprompt": "openai/gpt-oss-120b",
    "mistral": "mistralai/Ministral-3-3B-Instruct-2512",
    "glm5": "zai-org/GLM-5.1",
    "glm_moe_dsa": "zai-org/GLM-5.2",
    "glm53_interleaved": "zai-org/GLM-5.3",
    "glm53_preserve_thinking": "zai-org/GLM-5.3",
    "kimi_k25": "moonshotai/Kimi-K2.5",
    "kimi_k27_code": "moonshotai/Kimi-K2.7-Code",
}

_TOKENIZER_FOR_RENDERER = {**_MATRIX_TOKENIZERS, **_EXTRA_TOKENIZERS}

# Registered names deliberately left out, with the reason. The coverage guard
# below fails when a new renderer lands in neither map, so enrolling it is a
# decision rather than an oversight. Only claim "same class" after checking:
# most legacy/`_interleaved` pairs are DIFFERENT classes.
_UNCOVERED_RENDERERS = {
    "gpt_oss_low_reasoning": "same class and synthesis path as gpt_oss_high_reasoning",
    "gpt_oss_medium_reasoning": "same class and synthesis path as gpt_oss_high_reasoning",
    "qwen3_vl": "vision renderer; needs an image processor fixture",
    "qwen3_vl_instruct": "vision renderer; needs an image processor fixture",
    "deepseek_v4": (
        "preview tokenizer does not load under the pinned transformers, so every "
        "case would skip; training/tests/unit/test_deepseek_v4_renderer.py skips "
        "for the same reason"
    ),
    "deepseek_v4_disable_thinking": "shares the deepseek_v4 tokenizer, so it skips for that reason too",
}

# Renderers that cannot render ANY multi-target multi-turn row, weighted or not.
# Pinned by ``test_multi_turn_gap_is_weight_independent`` so a crash there is
# never mistaken for weighted-row breakage.
_NO_MULTI_TURGET_SPLIT_REASON = (
    "reports has_extension_property=False but ships no build_supervised_examples"
)
_NO_MULTI_TARGET_SPLIT = {"minimax_m2": _NO_MULTI_TURGET_SPLIT_REASON}

# Renderer families that synthesize a template message while rendering, with a
# marker from that message's rendered text. Under per-message weights the
# marker must reach the prompt and stay out of the loss.
_SYNTHESIZED_CONTEXT_MARKERS = [
    ("nemotron3", "<|im_start|>system"),
    ("gpt_oss_high_reasoning", "Reasoning: high"),
    ("gemma4_thinking", "<|think|>"),
    ("mistral", "[SYSTEM_PROMPT]"),
]


class _TokenizerLoadTimeout(BaseException):
    """Escape huggingface_hub's broad retry handling on a stalled request."""


@contextmanager
def _tokenizer_load_timeout(seconds: int = 30):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise _TokenizerLoadTimeout(f"tokenizer load exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@cache
def _load_tokenizer(model_id: str):
    try:
        with _tokenizer_load_timeout():
            return transformers.AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
    except _TokenizerLoadTimeout:
        return None
    except Exception:  # noqa: BLE001 — network / gated repo / config drift
        return None


def _resolve_renderer(name: str):
    tokenizer = _load_tokenizer(_TOKENIZER_FOR_RENDERER[name])
    if tokenizer is None:
        pytest.skip(f"tokenizer for {_TOKENIZER_FOR_RENDERER[name]!r} not available")
    return tokenizer, get_renderer(name, tokenizer)


def _without_weights(row: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in message.items() if key != "weight"}
        for message in row
    ]


def _render(renderer, row: list[dict[str, Any]]):
    return render_messages_to_datums(
        [dict(message) for message in row],
        renderer=renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )


def _trained_text(tokenizer, datum) -> str:
    return tokenizer.decode(
        [
            token
            for token, weight in zip(datum.token_ids, datum.token_weights)
            if weight > 0
        ]
    )


def _trained_token_count(datum) -> int:
    return sum(1 for weight in datum.token_weights if weight > 0)


# ─────────────────────────────────────────────────────────────────────────
# Rows. Every round's terminal assistant stays trainable so no round is
# skipped and the weighted render has one datum per unweighted datum.
# ─────────────────────────────────────────────────────────────────────────

_SINGLE_TURN_MASKED_DRAFT = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "DRAFT", "weight": 0},
    {"role": "assistant", "content": "ANSWER", "weight": 1},
]

_MULTI_TURN_ALL_TRAINED = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "weight": 1},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "weight": 1},
]

_MULTI_TURN_MASKED_DRAFT = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "weight": 1},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "DRAFT", "weight": 0},
    {"role": "assistant", "content": "ANSWER", "weight": 1},
]

# Masking a MIDDLE message of the terminal turn selects a set no built-in mode
# expresses, so this row is the one that actually reaches CUSTOMIZED — the path
# every synthesizing renderer used to fail on.
_MULTI_TURN_MASKED_MIDDLE = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "weight": 1},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "CALL", "weight": 1},
    {"role": "assistant", "content": "RETRY", "weight": 0},
    {"role": "assistant", "content": "ANSWER", "weight": 1},
]

_ROWS = {
    "single_turn_masked_draft": _SINGLE_TURN_MASKED_DRAFT,
    "multi_turn_all_trained": _MULTI_TURN_ALL_TRAINED,
    "multi_turn_masked_draft": _MULTI_TURN_MASKED_DRAFT,
    "multi_turn_masked_middle": _MULTI_TURN_MASKED_MIDDLE,
}

_RENDERER_NAMES = sorted(_TOKENIZER_FOR_RENDERER)


def _is_multi_turn(row: list[dict[str, Any]]) -> bool:
    return sum(1 for message in row if message["role"] == "user") > 1


# ─────────────────────────────────────────────────────────────────────────
# The invariant
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("row_id", sorted(_ROWS))
@pytest.mark.parametrize("name", _RENDERER_NAMES)
def test_a_weighted_row_renders_and_trains_its_answer(name: str, row_id: str):
    """Per-message weights must not stop a row from rendering.

    ``CUSTOMIZED`` demands a ``trainable`` field on every rendered message, and
    a renderer that synthesizes one mid-render cannot know about the dataset's
    flags, so this used to fail outright on whole renderer families — including
    on a single-turn row, which never reaches any multi-turn machinery.
    """
    row = _ROWS[row_id]
    if name in _NO_MULTI_TARGET_SPLIT and _is_multi_turn(row):
        pytest.skip(
            f"{name!r} {_NO_MULTI_TARGET_SPLIT[name]}; pinned by "
            "test_multi_turn_gap_is_weight_independent"
        )
    tokenizer, renderer = _resolve_renderer(name)

    datums = _render(renderer, row)

    assert datums, f"{name!r}/{row_id}: expected at least one datum"
    trained = [_trained_text(tokenizer, datum) for datum in datums]
    assert any("ANSWER" in slice_ or "A2" in slice_ for slice_ in trained), (
        f"{name!r}/{row_id}: the row's final answer must be trained; got {trained!r}"
    )


@pytest.mark.parametrize("row_id", sorted(_ROWS))
@pytest.mark.parametrize("name", _RENDERER_NAMES)
def test_weights_only_remove_loss_from_the_terminal_turn(name: str, row_id: str):
    """Per-message weights must not move tokens, must not add loss anywhere, and
    must only ever withhold loss from the datum's own terminal turn.

    The unweighted render of the same row is the reference: it trains exactly
    each datum's terminal turn, so "weighted loss ⊆ unweighted loss" is what
    rules out training history a second time and rules out loss leaking onto a
    system block, thinking marker, or reasoning preamble the renderer
    synthesized after the flags were resolved.
    """
    row = _ROWS[row_id]
    if name in _NO_MULTI_TARGET_SPLIT and _is_multi_turn(row):
        pytest.skip(
            f"{name!r} {_NO_MULTI_TARGET_SPLIT[name]}; pinned by "
            "test_multi_turn_gap_is_weight_independent"
        )
    tokenizer, renderer = _resolve_renderer(name)

    weighted = _render(renderer, row)
    reference = _render(renderer, _without_weights(row))

    assert len(weighted) == len(reference), (
        f"{name!r}/{row_id}: weights must not change how many datums a row "
        f"produces; got {len(weighted)} vs {len(reference)}"
    )
    for index, (datum, reference_datum) in enumerate(zip(weighted, reference)):
        assert datum.token_ids == reference_datum.token_ids, (
            f"{name!r}/{row_id} datum {index}: weights select loss, so they must "
            "not change the rendered token sequence"
        )
        leaked = [
            position
            for position, (weight, reference_weight) in enumerate(
                zip(datum.token_weights, reference_datum.token_weights)
            )
            if weight > 0 and reference_weight == 0
        ]
        assert not leaked, (
            f"{name!r}/{row_id} datum {index}: positions {leaked[:8]} carry loss "
            "that the unweighted render does not — weights may only withhold "
            "loss from the terminal turn, never add it to history or to "
            "renderer-synthesized template context"
        )

    masked_contents = [
        message["content"] for message in row if message.get("weight") == 0
    ]
    if not masked_contents:
        # Flags that only restate the default must be a complete no-op, so the
        # preview a customer inspects matches the unweighted row byte for byte.
        assert [datum.token_weights for datum in weighted] == [
            datum.token_weights for datum in reference
        ], f"{name!r}/{row_id}: weight=1 everywhere must not change any mask"
        return

    weighted_trained = [_trained_text(tokenizer, datum) for datum in weighted]
    for content in masked_contents:
        assert not [trained for trained in weighted_trained if content in trained], (
            f"{name!r}/{row_id}: masked message {content!r} must not be trained by any "
            f"datum; got {weighted_trained!r}"
        )
    # Guard against a vacuous pass: when the unweighted render does train the
    # masked message, the weighted one has to end up with strictly less loss.
    # Renderers whose built-in modes already exclude it (Kimi treats
    # back-to-back assistants as separate turns) legitimately match.
    reference_trained = [_trained_text(tokenizer, datum) for datum in reference]
    if any(
        content in trained
        for content in masked_contents
        for trained in reference_trained
    ):
        assert sum(_trained_token_count(datum) for datum in weighted) < sum(
            _trained_token_count(datum) for datum in reference
        ), f"{name!r}/{row_id}: masking must withhold loss the unweighted row applies"


@pytest.mark.parametrize("name", _RENDERER_NAMES)
def test_all_weights_one_renders_like_an_unweighted_row(name: str):
    """``weight: 1`` on every assistant states the default and must therefore be
    a complete no-op: same datum count, same tokens, same masks.

    This is the sharpest check that the unroll applies the flags per turn rather
    than across the whole row. A renderer that hands each prefix the caller's
    mode verbatim re-trains every earlier turn in every later split, which shows
    up here as a mask that differs from the unweighted row's.
    """
    if name in _NO_MULTI_TARGET_SPLIT:
        pytest.skip(f"{name!r} {_NO_MULTI_TARGET_SPLIT[name]}")
    tokenizer, renderer = _resolve_renderer(name)

    weighted = _render(renderer, _MULTI_TURN_ALL_TRAINED)
    reference = _render(renderer, _without_weights(_MULTI_TURN_ALL_TRAINED))

    assert len(weighted) == len(reference), (
        f"{name!r}: weight=1 everywhere must not change the datum count"
    )
    for index, (datum, reference_datum) in enumerate(zip(weighted, reference)):
        assert datum.token_ids == reference_datum.token_ids, (
            f"{name!r} datum {index}: weights select loss, so they must not change "
            "the rendered token sequence"
        )
        assert datum.token_weights == reference_datum.token_weights, (
            f"{name!r} datum {index}: weight=1 restates the default, so it must not "
            f"change the mask; trained {_trained_text(tokenizer, datum)!r} vs "
            f"{_trained_text(tokenizer, reference_datum)!r}"
        )


@pytest.mark.parametrize("name", _RENDERER_NAMES)
def test_masked_answer_is_never_trained_by_any_datum(name: str):
    """The masked message stays in the prompt as history and never picks up loss
    in any datum, including the later rounds that carry it as context."""
    if name in _NO_MULTI_TARGET_SPLIT:
        pytest.skip(f"{name!r} {_NO_MULTI_TARGET_SPLIT[name]}")
    tokenizer, renderer = _resolve_renderer(name)

    datums = _render(renderer, _MULTI_TURN_MASKED_MIDDLE)

    trained_slices = [_trained_text(tokenizer, datum) for datum in datums]
    assert not [slice_ for slice_ in trained_slices if "RETRY" in slice_], (
        f"{name!r}: the masked message must never be trained; got {trained_slices!r}"
    )
    assert any("ANSWER" in slice_ for slice_ in trained_slices), (
        f"{name!r}: the trained answer must still be trained; got {trained_slices!r}"
    )
    full_decodes = [tokenizer.decode(datum.token_ids) for datum in datums]
    assert any("RETRY" in decoded for decoded in full_decodes), (
        f"{name!r}: the masked message must remain in the prompt as history"
    )


@pytest.mark.parametrize(
    "name,marker",
    _SYNTHESIZED_CONTEXT_MARKERS,
    ids=[name for name, _ in _SYNTHESIZED_CONTEXT_MARKERS],
)
def test_synthesized_template_context_is_rendered_but_never_trained(
    name: str, marker: str
):
    """Named check for the renderers that synthesize a message mid-render.

    These families used to fail the render outright on any weighted row,
    because the message they add reaches the base renderer with no ``trainable``
    field. It has to be rendered (the prompt would otherwise differ from
    inference) and it has to stay out of the loss.
    """
    tokenizer, renderer = _resolve_renderer(name)

    datums = _render(renderer, _MULTI_TURN_MASKED_MIDDLE)

    assert datums, f"{name!r}: expected at least one datum"
    for index, datum in enumerate(datums):
        assert marker in tokenizer.decode(datum.token_ids), (
            f"{name!r} datum {index}: synthesized template context {marker!r} must "
            "still be rendered into the prompt"
        )
        assert marker not in _trained_text(tokenizer, datum), (
            f"{name!r} datum {index}: synthesized template context {marker!r} must "
            "carry no loss"
        )


# The reported shape: a tool-calling Qwen 3.5 row whose only weight is a single
# `"weight": 0` on a superseded draft in the last round.
_CUSTOMER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def _customer_row() -> list[dict[str, Any]]:
    def tool_call(order_id: str, call_id: str) -> dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "lookup_order",
                "arguments": '{"order_id": "%s"}' % order_id,
            },
        }

    return [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Where is order A?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASON_1",
            "tool_calls": [tool_call("A", "call_a")],
        },
        {"role": "tool", "content": "shipped", "tool_call_id": "call_a"},
        {"role": "assistant", "content": "ANSWER_1", "reasoning_content": "REASON_2"},
        {"role": "user", "content": "And order B?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASON_3",
            "tool_calls": [tool_call("B", "call_b")],
        },
        {"role": "tool", "content": "delayed", "tool_call_id": "call_b"},
        {"role": "assistant", "content": "ANSWER_2", "reasoning_content": "REASON_4"},
        {"role": "user", "content": "Summarize both."},
        {
            "role": "assistant",
            "content": "ANSWER_3_DRAFT",
            "reasoning_content": "REASON_5",
            "weight": 0,
        },
        {"role": "assistant", "content": "ANSWER_3", "reasoning_content": "REASON_6"},
    ]


def test_customer_qwen3_5_row_trains_each_answer_in_exactly_one_datum():
    """One `"weight": 0` must not move loss anywhere else in the row.

    This is the reported symptom: masking a single superseded draft in the last
    round changed loss placement in the earlier splits, which do not even
    contain the masked message.
    """
    tokenizer, renderer = _resolve_renderer("qwen3_5_interleaved")

    def render(row):
        return render_messages_to_datums(
            [dict(message) for message in row],
            renderer=renderer,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            tools=_CUSTOMER_TOOLS,
        )

    weighted = render(_customer_row())
    reference = render(_without_weights(_customer_row()))

    assert len(weighted) == 3, f"expected one datum per user turn, got {len(weighted)}"
    trained = [_trained_text(tokenizer, datum) for datum in weighted]

    assert (
        "ANSWER_1" in trained[0]
        and "REASON_1" in trained[0]
        and "REASON_2" in trained[0]
    )
    assert (
        "ANSWER_2" in trained[1]
        and "REASON_3" in trained[1]
        and "REASON_4" in trained[1]
    )
    assert "ANSWER_3" in trained[2] and "REASON_6" in trained[2]
    assert "ANSWER_3_DRAFT" not in trained[2] and "REASON_5" not in trained[2]
    for index, answer in enumerate(["ANSWER_1", "ANSWER_2", "ANSWER_3"]):
        for other_index, other in enumerate(["ANSWER_1", "ANSWER_2", "ANSWER_3"]):
            if other_index == index:
                continue
            assert other not in trained[index], (
                f"datum {index} must train only {answer!r}; {other!r} also carries loss "
                f"in {trained[index]!r}"
            )

    # Historical reasoning is stripped once a turn becomes history, which is why
    # the row is unrolled at all.
    assert "REASON_2" not in tokenizer.decode(weighted[1].token_ids)
    assert "REASON_4" not in tokenizer.decode(weighted[2].token_ids)

    # The splits that do not contain the masked message must be untouched by it.
    for index in (0, 1):
        assert weighted[index].token_ids == reference[index].token_ids
        assert weighted[index].token_weights == reference[index].token_weights, (
            f"datum {index} does not contain the masked message, so its loss mask must "
            "not change when that message is masked"
        )


# Modes whose target set is some subset of the assistant messages. For these,
# per-message weights that are all 1 restate exactly what the mode already says,
# so they must be a complete no-op.
_ASSISTANT_TARGET_REQUESTS = [
    TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    TrainOnWhat.LAST_ASSISTANT_TURN,
    TrainOnWhat.ALL_ASSISTANT_MESSAGES,
]

# Modes that also target user/system messages. The weight schema resolves an
# unflagged non-assistant message to untrained, so weights genuinely narrow these
# to the assistants — a contradictory configuration, but it must still only ever
# narrow.
_WHOLE_SEQUENCE_REQUESTS = [
    TrainOnWhat.ALL_MESSAGES,
    TrainOnWhat.ALL_TOKENS,
]

_MODE_PROBE_ROW = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2"},
]


def _render_mode_probe(renderer, messages, requested: TrainOnWhat):
    return render_messages_to_datums(
        [dict(message) for message in messages],
        renderer=renderer,
        train_on_what=requested,
    )


def _all_assistant_weights_one(row: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**message, "weight": 1} if message["role"] == "assistant" else message
        for message in row
    ]


@pytest.mark.parametrize("requested", _ASSISTANT_TARGET_REQUESTS, ids=lambda m: m.value)
def test_all_weights_one_is_a_no_op_under_any_assistant_target_mode(
    requested: TrainOnWhat,
):
    """``train_on_what`` is user-settable config, and a narrow one must stay
    narrow. Under a mode that already targets only assistants, ``weight: 1``
    everywhere states nothing new, so the render must not move — not the datum
    count, not a single token, not a single weight."""
    tokenizer, renderer = _resolve_renderer("qwen3_5")

    weighted = _render_mode_probe(
        renderer, _all_assistant_weights_one(_MODE_PROBE_ROW), requested
    )
    reference = _render_mode_probe(renderer, _MODE_PROBE_ROW, requested)

    assert len(weighted) == len(reference), (
        f"{requested.value}: weight=1 everywhere must not change the datum count"
    )
    for index, (datum, reference_datum) in enumerate(zip(weighted, reference)):
        assert datum.token_ids == reference_datum.token_ids
        assert datum.token_weights == reference_datum.token_weights, (
            f"{requested.value} datum {index}: weight=1 restates the default, so it "
            f"must not change the mask; trained "
            f"{_trained_text(tokenizer, datum)!r} vs "
            f"{_trained_text(tokenizer, reference_datum)!r}"
        )


@pytest.mark.parametrize("requested", _WHOLE_SEQUENCE_REQUESTS, ids=lambda m: m.value)
def test_weights_never_widen_loss_under_a_whole_sequence_mode(requested: TrainOnWhat):
    """Combining weights with a mode that targets user/system messages narrows to
    the assistants, because an unflagged non-assistant message resolves to
    untrained. Contradictory config, but it must never gain loss the requested
    mode does not assign."""
    _tokenizer, renderer = _resolve_renderer("qwen3_5")

    weighted = _render_mode_probe(
        renderer, _all_assistant_weights_one(_MODE_PROBE_ROW), requested
    )
    reference = _render_mode_probe(renderer, _MODE_PROBE_ROW, requested)

    assert len(weighted) == len(reference)
    for index, (datum, reference_datum) in enumerate(zip(weighted, reference)):
        assert datum.token_ids == reference_datum.token_ids
        leaked = [
            position
            for position, (weight, reference_weight) in enumerate(
                zip(datum.token_weights, reference_datum.token_weights)
            )
            if weight > 0 and reference_weight == 0
        ]
        assert not leaked, (
            f"{requested.value} datum {index}: positions {leaked[:8]} carry loss the "
            "requested mode does not assign"
        )


@pytest.mark.parametrize(
    "requested",
    # The single-target modes, which render the row as one datum. A row asking
    # for ALL_ASSISTANT_MESSAGES goes through the unroll instead, where a masked
    # terminal answer drops its whole round rather than falling through.
    [TrainOnWhat.LAST_ASSISTANT_MESSAGE, TrainOnWhat.LAST_ASSISTANT_TURN],
    ids=lambda m: m.value,
)
def test_a_mask_no_mode_expresses_still_respects_the_requested_mode(
    requested: TrainOnWhat,
):
    """A mask no built-in mode expresses falls through to ``CUSTOMIZED``, which
    honours every flag verbatim. Flags on messages the requested mode never
    selects have to be cleared first, or masking the terminal answer under a
    narrow mode would move loss onto an earlier turn the caller excluded — and
    on a strip-history renderer those tokens have already lost their thinking.
    """
    tokenizer, renderer = _resolve_renderer("qwen3_5")

    row = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "weight": 1},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "weight": 1},
        {"role": "assistant", "content": "A3", "weight": 0},
    ]
    weighted = _render_mode_probe(renderer, row, requested)
    reference = _render_mode_probe(renderer, _without_weights(row), requested)

    assert len(weighted) == len(reference)
    for index, (datum, reference_datum) in enumerate(zip(weighted, reference)):
        assert datum.token_ids == reference_datum.token_ids
        leaked = [
            position
            for position, (weight, reference_weight) in enumerate(
                zip(datum.token_weights, reference_datum.token_weights)
            )
            if weight > 0 and reference_weight == 0
        ]
        assert not leaked, (
            f"{requested.value} datum {index}: positions {leaked[:8]} carry loss the "
            f"requested mode does not assign; trained "
            f"{_trained_text(tokenizer, datum)!r} vs the unweighted row's "
            f"{_trained_text(tokenizer, reference_datum)!r}"
        )


@pytest.mark.parametrize("name", ["qwen3_5", "nemotron3", "gemma4_thinking"])
def test_a_weighted_tool_result_inside_the_terminal_turn_is_not_trained(name: str):
    """Demoting history clears the flags BEFORE the terminal turn; inside it,
    a flag on something the turn's own mode would not train has to be cleared
    too.

    A tool result carrying ``weight: 1`` is the concrete case: an unrolled datum
    trains its terminal assistant turn, and a tool result is not part of that
    however the dataset flags it.
    """
    tokenizer, renderer = _resolve_renderer(name)

    datums = _render(
        renderer,
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "weight": 1},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "CALL", "weight": 1},
            {"role": "tool", "content": "TOOLRESULT", "weight": 1},
            {"role": "assistant", "content": "RETRY", "weight": 0},
            {"role": "assistant", "content": "ANSWER", "weight": 1},
        ],
    )

    trained = [_trained_text(tokenizer, datum) for datum in datums]
    assert not [slice_ for slice_ in trained if "TOOLRESULT" in slice_], (
        f"{name!r}: a tool result is never a training target; got {trained!r}"
    )
    assert any("ANSWER" in slice_ for slice_ in trained), (
        f"{name!r}: the terminal answer must still be trained; got {trained!r}"
    )


@pytest.mark.parametrize("name", ["qwen3_5", "nemotron3", "mistral"])
def test_customized_without_any_weights_still_fails_loudly(name: str):
    """A row with no per-message flags cannot be rendered with ``CUSTOMIZED``.

    Nothing says which messages to train, so the renderer rejects it. That has
    to stay an error: quietly treating every message as untrained would train a
    misconfigured job on nothing at all.
    """
    _tokenizer, renderer = _resolve_renderer(name)

    with pytest.raises(AssertionError, match="trainable"):
        render_messages_to_datums(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            renderer=renderer,
            train_on_what=TrainOnWhat.CUSTOMIZED,
        )


def test_an_image_slot_is_identified_by_content_not_by_chunk_position():
    """The two renders are compared position by position, and a non-text chunk
    has no tokens to compare, so it needs a stand-in.

    That stand-in has to come from the chunk's content. Deriving it from the
    chunk's index in the chunk list would break the moment a mode splits a text
    chunk somewhere else in the sequence — which GLM does, to mask the injected
    ``<think>`` — shifting every later index and making identical image slots
    look different.
    """
    image = tinker.types.ImageAssetPointerChunk(
        location="gs://bucket/image.png",
        format="png",
        expected_tokens=4,
    )
    same_image = tinker.types.ImageAssetPointerChunk(
        location="gs://bucket/image.png",
        format="png",
        expected_tokens=4,
    )
    other_image = tinker.types.ImageAssetPointerChunk(
        location="gs://bucket/other.png",
        format="png",
        expected_tokens=4,
    )

    # One text chunk on the left, versus the same tokens split in two.
    whole = tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=[1, 2, 3]), image]
    )
    split = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[1]),
            tinker.types.EncodedTextChunk(tokens=[2, 3]),
            same_image,
        ]
    )

    assert _rendered_positions(whole) == _rendered_positions(split), (
        "splitting a text chunk must not change how the image slot is identified"
    )
    assert stable_chunk_sentinel(image) != stable_chunk_sentinel(other_image), (
        "a different image must still compare as different"
    )


def test_mistral_tool_declarations_survive_a_second_render():
    """Declared tools must ride on the message, not on the renderer.

    Masking a subset of a turn renders the same conversation twice to intersect
    the flagged loss with the default loss. A renderer that consumed per-render
    state would drop its tool block on the second pass and silently misalign the
    two weight vectors.
    """
    tokenizer, renderer = _resolve_renderer("mistral")

    messages = [
        {"role": "user", "content": "Where is order A?"},
        {"role": "assistant", "content": "CALL", "weight": 1},
        {"role": "assistant", "content": "RETRY", "weight": 0},
        {"role": "assistant", "content": "ANSWER", "weight": 1},
    ]
    datums = render_messages_to_datums(
        [dict(message) for message in messages],
        renderer=renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        tools=_CUSTOMER_TOOLS,
    )

    assert len(datums) == 1
    decoded = tokenizer.decode(datums[0].token_ids)
    assert "[AVAILABLE_TOOLS]" in decoded, (
        "the tool declaration block must survive rendering the row twice"
    )
    assert "[AVAILABLE_TOOLS]" not in _trained_text(tokenizer, datums[0])


def test_weights_with_a_user_and_system_mode_relocate_loss_to_the_assistants():
    """Known limitation, pinned so it stays visible.

    ``ALL_USER_AND_SYSTEM_MESSAGES`` and the weight schema disagree about every
    message: the mode targets exactly the roles the schema resolves to untrained.
    On an unrolling renderer the requested mode never reaches the unroll, whose
    per-prefix default is the terminal assistant turn, so the flags win and loss
    moves from the user/system turns onto the assistants rather than emptying
    out. Combining the two is contradictory configuration and no SFT recipe does
    it; making the requested mode reach the unroll would mean widening
    ``build_supervised_examples``, which is an upstream-compatible signature.
    """
    tokenizer, renderer = _resolve_renderer("qwen3_5")

    weighted = _render_mode_probe(
        renderer,
        _all_assistant_weights_one(_MODE_PROBE_ROW),
        TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES,
    )

    trained = [_trained_text(tokenizer, datum) for datum in weighted]
    assert all("A" in slice_ for slice_ in trained), (
        f"expected the flags to win over the requested mode; got {trained!r}"
    )
    # Still one target per datum, which is what the unroll exists to guarantee.
    assert "A1" not in trained[1], f"history must not be re-trained; got {trained!r}"


def test_a_row_with_no_weights_is_left_completely_alone():
    """The common upload states no weights at all, and must stay on the path it
    has always taken.

    Renderers *reject* a ``trainable`` field outside ``CUSTOMIZED``, so filling
    one in here would not degrade quietly — it would fail every unweighted row
    on every renderer that synthesizes template context. Hence the early return,
    pinned here.
    """
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "tool", "content": "T"},
        {"role": "assistant", "content": "A2"},
    ]

    assert untrained_synthesized_context(messages) == messages
    assert not any(
        "trainable" in message for message in untrained_synthesized_context(messages)
    )


def test_only_messages_without_a_flag_are_filled():
    """Fill the messages the renderer just added; never touch a resolved flag.

    The helper cannot tell a synthesized message from a dataset message whose
    flag a renderer dropped, so it must at least never overwrite a flag that is
    already there — an assistant marked trainable stays trainable.
    """
    filled = untrained_synthesized_context(
        [
            {"role": "system", "content": "synthesized by the renderer"},
            {"role": "user", "content": "Q1", "trainable": False},
            {"role": "assistant", "content": "A1", "trainable": True},
            {"role": "assistant", "content": "A2", "trainable": False},
        ]
    )

    assert [message["trainable"] for message in filled] == [False, False, True, False]


def test_deepseek_v4_tool_merge_preserves_per_message_weights():
    """DeepSeek-V4 rebuilds user messages while folding tool results into them,
    copying an explicit key list, and synthesizes a user message for a tool
    result that follows no user turn. Every message it hands on must carry a
    resolved flag: the copy must keep an existing one, and the synthesized turn
    must declare itself untrained.

    Tokenizer-free on purpose — the V4 preview tokenizer does not load under the
    pinned ``transformers``, so every rendering test for it skips.
    """
    merged = _merge_tool_messages(
        [
            {"role": "user", "content": "Q1", "trainable": True},
            {"role": "assistant", "content": "A1", "trainable": True},
            {"role": "tool", "content": "T1", "tool_call_id": "c1"},
        ]
    )
    assert merged[0].get("trainable") is True, (
        f"the rebuilt user message must keep its resolved flag; got {merged[0]!r}"
    )

    # ``_preprocess`` finishes the job for the message the fold synthesized.
    resolved = untrained_synthesized_context(merged)
    assert [message.get("trainable", "<missing>") for message in resolved] == [
        True,
        True,
        False,
    ], f"every message must reach the renderer with a flag; got {resolved!r}"


def test_multi_turn_gap_is_weight_independent():
    """Pin the renderers that cannot render a multi-target multi-turn row at all.

    They report ``has_extension_property=False`` without shipping
    ``build_supervised_examples``, so an UNWEIGHTED multi-turn row fails the same
    way a weighted one does. Pinning it here keeps that pre-existing gap from
    being read as weighted-row breakage, and turns fixing it into a green test
    once this entry is dropped.
    """
    for name in _NO_MULTI_TARGET_SPLIT:
        _tokenizer, renderer = _resolve_renderer(name)
        with pytest.raises(NotImplementedError):
            _render(renderer, _without_weights(_MULTI_TURN_ALL_TRAINED))


def test_every_registered_renderer_is_covered_or_excluded():
    """A new renderer must be enrolled in this contract or excluded on purpose."""
    registered = set(get_registered_renderer_names())
    accounted = set(_TOKENIZER_FOR_RENDERER) | set(_UNCOVERED_RENDERERS)
    assert not registered - accounted, (
        "these registered renderers are neither bound to a tokenizer nor listed "
        f"in _UNCOVERED_RENDERERS: {sorted(registered - accounted)}"
    )
