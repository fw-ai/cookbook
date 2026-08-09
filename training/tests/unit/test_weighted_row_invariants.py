"""What must hold when an SFT row carries per-message weights.

A Fireworks SFT row marks which turns carry loss with a per-message ``weight``.
The production entry point (:func:`render_messages_to_datums`, also behind
``RenderDatasetPreview`` / Render Samples) resolves those flags and renders with
``TrainOnWhat.CUSTOMIZED``, which requires every rendered message to declare
``trainable``.

Renderers routinely break that requirement, because several synthesize a
template message *while* rendering, after the dataset's flags were resolved. So
this module starts from the most basic property there is:

    A row that carries weights renders at all, and the template context the
    renderer invented for itself carries no loss.

Later invariants about *where* the loss lands build on this file.

Network-dependent: tokenizers load via ``transformers.AutoTokenizer`` and a
case skips cleanly when its model cannot be loaded.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from functools import cache
from typing import Any

import pytest
import transformers

import training.renderer  # noqa: F401  — registers the cookbook renderers
from tinker_cookbook.renderers import get_registered_renderer_names, get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat
from training.renderer.deepseek_v4 import _merge_tool_messages
from training.renderer.message_weights import untrained_synthesized_context
from training.tests.unit.renderer_matrix import RENDERER_MATRIX
from training.utils.supervised import render_messages_to_datums

# The QA matrix already binds most renderers to their canonical tokenizer.
_MATRIX_TOKENIZERS = {
    case.renderer: case.resolved_tokenizer_model() for case in RENDERER_MATRIX
}

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
    "deepseekv3_thinking": "deepseek-ai/DeepSeek-V3.1",
    "nemotron3_disable_thinking": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "gpt_oss_high_reasoning": "openai/gpt-oss-120b",
    "gpt_oss_no_sysprompt": "openai/gpt-oss-120b",
    "mistral": "mistralai/Ministral-3-3B-Instruct-2512",
    "glm5": "zai-org/GLM-5.1",
    "glm_moe_dsa": "zai-org/GLM-5.2",
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
    "gpt_oss_medium_reasoning": (
        "same class and synthesis path as gpt_oss_high_reasoning"
    ),
    "qwen3_vl": "vision renderer; needs an image processor fixture",
    "qwen3_vl_instruct": "vision renderer; needs an image processor fixture",
    "deepseek_v4": (
        "preview tokenizer does not load under the pinned transformers, so every "
        "case would skip; training/tests/unit/test_deepseek_v4_renderer.py skips "
        "for the same reason"
    ),
}

# Renderers that cannot render ANY multi-target multi-turn row, weighted or not.
# Pinned by ``test_multi_turn_gap_is_weight_independent`` so a crash there is
# never mistaken for weighted-row breakage.
_NO_MULTI_TARGET_SPLIT = {
    "minimax_m2": (
        "reports has_extension_property=False but ships no build_supervised_examples"
    )
}

# Renderer families that synthesize a template message while rendering, with a
# marker from that message's rendered text. Under per-message weights the
# marker must reach the prompt and stay out of the loss.
_SYNTHESIZED_CONTEXT_MARKERS = [
    ("nemotron3", "<|im_start|>system"),
    ("gpt_oss_high_reasoning", "Reasoning: high"),
    ("gemma4_thinking", "<|think|>"),
    ("mistral", "[SYSTEM_PROMPT]"),
    ("kimi_k25", "You are Kimi"),
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
# A weighted row renders
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


@pytest.mark.parametrize(
    "name,marker",
    _SYNTHESIZED_CONTEXT_MARKERS,
    ids=[name for name, _ in _SYNTHESIZED_CONTEXT_MARKERS],
)
def test_synthesized_template_context_is_rendered_but_never_trained(
    name: str, marker: str
):
    """Named check for the renderers that synthesize a message mid-render.

    The message they add has to be rendered — the prompt would otherwise differ
    from what the model sees at inference — and it has to stay out of the loss.
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
