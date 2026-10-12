"""Decision-row rendering: gateway text contract, packed-serving tokenization, option-code checks."""

from __future__ import annotations

import re

import pytest
import transformers

from training.renderer import get_renderer
from training.renderer.gemma4 import Gemma4Renderer
from training.renderer.muse_glimmer import MuseGlimmerOptions, MuseGlimmerRenderer
from training.utils.decision import (
    BRANCH_SENTINEL,
    OPTION_CODES,
    RENDERER_PROMPT_TEXT,
    SYSTEM_PROMPT,
    decision_datums,
    label_index,
    options_for,
    render_branch,
    render_state,
)
from training.utils.runner import DatasetError


class _CharTokenizer:
    """One id per character, one id per registered special-token string."""

    bos_token: str | None = None

    def __init__(self, specials: list[str]) -> None:
        self._special = {token: 300000 + i for i, token in enumerate(specials)}
        self._reverse = {value: key for key, value in self._special.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        while text:
            match = next((t for t in self._special if text.startswith(t)), None)
            if match is not None:
                ids.append(self._special[match])
                text = text[len(match) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, tokens, **_kwargs) -> str:
        return "".join(self._reverse.get(int(t), chr(int(t))) for t in tokens)


class _GlimmerTokenizer(_CharTokenizer):
    bos_token = "<|begin_of_text|>"

    def __init__(self) -> None:
        super().__init__(["<|begin_of_text|>", "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|patch|>"])


def _glimmer() -> MuseGlimmerRenderer:
    return MuseGlimmerRenderer(_GlimmerTokenizer(), options=MuseGlimmerOptions(include_current_date=False))


def _gemma4() -> Gemma4Renderer:
    specials = ["<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>", '<|"|>']
    specials += ["<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>"]
    return Gemma4Renderer(_CharTokenizer(specials))


ROW = {
    "state": {"command": "rm -rf build/", "cwd": "/repo"},
    "questions": {
        "unsafe": {"type": "noul", "instructions": "Is it unsafe?", "label": False},
        "category": {
            "type": "choice",
            "instructions": "Kind?",
            "criteria": {"read": "Reads files", "build": "Builds code"},
            "label": "build",
        },
        "risk": {
            "type": "score",
            "instructions": "Risk?",
            "criteria": ["none", "low", "high"],
            "label": 1,
        },
    },
}


def _glimmer_datums(row=ROW, **kwargs):
    return decision_datums(row, renderer=_glimmer(), renderer_name="muse_glimmer", **kwargs)


def test_text_contract_matches_gateway_renderer() -> None:
    assert len(set(OPTION_CODES)) == len(OPTION_CODES) == 208
    assert render_state("x") == "State:\nx\n\nQuestion:"
    assert options_for({"type": "noul", "criteria": {"true": "block it"}}) == [("true", "block it"), ("false", "no")]
    choice = options_for({"type": "choice", "criteria": {"zeta": "Z", "alpha": None}})
    assert choice == [("alpha", "alpha"), ("zeta", "Z")]
    assert options_for({"type": "score", "criteria": ["none", "high"]}) == [("0", "none"), ("1", "high")]
    assert label_index({"type": "noul", "label": "false"}, ["true", "false"]) == 1
    assert label_index({"type": "score"}, ["0", "1"]) is None


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: options_for({"type": "rank"}), "unknown question type"),
        (lambda: options_for({"type": "choice", "criteria": {}}), "choice criteria"),
        (lambda: options_for({"type": "score", "criteria": ["only"]}), "2 to 10 levels"),
        (lambda: options_for({"type": "choice", "criteria": {str(i): "" for i in range(209)}}), "at most 208"),
        (lambda: label_index({"type": "choice", "label": "missing"}, ["alpha", "zeta"]), "not an option"),
        (lambda: label_index({"type": "noul", "label": "yes"}, ["true", "false"]), "noul label"),
        (lambda: label_index({"type": "score", "label": True}, ["0", "1"]), "score label"),
        (lambda: _glimmer_datums({"state": "s", "questions": {}}), "non-empty 'questions'"),
        # Unlabeled questions are validated too: the trainer needs K >= 2 candidates.
        (
            lambda: _glimmer_datums({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": "x"}}}}),
            r"question 'q': choice requires at least 2 options, got 1",
        ),
    ],
)
def test_invalid_rows_raise_dataset_errors(call, match) -> None:
    with pytest.raises(DatasetError, match=match):
        call()


_GLIMMER_REPO = "meta-models/Muse-Glimmer-30B"
_GLIMMER_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
_BLANK_THINK = "<|start|>assistant to=self<|message|><|eom|><|start|>assistant to=user<|message|>"


def _serving_glimmer_ids(tokenizer, state_text: str, branches: list[str]) -> tuple[list[int], list[list[int]]]:
    """Packed serving's Glimmer prompt: HF template at low strength, generation prompt swapped for a blank think."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": state_text + BRANCH_SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength="low",
    )
    specials = sorted(tokenizer.get_added_vocab(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")

    def encode(text: str) -> list[int]:
        ids: list[int] = []
        for part in pattern.split(text):
            if part in specials:
                ids.append(tokenizer.convert_tokens_to_ids(part))
            elif part:
                ids.extend(tokenizer.encode(part, add_special_tokens=False))
        return ids

    head, tail = rendered.split(BRANCH_SENTINEL)
    generation_prompt = tail.rindex("<|start|>")
    tail = tail[:generation_prompt] + _BLANK_THINK
    return encode(head), [encode(branch + tail) for branch in branches]


def test_glimmer_rows_match_packed_serving_ids_on_the_real_tokenizer() -> None:
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(_GLIMMER_REPO, revision=_GLIMMER_REVISION)
    except Exception:  # noqa: BLE001 — gated repo / network
        pytest.skip(f"tokenizer for {_GLIMMER_REPO!r} not available")
    pairs = decision_datums(ROW, renderer=get_renderer("muse_glimmer", tokenizer), renderer_name="muse_glimmer")
    branches = []
    for question_id in sorted(ROW["questions"]):
        question = ROW["questions"][question_id]
        branches.append(render_branch(question["instructions"], [label for _, label in options_for(question)]))
    state_ids, branch_ids = _serving_glimmer_ids(tokenizer, render_state(ROW["state"]), branches)

    assert [datum.model_input.to_ints() for _, datum in pairs] == [state_ids + ids for ids in branch_ids]
    assert tokenizer.decode(pairs[0][1].model_input.to_ints()[-1:]) == "<|message|>"


def test_glimmer_rows_answer_after_an_empty_closed_think_turn() -> None:
    renderer = _glimmer()
    row = {**ROW, "questions": {**ROW["questions"], "unlabeled": {"type": "noul", "instructions": "?"}}}
    pairs = decision_datums(row, renderer=renderer, renderer_name="muse_glimmer")
    assert [question_type for question_type, _ in pairs] == ["choice", "score", "noul"]
    prompt = renderer.tokenizer.decode(pairs[0][1].model_input.to_ints())
    # Serving's low reasoning strength, stated once over the renderer's high default.
    assert prompt.count("Reasoning strength") == 1 and "Reasoning strength: low." in prompt
    assert prompt.endswith("Reply with only the letter of the correct option.<|eot|>" + _BLANK_THINK)
    assert _glimmer_datums(max_seq_len=8) == []


def test_gemma4_answer_position_is_the_empty_thought_channel() -> None:
    renderer = _gemma4()
    _, datum = decision_datums(ROW, renderer=renderer, renderer_name="gemma4")[0]
    prompt = renderer.tokenizer.decode(datum.model_input.to_ints())
    assert "gemma4" not in RENDERER_PROMPT_TEXT
    assert prompt.startswith("<bos><|turn>system\n" + SYSTEM_PROMPT)
    assert prompt.endswith(
        "Reply with only the letter of the correct option.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


class _OverrideTokenizer(_GlimmerTokenizer):
    def __init__(self, overrides: dict[str, list[int]]) -> None:
        super().__init__()
        self._overrides = overrides

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._overrides.get(text) or super().encode(text, add_special_tokens)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"B": [7, 7]}, "option code 'B' is 2 tokens for this model, expected 1"),
        (
            {"C": [ord("B")]},
            r"question 'kind': option codes encode to the same token id \(\['B', 'C'\] -> token 66\); "
            "this tokenizer collapses distinct codes",
        ),
    ],
)
def test_option_codes_must_be_distinct_single_tokens(overrides, match) -> None:
    renderer = MuseGlimmerRenderer(
        _OverrideTokenizer(overrides), options=MuseGlimmerOptions(include_current_date=False)
    )
    row = {
        "state": "s",
        "questions": {"kind": {"type": "choice", "criteria": {"a": "", "b": "", "c": ""}, "label": "a"}},
    }
    with pytest.raises(DatasetError, match=match):
        decision_datums(row, renderer=renderer, renderer_name="muse_glimmer")
