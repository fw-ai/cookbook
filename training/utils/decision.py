"""Decision-model training rows: one state, typed questions, one option-code token.

A decision row asks typed questions about one shared ``state``::

    {"state": "...",
     "questions": {
       "unsafe":   {"type": "noul", "instructions": "...", "label": true},
       "category": {"type": "choice", "instructions": "...",
                    "criteria": {"benign": "...", "exfiltration": "..."}, "label": "exfiltration"},
       "urgency":  {"type": "score", "instructions": "...", "criteria": ["none", "low", "high"], "label": 2}}}

The text matches the System One gateway renderer (``api_gateway/pkg/systemone``).
Each question is tokenized the way packed serving tokenizes it: the state head and
the branch plus template tail are encoded separately, each split at special tokens,
because encoding the joined string can merge tokens across the boundary. States and
instructions must not contain special-token strings. The target is the single
option-code token at the answer position, which ``RENDERER_PROMPT_TEXT`` aligns
with the position serving reads.

Each labeled question becomes one K-way datum: the prompt plus the gold code, with
the question's option ids in ``candidate_tokens`` on the answer (last) row.
The trainer normalizes that row over those options when the request carries
``DECISION_LOSS_FN_CONFIG``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fireworks.training import sdk

from training.renderer.base import Message, Renderer
from training.utils.runner import DatasetError
from training.utils.supervised import build_datum_from_tokens_and_weights

SYSTEM_PROMPT = "You are a decision engine. You answer questions about a state by replying " "with a single letter."

# Option codes in order. Every code used by a question must encode to exactly
# one token for the model being trained.
OPTION_CODES: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "!", '"',
    "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<",
    "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "①", "②",
    "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫", "⑬", "⑭", "⑮", "⑰", "⑴", "⑵",
    "⑶", "⑷", "⑸", "⑹", "Ⓐ", "Ⓒ", "Ⓘ", "Ⓜ", "Ⓡ", "ⓐ", "ⓑ", "ⓒ", "ⓓ", "ⓔ", "ⓗ", "ⓘ",
    "ⓛ", "ⓜ", "ⓝ", "ⓞ", "ⓡ", "ⓢ", "ⓤ", "ⓧ", "一", "二", "三", "四", "五", "六", "七", "八",
    "九", "十", "百", "千", "万", "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ",
    "μ", "ν", "ξ", "ο", "π", "ρ", "ς", "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "Α", "Β",
    "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ",
    "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω", "０", "１", "２", "３", "４", "５", "６", "７", "８", "９",
)  # fmt: skip

MAX_QUESTIONS = 256
# The trainer normalizes each answer row over its K options and requires K >= 2.
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = len(OPTION_CODES)
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# A private-use code point that no tokenizer or chat template emits.
BRANCH_SENTINEL = "\ue000packed-branch\ue000"

# Cross-entropy config under which the trainer normalizes each datum's answer
# row over its ``candidate_tokens`` instead of the full vocabulary.
DECISION_LOSS_FN_CONFIG: Mapping[str, str] = {"target_logprob_support": "candidates"}
CANDIDATE_TOKENS_KEY = "candidate_tokens"
CANDIDATE_TOKENS_PAD = -100

QuestionType = Literal["noul", "choice", "score"]

# Per renderer: (text after SYSTEM_PROMPT, text after the generation prompt). The last
# token of the second is the row serving reads. Muse Glimmer is thinking-only: serving
# renders reasoning "low" and reads the option logits after an empty, closed to=self
# turn and the to=user header. The template adds a reasoning line only when the system
# text has none, so stating it here renders serving's text whatever the renderer default is.
RENDERER_PROMPT_TEXT: Mapping[str, tuple[str, str]] = {
    "muse_glimmer": (
        "\n\nReasoning strength: low.",
        " to=self<|message|><|eom|><|start|>assistant to=user<|message|>",
    ),
}


def is_decision_row(row: Mapping[str, Any]) -> bool:
    """True for ``{"state", "questions"}`` rows; chat rows carry ``messages``."""
    return "questions" in row and "state" in row and "messages" not in row


def render_text(value: Any, fallback: str = "") -> str:
    """A field as prompt text: strings as-is, other JSON values compact."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_state(state: Any) -> str:
    """The user turn's text; each branch continues it after ``Question:``."""
    return "State:\n" + render_text(state) + "\n\nQuestion:"


def render_branch(instructions: str, labels: Sequence[str]) -> str:
    lines = "".join(f"{OPTION_CODES[i]}) {label}\n" for i, label in enumerate(labels))
    return f" {instructions}\n{lines}\nReply with only the letter of the correct option."


def options_for(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The question's ``(key, label)`` options in code order: option ``i`` reads ``OPTION_CODES[i]``."""
    question_type = question.get("type")
    criteria = question.get("criteria")
    if question_type == "noul":
        yes, no = "yes", "no"
        if criteria is not None:
            if not isinstance(criteria, Mapping):
                raise DatasetError("noul criteria must be an object with true/false descriptions")
            yes = render_text(criteria.get("true"), yes)
            no = render_text(criteria.get("false"), no)
        return [("true", yes), ("false", no)]
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise DatasetError("choice criteria must be a non-empty map of option to description")
        if len(criteria) < MIN_CHOICE_OPTIONS:
            raise DatasetError(
                f"choice requires at least {MIN_CHOICE_OPTIONS} options, got {len(criteria)}: "
                "the K-way loss normalizes over the question's options"
            )
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise DatasetError(f"options per question: at most {MAX_CHOICE_OPTIONS}, got {len(criteria)}")
        return [(key, render_text(criteria[key], key)) for key in sorted(criteria)]
    if question_type == "score":
        if not isinstance(criteria, Sequence) or isinstance(criteria, str):
            raise DatasetError("score criteria must be an ordered array of level descriptions")
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise DatasetError(f"score requires {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(criteria)}")
        return [(str(i), render_text(level, str(i))) for i, level in enumerate(criteria)]
    raise DatasetError(f"unknown question type {question_type!r}")


def label_index(question: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    """Index of the labeled option, or None for an unlabeled question."""
    if "label" not in question or question["label"] is None:
        return None
    label = question["label"]
    if question["type"] == "noul":
        if isinstance(label, bool):
            key = "true" if label else "false"
        elif label in ("true", "false"):
            key = label
        else:
            raise DatasetError(f"noul label must be true or false, got {label!r}")
    elif question["type"] == "score":
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise DatasetError(f"score label must be a 0-based level index, got {label!r}")
        key = str(label)
    else:
        key = str(label)
    if key not in keys:
        raise DatasetError(f"label {label!r} is not an option; expected one of {list(keys)}")
    return list(keys).index(key)


def render_state_head_and_tail(
    renderer: Renderer,
    state_text: str,
    *,
    renderer_name: str,
) -> tuple[list[int], str]:
    """The encoded state head and the template tail text, split where branch text joins the user turn."""
    tokenizer = renderer.tokenizer
    system_suffix, answer_prefill = RENDERER_PROMPT_TEXT.get(renderer_name, ("", ""))
    prefill = answer_prefill or None
    messages = [
        Message(role="system", content=SYSTEM_PROMPT + system_suffix),
        Message(role="user", content=state_text + BRANCH_SENTINEL),
    ]
    marked = renderer.build_generation_prompt(messages, prefill=prefill)
    text = tokenizer.decode(marked.to_ints(), skip_special_tokens=False)
    if text.count(BRANCH_SENTINEL) != 1:
        raise DatasetError("the chat template must render the branch marker exactly once")
    head, tail = text.split(BRANCH_SENTINEL)

    unmarked = renderer.build_generation_prompt(
        [messages[0], Message(role="user", content=state_text)],
        prefill=prefill,
    )
    if tokenizer.decode(unmarked.to_ints(), skip_special_tokens=False) != head + tail:
        raise DatasetError(
            "the tokenizer does not decode the rendered decision prompt losslessly; "
            "the state head and template tail cannot be recovered"
        )
    return _encode(tokenizer, head), tail


def decision_datums(
    row: Mapping[str, Any],
    *,
    renderer: Renderer,
    renderer_name: str,
    max_seq_len: int | None = None,
) -> list[tuple[QuestionType, sdk.Datum]]:
    """One ``(question type, K-way datum)`` per labeled question, in sorted question-id order.

    Questions longer than ``max_seq_len`` (prompt plus code) are dropped.
    """
    questions = row.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise DatasetError("decision row needs a non-empty 'questions' object")
    if len(questions) > MAX_QUESTIONS:
        raise DatasetError(f"questions per row: at most {MAX_QUESTIONS}, got {len(questions)}")
    if row.get("state") is None:
        raise DatasetError("decision row needs a 'state'")

    tokenizer = renderer.tokenizer
    state_tokens, tail = render_state_head_and_tail(renderer, render_state(row["state"]), renderer_name=renderer_name)
    code_ids: dict[str, int] = {}
    out: list[tuple[QuestionType, sdk.Datum]] = []
    for question_id in sorted(questions):
        question = questions[question_id]
        if not isinstance(question, Mapping):
            raise DatasetError(f"question {question_id!r} must be an object")
        try:
            options = options_for(question)
            gold = label_index(question, [key for key, _ in options])
        except DatasetError as exc:
            raise DatasetError(f"question {question_id!r}: {exc}") from exc
        if gold is None:
            continue
        branch = render_branch(render_text(question.get("instructions")), [label for _, label in options])
        codes = OPTION_CODES[: len(options)]
        option_ids = []
        for code in codes:
            if code not in code_ids:
                code_ids[code] = _single_token(tokenizer, code)
            option_ids.append(code_ids[code])
        _require_distinct_option_ids(question_id, codes, option_ids)
        prompt = [*state_tokens, *_encode(tokenizer, branch + tail)]
        if max_seq_len is not None and len(prompt) + 1 > max_seq_len:
            continue
        out.append((question["type"], kway_datum(prompt, option_ids, gold)))
    return out


def kway_datum(prompt_tokens: Sequence[int], option_token_ids: Sequence[int], gold: int) -> sdk.Datum:
    """Cross-entropy datum on the gold option code, with the K options on the answer row."""
    datum = build_datum_from_tokens_and_weights(
        [*prompt_tokens, option_token_ids[gold]], [0.0] * len(prompt_tokens) + [1.0]
    ).datum
    rows, k = len(prompt_tokens), len(option_token_ids)
    candidates = [CANDIDATE_TOKENS_PAD] * ((rows - 1) * k) + list(option_token_ids)
    return sdk.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            **datum.loss_fn_inputs,
            CANDIDATE_TOKENS_KEY: sdk.TensorData(data=candidates, dtype="int64", shape=[rows, k]),
        },
    )


def option_token_ids(datum: sdk.Datum) -> list[int]:
    """The K option ids on a K-way datum's answer row."""
    candidates = datum.loss_fn_inputs[CANDIDATE_TOKENS_KEY]
    return [int(token) for token in candidates.data[-candidates.shape[1] :]]


def gold_option_index(datum: sdk.Datum) -> int:
    """Index of a K-way datum's target among its options."""
    return option_token_ids(datum).index(int(datum.loss_fn_inputs["target_tokens"].data[-1]))


def option_logprob_datum(datum: sdk.Datum) -> sdk.Datum:
    """Forward datum whose ``[N, K]`` targets read every option's logprob at the answer (last) row.

    They share one normalizer, so their K-way softmax is the one serving applies.
    """
    options = option_token_ids(datum)
    rows, k = datum.model_input.length, len(options)
    return sdk.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=options * rows, dtype="int64", shape=[rows, k]),
            "weights": sdk.TensorData(data=[0.0] * (rows * k), dtype="float32", shape=[rows, k]),
        },
    )


def option_logprobs(
    training_client: Any,
    datums: Sequence[sdk.Datum],
    *,
    chunk: int = 64,
) -> list[list[float]]:
    """Each K-way datum's K option logprobs at its answer row, grouped by K per forward.

    ``training_client.forward`` returns the forward result, as ``ReconnectableClient`` does.
    """
    by_width: dict[int, list[int]] = {}
    for index, datum in enumerate(datums):
        by_width.setdefault(len(option_token_ids(datum)), []).append(index)
    rows: list[list[float]] = [[] for _ in datums]
    for k, indices in sorted(by_width.items()):
        for start in range(0, len(indices), chunk):
            part = indices[start : start + chunk]
            fwd = training_client.forward([option_logprob_datum(datums[i]) for i in part], "cross_entropy")
            for index, output in zip(part, fwd.loss_fn_outputs, strict=True):
                values = [float(v) for v in output["logprobs"].data]
                if len(values) % k:
                    raise ValueError(f"forward returned {len(values)} logprobs for K={k} targets")
                rows[index] = values[-k:]
    return rows


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _require_distinct_option_ids(question_id: str, codes: Sequence[str], option_ids: Sequence[int]) -> None:
    by_id: dict[int, list[str]] = {}
    for code, token_id in zip(codes, option_ids, strict=True):
        by_id.setdefault(token_id, []).append(code)
    collisions = {token_id: group for token_id, group in by_id.items() if len(group) > 1}
    if collisions:
        detail = "; ".join(f"{group} -> token {token_id}" for token_id, group in collisions.items())
        raise DatasetError(
            f"question {question_id!r}: option codes encode to the same token id ({detail}); "
            "this tokenizer collapses distinct codes, so the K-way loss and readout cannot tell those options apart"
        )


def _single_token(tokenizer: Any, code: str) -> int:
    ids = _encode(tokenizer, code)
    if len(ids) != 1:
        raise DatasetError(f"option code {code!r} is {len(ids)} tokens for this model, expected 1")
    return ids[0]
