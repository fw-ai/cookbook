"""Exact-token certification tests for the Kimi K3.1 full-history TITO renderer.

The oracle reproduces the checkpoint tokenizer's ``apply_chat_template`` from
data only: a ``tiktoken`` encoding built from the pinned ``tiktoken.model``,
the checkpoint's token pattern and reserved specials, and the same
segment-by-segment encoding (structural segments with specials, text segments
without, tiktoken's long-run chunking) over the vendored
``encoding_k3.build_chat_segments``. No checkpoint Python is imported.

Source files come from the pinned Hugging Face revision. Set
``KIMI_K31_TOKENIZER_DIR`` to a local checkpoint directory to run offline.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOError, TITOParseError
from training.renderer.tito import build_sidecar_tito_renderer
from training.renderer.tito._vendored.kimi_k31 import encoding_k3
from training.renderer.tito.kimi_k31 import (
    KIMI_K31_ENCODER_SHA256,
    KIMI_K31_RENDERER_NAME,
    KIMI_K31_REVISION,
    KIMI_K31_TOKENIZER_FINGERPRINT,
    THINKING_EFFORTS,
    KimiK31TITORenderer,
    _normalize_tools,
    kimi_k31_certifications,
)
from training.renderer.tito.shared import _tokenizer_fingerprint, load_sidecar_tokenizer
from training.utils.kimi_k31_tokenizer import (
    KIMI_K31_PAT_STR,
    download_source,
    special_tokens,
    verify_source_dir,
    write_bundle,
)

_SOURCE_ENV = "KIMI_K31_TOKENIZER_DIR"
_OPEN, _CLOSE, _SEP, _EOM = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"
_VENDORED_PATH = Path(encoding_k3.__file__)


class _TiktokenOracle:
    """``TikTokenTokenizer``'s encoding path, rebuilt from the pinned data files."""

    _MAX_ENCODE_CHARS = 400_000
    _MAX_NO_WHITESPACES_CHARS = 25_000

    def __init__(self, source_dir: Path) -> None:
        tiktoken = pytest.importorskip("tiktoken")
        # lazy: tiktoken.load is a submodule that the package import does not load.
        from tiktoken.load import load_tiktoken_bpe

        self._model = tiktoken.Encoding(
            name="kimi_k31_oracle",
            pat_str=KIMI_K31_PAT_STR,
            mergeable_ranks=load_tiktoken_bpe(str(source_dir / "tiktoken.model")),
            special_tokens=dict(special_tokens(source_dir)),
        )

    @staticmethod
    def _split(text: str, limit: int) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = text[0].isspace() if text else False
        slice_start = 0
        for index, char in enumerate(text):
            is_now_space = char.isspace()
            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > limit:
                    yield text[slice_start:index]
                    slice_start = index
                    current_slice_len = 1
        yield text[slice_start:]

    def encode_piece(self, text: str, *, allow_special: bool) -> list[int]:
        ids: list[int] = []
        for start in range(0, len(text), self._MAX_ENCODE_CHARS):
            for piece in self._split(text[start : start + self._MAX_ENCODE_CHARS], self._MAX_NO_WHITESPACES_CHARS):
                if allow_special:
                    ids.extend(self._model.encode(piece, allowed_special="all"))
                else:
                    ids.extend(self._model.encode(piece, disallowed_special=()))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._model.decode(list(ids))

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict],
        effort: str | None,
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        segments = encoding_k3.build_chat_segments(
            messages,
            tools=_normalize_tools(tools) or None,
            add_generation_prompt=add_generation_prompt,
            thinking=True,
            thinking_effort=effort,
        )
        ids: list[int] = []
        for segment in segments:
            ids.extend(self.encode_piece(segment.text, allow_special=segment.allow_special))
        return ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_dir() -> Path:
    local = os.environ.get(_SOURCE_ENV)
    return verify_source_dir(local) if local else download_source()


@pytest.fixture(scope="module")
def bundle_dir(source_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("kimi_k31_bundle")
    assert write_bundle(destination, source_dir=source_dir) == KIMI_K31_TOKENIZER_FINGERPRINT
    return destination


@pytest.fixture(scope="module")
def fast_tokenizer(bundle_dir: Path) -> Any:
    # lazy: tokenizer loading requires the optional transformers dependency.
    from training.utils.tokenizers import load_tokenizer

    return load_tokenizer(str(bundle_dir))


@pytest.fixture(scope="module")
def oracle(source_dir: Path) -> _TiktokenOracle:
    return _TiktokenOracle(source_dir)


def _certification(effort: str | None) -> Any:
    label = effort or "default"
    for certification in kimi_k31_certifications():
        if certification.certification_id.endswith(f"-{label}-v1"):
            return certification
    raise AssertionError(f"no certification for {effort!r}")


@pytest.fixture(scope="module")
def renderers(fast_tokenizer: Any) -> dict[str | None, KimiK31TITORenderer]:
    return {
        effort: KimiK31TITORenderer(fast_tokenizer, certification=_certification(effort), thinking_effort=effort)
        for effort in (None, *THINKING_EFFORTS)
    }


# ---------------------------------------------------------------------------
# Random conversations
# ---------------------------------------------------------------------------

_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \n\t.,;:!?'\"()[]{}<>/\\|-_=+*&^%$#@~`中文日本語한국어ñéü😀"
)
_LITERALS = (_OPEN, _CLOSE, _SEP, _EOM, "[EOS]", "[BOS]", "<think>", "&quot;", "&amp;", "message", "  ", "\n\n")
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "act",
            "description": "Send one game command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "depth": {"type": "number"},
                    "options": {"type": "object"},
                    "items": {"type": "array"},
                    "verbose": {"type": "boolean"},
                },
            },
        },
    },
]
_ARGUMENT_VALUES = (
    '"go north"',
    '"<|open|>literal"',
    '"line\\nbreak \\"quoted\\""',
    '""',
    "3",
    "1e2",
    "-0.50",
    "true",
    "null",
    '{"a": [1, 2], "b":{"c":"d"}}',
    '[1,  2, {"x": null}]',
)
_ARGUMENT_KEYS = ("command", "target", "depth", "options", "items", "verbose", 'q&"k')


def _text(rng: random.Random, max_len: int = 60) -> str:
    parts = []
    for _ in range(rng.randint(1, 4)):
        parts.append("".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, max_len))))
        if rng.random() < 0.3:
            parts.append(rng.choice(_LITERALS))
    return "".join(parts)


def _arguments(rng: random.Random) -> str:
    keys = rng.sample(_ARGUMENT_KEYS, rng.randint(1, 3))
    spacing = rng.choice((("{", ", ", ": ", "}"), ("{", ",", ":", "}"), ("{ ", " ,", " : ", " }")))
    body = spacing[1].join(f"{json.dumps(key)}{spacing[2]}{rng.choice(_ARGUMENT_VALUES)}" for key in keys)
    return spacing[0] + body + spacing[3]


def _assistant(rng: random.Random, tools: list[dict]) -> dict:
    message: dict = {"role": "assistant", "content": _text(rng) if rng.random() < 0.8 else ""}
    if rng.random() < 0.85:
        message["reasoning_content"] = _text(rng, 120)
    if tools and rng.random() < 0.7:
        message["tool_calls"] = [
            {
                "id": f"call_{rng.randrange(10**9)}",
                "type": "function",
                "function": {"name": rng.choice(tools)["function"]["name"], "arguments": _arguments(rng)},
            }
            for _ in range(rng.choice((1, 1, 1, 2)))
        ]
    return message


def _conversation(rng: random.Random) -> tuple[list[dict], list[dict]]:
    tools = _TOOLS[: rng.randint(0, 2)]
    messages: list[dict] = []
    if rng.random() < 0.7:
        messages.append({"role": "system", "content": _text(rng)})
    messages.append({"role": "user", "content": _text(rng)})
    for _ in range(rng.randint(0, 3)):
        assistant = _assistant(rng, tools)
        messages.append(assistant)
        for call in assistant.get("tool_calls") or ():
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _text(rng)})
        if rng.random() < 0.5:
            messages.append({"role": "user", "content": _text(rng)})
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": _text(rng)})
    return messages, tools


def _request(messages: list[dict], tools: list[dict]) -> TITOChatRequest:
    payload: dict = {"messages": messages}
    if tools:
        payload["tools"] = tools
    return TITOChatRequest.from_openai(payload)


# ---------------------------------------------------------------------------
# Certification table and vendored encoder (no tokenizer files needed)
# ---------------------------------------------------------------------------


def test_certifications_pin_one_contract_per_effort() -> None:
    assert THINKING_EFFORTS == ("low", "high", "max")
    certifications = kimi_k31_certifications()
    names = sorted(name for certification in certifications for name in certification.renderer_names)
    assert names == sorted([KIMI_K31_RENDERER_NAME, *(f"{KIMI_K31_RENDERER_NAME}_{e}" for e in THINKING_EFFORTS)])
    assert len({certification.certification_id for certification in certifications}) == len(certifications)
    for certification in certifications:
        assert certification.tokenizer_fingerprint == KIMI_K31_TOKENIZER_FINGERPRINT
        assert certification.certification_id.startswith(f"kimi-k3.1-pro@{KIMI_K31_REVISION[:8]}-full-history-")
        assert certification.packaged_contract
        assert certification.artifact_files == ()


def test_vendored_encoder_matches_certification_sha256() -> None:
    assert hashlib.sha256(_VENDORED_PATH.read_bytes()).hexdigest() == KIMI_K31_ENCODER_SHA256


def test_effort_tiers_match_the_vendored_encoder() -> None:
    from training.renderer.tito._vendored.kimi_k31 import encoding_k3

    assert set(THINKING_EFFORTS) == encoding_k3._VALID_THINKING_EFFORTS


def test_vendored_encoder_imports_only_stdlib_and_runs_nothing() -> None:
    tree = ast.parse(_VENDORED_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    banned_calls = {"exec", "eval", "compile", "open", "input", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_calls, f"line {node.lineno}: {node.func.id}"
    assert imported <= {"__future__", "json", "dataclasses", "typing"}


def test_tools_drop_none_strict_and_container() -> None:
    raw = [
        {
            "type": "function",
            "container": None,
            "function": {"name": "act", "description": None, "strict": True, "parameters": {"type": "object"}},
        }
    ]
    assert _normalize_tools(raw) == [
        {"type": "function", "function": {"name": "act", "parameters": {"type": "object"}}}
    ]


# ---------------------------------------------------------------------------
# Tokenizer conversion
# ---------------------------------------------------------------------------


def test_fast_tokenizer_matches_tiktoken_in_both_segment_modes(oracle, renderers) -> None:
    renderer = renderers[None]
    rng = random.Random(7)
    texts = [_text(rng, 200) for _ in range(1500)]
    texts += ["a" * 30_001, " " * 25_003 + "x", "中" * 26_000 + " end", ""]
    for text in texts:
        for allow_special in (True, False):
            expected = oracle.encode_piece(text, allow_special=allow_special)
            assert list(renderer._encode_piece(text, allow_special=allow_special)) == expected, (
                text[:60],
                allow_special,
            )
        ids = oracle.encode_piece(text, allow_special=True)
        assert renderer._decode(ids) == oracle.decode(ids)


def test_bundle_fingerprint_survives_the_sidecar_round_trip(fast_tokenizer, tmp_path) -> None:
    fast_tokenizer.save_pretrained(str(tmp_path))
    sidecar = load_sidecar_tokenizer(tmp_path, renderer_name=KIMI_K31_RENDERER_NAME)
    assert _tokenizer_fingerprint(sidecar) == KIMI_K31_TOKENIZER_FINGERPRINT
    request = _request([{"role": "user", "content": "look"}], _TOOLS)
    host_renderer = build_sidecar_tito_renderer(fast_tokenizer, KIMI_K31_RENDERER_NAME)
    sidecar_renderer = build_sidecar_tito_renderer(sidecar, KIMI_K31_RENDERER_NAME)
    assert sidecar_renderer.render_conversation_tokens(request) == host_renderer.render_conversation_tokens(request)


def test_registry_resolves_effort_variants_and_rejects_other_tokenizers(fast_tokenizer, bundle_dir) -> None:
    # lazy: tokenizer loading requires the optional transformers dependency.
    from training.utils.tokenizers import load_tokenizer

    for effort in THINKING_EFFORTS:
        renderer = build_sidecar_tito_renderer(fast_tokenizer, f"{KIMI_K31_RENDERER_NAME}_{effort}")
        assert isinstance(renderer, KimiK31TITORenderer)
        assert renderer.thinking_effort == effort
    altered = load_tokenizer(str(bundle_dir))
    altered.add_tokens(["<|not_kimi|>"])
    with pytest.raises(ValueError):
        build_sidecar_tito_renderer(altered, KIMI_K31_RENDERER_NAME)


# ---------------------------------------------------------------------------
# Rendering parity with the checkpoint template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effort", [None, *THINKING_EFFORTS])
def test_render_matches_checkpoint_apply_chat_template(effort, renderers, oracle) -> None:
    rng = random.Random(1000 + THINKING_EFFORTS.index(effort) if effort else 999)
    cases = 200 if effort is None else 40
    for _ in range(cases):
        messages, tools = _conversation(rng)
        rendered = list(renderers[effort].render_conversation_tokens(_request(messages, tools)))
        assert rendered == oracle.apply_chat_template(messages, tools, effort, add_generation_prompt=True)


def test_render_matches_template_for_long_tool_output(renderers, oracle) -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "check",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "act", "arguments": '{"command":"x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "z" * 30_000 + "\n" + " " * 26_000 + "done"},
    ]
    rendered = list(renderers[None].render_conversation_tokens(_request(messages, _TOOLS)))
    assert rendered == oracle.apply_chat_template(messages, _TOOLS, None, add_generation_prompt=True)


def test_effort_message_only_for_pinned_variants(renderers) -> None:
    request = _request([{"role": "user", "content": "hi"}], [])
    default = renderers[None].tokenizer.decode(list(renderers[None].render_conversation_tokens(request)))
    pinned = renderers["low"].tokenizer.decode(list(renderers["low"].render_conversation_tokens(request)))
    assert "thinking_effort" not in default
    assert "thinking_effort=low" in pinned
    assert default.endswith(f'{_OPEN}message role="assistant"{_SEP}{_OPEN}think{_SEP}')


# ---------------------------------------------------------------------------
# Full-history round trip: what TITO's split_ratio measures
# ---------------------------------------------------------------------------


def _without_call_ids(message) -> dict:
    plain = json.loads(json.dumps(dict(message), default=dict))
    for call in plain.get("tool_calls") or ():
        call.pop("id", None)
    return plain


def _assistant_completion(renderer, oracle, history, tools, assistant, effort) -> tuple[list[int], list[int]]:
    """Prompt ids and the completion a template-faithful model would sample."""
    prompt = list(renderer.render_conversation_tokens(_request(history, tools)))
    full = oracle.apply_chat_template([*history, assistant], tools, effort, add_generation_prompt=False)
    assert full[: len(prompt)] == prompt, "generation prompt must prefix the rendered assistant turn"
    completion = full[len(prompt) :]
    assert completion[-1] == renderer._end_of_msg_id
    return prompt, completion[:-1]


@pytest.mark.parametrize("effort", [None, "high"])
def test_parsed_turn_rerenders_as_an_exact_prefix(effort, renderers, oracle) -> None:
    renderer = renderers[effort]
    rng = random.Random(42 if effort is None else 43)
    turns = 0
    for _ in range(60):
        tools = _TOOLS
        history: list[dict] = [{"role": "system", "content": _text(rng)}, {"role": "user", "content": _text(rng)}]
        for _ in range(rng.randint(1, 4)):
            target = _assistant(rng, tools)
            prompt, completion = _assistant_completion(renderer, oracle, history, tools, target, effort)
            request = _request(history, tools)
            parsed = renderer.parse_assistant(request, completion, "", "stop")
            response = dict(parsed.response_message)

            assert response["content"] == target["content"]
            assert response.get("reasoning_content", "") == target.get("reasoning_content", "")
            expected_calls = target.get("tool_calls") or []
            got_calls = response.get("tool_calls") or []
            assert [c["function"]["name"] for c in got_calls] == [c["function"]["name"] for c in expected_calls]
            for got, want in zip(got_calls, expected_calls):
                assert json.loads(got["function"]["arguments"]) == json.loads(want["function"]["arguments"])

            # A stripped stop string parses identically (call ids hash the
            # completion ids, so they legitimately differ), including the
            # Fireworks completion shape that keeps only the stop's leading ids.
            for kept in range(len(renderer._stop_ids)):
                variant = completion[: len(completion) - len(renderer._stop_ids) + kept]
                assert _without_call_ids(renderer.parse_assistant(request, variant, "", "stop").message) == (
                    _without_call_ids(parsed.message)
                )

            history.append(response)
            for call in got_calls:
                history.append({"role": "tool", "tool_call_id": call["id"], "content": _text(rng)})
            if not got_calls or rng.random() < 0.3:
                history.append({"role": "user", "content": _text(rng)})
            extended = list(renderer.render_conversation_tokens(_request(history, tools)))
            checkpoint = prompt + completion
            assert extended[: len(checkpoint)] == checkpoint, "full-history re-render drifted from the sampled turn"
            # Pro's generation_config eos is <|end_of_msg|>; a sampler that stops
            # there instead of at the stop string keeps the same lineage.
            eos_stopped = completion + [renderer._end_of_msg_id]
            assert extended[: len(prompt) + len(eos_stopped)] == prompt + eos_stopped
            assert _without_call_ids(renderer.parse_assistant(request, eos_stopped, "", "stop").message) == (
                _without_call_ids(parsed.message)
            )
            turns += 1
    assert turns > 100


# ---------------------------------------------------------------------------
# Parser edge cases
# ---------------------------------------------------------------------------


def _xtml(renderer, text: str) -> list[int]:
    return list(renderer._encode_piece(text, allow_special=True))


_CALL = f'{_OPEN}call tool="act" index="1"{_SEP}{_OPEN}argument key="command" type="string"{_SEP}look{_CLOSE}argument{_SEP}{_CLOSE}call{_SEP}'
_RESPONSE = f"plan{_CLOSE}think{_SEP}{_OPEN}response{_SEP}ok{_CLOSE}response{_SEP}"


def test_length_stop_inside_reasoning_is_reasoning_only(renderers) -> None:
    renderer = renderers[None]
    parsed = renderer.parse_assistant(
        _request([{"role": "user", "content": "go"}], _TOOLS), _xtml(renderer, "half a th"), "", "length"
    )
    assert dict(parsed.message)["reasoning_content"] == "half a th"
    assert dict(parsed.message)["content"] == ""
    assert parsed.output_kind == "reasoning"


def test_partial_stop_ids_parse_as_a_clean_stop(renderers) -> None:
    """Completions may strip the stop text but keep ``<|close|>`` in completion ids."""
    renderer = renderers[None]
    request = _request([{"role": "user", "content": "go"}], _TOOLS)
    ids = _xtml(renderer, f"{_RESPONSE}{_OPEN}tools{_SEP}{_CALL}{_CLOSE}tools{_SEP}{_CLOSE}")
    message = dict(renderer.parse_assistant(request, ids, "", "stop").message)
    assert message["content"] == "ok"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "look"}


def test_empty_reasoning_matches_the_harness_echo(renderers) -> None:
    """An empty think block parses to ``reasoning_content == ""``.

    Pi echoes such a turn back with ``reasoning_content: ""``; the engine
    compares canonical messages before tokens, so a parsed message without
    the field would split every following turn into a ``history_rewrite``
    segment. The echo must also keep the lineage.
    """
    renderer = renderers[None]
    history = [{"role": "user", "content": "go"}]
    request = _request(history, _TOOLS)
    empty_think = f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}"
    completion = _xtml(renderer, f"{empty_think}{_OPEN}tools{_SEP}{_CALL}{_CLOSE}tools{_SEP}")
    parsed = renderer.parse_assistant(request, completion, "", "stop")
    stored = _without_call_ids(parsed.message)
    assert stored["reasoning_content"] == ""
    assert parsed.output_kind == "tool_calls"

    call_id = parsed.message["tool_calls"][0]["id"]
    echoed = {**json.loads(json.dumps(dict(parsed.message), default=dict)), "reasoning_content": ""}
    nxt = _request([*history, echoed, {"role": "tool", "tool_call_id": call_id, "content": "a room"}], _TOOLS)
    assert nxt.messages[1] == parsed.message
    prompt = list(renderer.render_conversation_tokens(request))
    extended = list(renderer.render_conversation_tokens(nxt))
    assert extended[: len(prompt) + len(completion)] == prompt + completion

    no_tools = renderer.parse_assistant(request, _xtml(renderer, f"{empty_think}{_CLOSE}message{_SEP}"), "", "stop")
    assert dict(no_tools.message) == {"role": "assistant", "content": "", "reasoning_content": ""}


def test_length_stop_inside_response_keeps_the_reasoning_field(renderers) -> None:
    renderer = renderers[None]
    request = _request([{"role": "user", "content": "go"}], _TOOLS)
    for reasoning in ("plan", ""):
        ids = _xtml(renderer, f"{reasoning}{_CLOSE}think{_SEP}{_OPEN}response{_SEP}half an ans")
        message = dict(renderer.parse_assistant(request, ids, "", "length").message)
        assert message == {"role": "assistant", "content": "half an ans", "reasoning_content": reasoning}


def test_length_stop_at_any_token_keeps_lineage_or_discards(renderers) -> None:
    """Every accepted length-stopped prefix re-renders as the exact stored tokens.

    A stop inside a multi-token closing tag leaves control tokens in a text
    channel; those turns must be rejected rather than stored as text.
    """
    renderer = renderers[None]
    history = [{"role": "user", "content": "go"}]
    request = _request(history, _TOOLS)
    prompt = list(renderer.render_conversation_tokens(request))
    full = _xtml(renderer, f"{_RESPONSE}{_OPEN}tools{_SEP}{_CALL}{_CLOSE}tools{_SEP}")
    accepted = 0
    for cut in range(1, len(full) + 1):
        partial = full[:cut]
        try:
            parsed = renderer.parse_assistant(request, partial, "", "length")
        except TITOParseError:
            continue
        accepted += 1
        message = json.loads(json.dumps(dict(parsed.message), default=dict))
        follow = [{"role": "tool", "tool_call_id": call["id"], "content": "a room"} for call in message.get("tool_calls") or []]
        follow = follow or [{"role": "user", "content": "go on"}]
        extended = list(renderer.render_conversation_tokens(_request([*history, message, *follow], _TOOLS)))
        assert extended[: len(prompt) + cut] == prompt + partial, (cut, renderer._decode(partial[-3:]))
    assert accepted >= 3


def test_length_stop_inside_tools_discards(renderers) -> None:
    renderer = renderers[None]
    ids = _xtml(
        renderer, f'{_RESPONSE}{_OPEN}tools{_SEP}{_OPEN}call tool="act" index="1"{_SEP}{_OPEN}argument key="command"'
    )
    with pytest.raises(TITOParseError) as caught:
        renderer.parse_assistant(_request([{"role": "user", "content": "go"}], _TOOLS), ids, "", "length")
    assert caught.value.reason_code == "tool_unparsed_boundary"


@pytest.mark.parametrize(
    ("completion", "reason"),
    [
        (
            f"{_RESPONSE}{_OPEN}tools{_SEP}{_CALL.replace('act', 'fly')}{_CLOSE}tools{_SEP}{_CLOSE}message{_SEP}",
            "tool_unknown_name",
        ),
        (
            f'{_RESPONSE}{_OPEN}tools{_SEP}{_OPEN}call tool="act" index="1"{_SEP}{_OPEN}argument key="n" type="number"{_SEP}abc'
            f"{_CLOSE}argument{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}{_CLOSE}message{_SEP}",
            "tool_invalid_argument",
        ),
        (
            f"plan{_CLOSE}think{_SEP}{_OPEN}response{_SEP}a{_SEP}b{_CLOSE}response{_SEP}{_CLOSE}message{_SEP}",
            "control_token_in_text",
        ),
        (f"plan{_CLOSE}message{_SEP}", "reasoning_missing_close"),
        *(
            (
                f'{_RESPONSE}{_OPEN}tools{_SEP}{_OPEN}call tool="act" index="1"{_SEP}{_OPEN}json{_SEP}{raw}'
                f"{_CLOSE}json{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}{_CLOSE}message{_SEP}",
                reason,
            )
            for raw, reason in (
                ("not json", "tool_arguments_invalid_json"),
                ("  ", "tool_arguments_invalid_json"),
                ("[1, 2]", "tool_arguments_not_object"),
                ('{"command": "look"}', "tool_unparsed_content"),
            )
        ),
        (f"{_RESPONSE}stray{_CLOSE}message{_SEP}", "tool_unparsed_content"),
    ],
)
def test_malformed_completions_fail_with_stable_reason(completion, reason, renderers) -> None:
    renderer = renderers[None]
    with pytest.raises(TITOParseError) as caught:
        renderer.parse_assistant(
            _request([{"role": "user", "content": "go"}], _TOOLS), _xtml(renderer, completion), "", "stop"
        )
    assert caught.value.reason_code == reason


def test_media_and_dynamic_template_fields_are_rejected(renderers) -> None:
    renderer = renderers[None]
    image = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    with pytest.raises(TITOError):
        renderer.render_conversation_tokens(_request([{"role": "user", "content": image}], []))
    request = TITOChatRequest(
        messages=({"role": "user", "content": "hi"},),
        sampling_fields={"reasoning_effort": "low"},
    )
    with pytest.raises(TITOError):
        renderer.render_conversation_tokens(request)
