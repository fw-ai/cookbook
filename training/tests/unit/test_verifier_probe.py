"""Smoke test for ``training.verifier.probe.run_probe``.

Network-free: a stub Fireworks-like client returns canned token IDs and
text. The test exercises the whole probe pipeline (render, fake API call,
full-conversation render, alignment, audit table) and asserts the
artifact's invariants — schema, sanity flags, provenance partition.

The renderer used is a tiny custom one registered just for this test, so
the test does not depend on any specific HuggingFace model being
downloadable in CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import tinker
from tinker_cookbook.renderers import register_renderer, unregister_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TrainOnWhat,
)

from training.verifier.probe import (
    _PROV_NATIVE,
    _PROV_PROMPT,
    _PROV_TRAILING,
    SCHEMA_VERSION,
    run_probe,
)


# Token IDs picked so each role / chunk produces a distinct token.
_T = {
    "<sys>": 100,
    "<user>": 101,
    "<asst>": 102,
    "<eot>": 103,
    "hello": 200,
    "world": 201,
    "fine": 202,
    "thanks": 203,
    " ": 250,  # placeholder; not used in this test
}


class _StubTokenizer:
    """Maps short test strings to canned IDs and back. Just enough surface
    area to satisfy the probe's tokenizer probing."""

    name_or_path = "test-tokenizer"
    all_special_ids = [_T["<sys>"], _T["<user>"], _T["<asst>"], _T["<eot>"]]
    additional_special_tokens_ids: list[int] = []
    added_tokens_decoder: dict[int, Any] = {}

    _id_to_str = {v: k for k, v in _T.items()}
    _str_to_ids = {
        "hello world": [_T["hello"], _T["world"]],
        "fine thanks": [_T["fine"], _T["thanks"]],
        "hello": [_T["hello"]],
        "world": [_T["world"]],
        "fine": [_T["fine"]],
        "thanks": [_T["thanks"]],
        "": [],
    }

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        if s in self._str_to_ids:
            return list(self._str_to_ids[s])
        # Fall back to char-by-char for unknown strings; tests use known ones.
        raise KeyError(f"unexpected encode input: {s!r}")

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._id_to_str.get(int(i), f"<{i}>") for i in ids)


class _ToyRenderer(Renderer):
    """Three-role toy renderer: <sys>{c}, <user>{c}, <asst>{c}<eot>.

    The trailing ``<eot>`` is *only* emitted as ``stop_overlap`` on the
    last assistant turn — mirroring the GLM5-style trailing-token shape
    the probe needs to disambiguate.
    """

    _bos_tokens = []  # type: ignore[assignment]

    def get_stop_sequences(self) -> list[int]:
        return [_T["<eot>"]]

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        content = message["content"] or ""
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)

        if role == "system":
            header = tinker.types.EncodedTextChunk(tokens=[_T["<sys>"]])
            output = [tinker.types.EncodedTextChunk(tokens=content_ids)]
            return RenderedMessage(output=output, header=header)
        if role == "user":
            header = tinker.types.EncodedTextChunk(tokens=[_T["<user>"]])
            output = [tinker.types.EncodedTextChunk(tokens=content_ids)]
            return RenderedMessage(output=output, header=header)
        # assistant
        header = tinker.types.EncodedTextChunk(tokens=[_T["<asst>"]])
        output = [tinker.types.EncodedTextChunk(tokens=content_ids)]
        stop_overlap = tinker.types.EncodedTextChunk(tokens=[_T["<eot>"]])
        return RenderedMessage(output=output, header=header, stop_overlap=stop_overlap)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:  # pragma: no cover - unused in probe
        text = self.tokenizer.decode(response, skip_special_tokens=True)
        return Message(role="assistant", content=text), True


def _toy_factory(tokenizer, image_processor=None):
    del image_processor
    return _ToyRenderer(tokenizer)


@pytest.fixture
def toy_renderer():
    register_renderer("__verifier_test_toy", _toy_factory)
    try:
        yield "__verifier_test_toy"
    finally:
        unregister_renderer("__verifier_test_toy")


class _StubClient:
    """Returns prompt + completion token IDs that match the toy renderer's
    own render of the conversation, so the probe's alignment check
    classifies tokens cleanly."""

    def __init__(self, prompt_token_ids: list[int], completion_token_ids: list[int], completion_text: str):
        self._prompt_token_ids = prompt_token_ids
        self._completion_token_ids = completion_token_ids
        self._completion_text = completion_text

        # Mimic the .chat.completions.create surface
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any):
        # The probe sends echo=True, raw_output=True, return_token_ids=True
        return SimpleNamespace(
            model_dump=lambda: {
                "prompt_token_ids": self._prompt_token_ids,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": self._completion_text},
                        "finish_reason": "stop",
                        "raw_output": {
                            "completion_token_ids": self._completion_token_ids,
                        },
                    }
                ],
            }
        )


def test_run_probe_artifact_shape_and_provenance(toy_renderer):
    tokenizer = _StubTokenizer()

    messages = [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "world"},
    ]

    # The toy renderer's prompt for these messages would be:
    #   <sys> hello <user> world <asst>
    expected_prompt_ids = [
        _T["<sys>"], _T["hello"],
        _T["<user>"], _T["world"],
        _T["<asst>"],
    ]
    completion_text = "fine thanks"
    expected_completion_ids = [_T["fine"], _T["thanks"]]

    client = _StubClient(
        prompt_token_ids=expected_prompt_ids,
        completion_token_ids=expected_completion_ids,
        completion_text=completion_text,
    )

    artifact = run_probe(
        renderer_name=toy_renderer,
        tokenizer=tokenizer,
        client=client,
        model="test/model",
        messages=messages,
        max_tokens=16,
        temperature=0.0,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )

    # Top-level shape
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["kind"] == "probe"
    assert artifact["renderer"]["name"] == toy_renderer

    # Prompt parity: renderer agrees with API
    assert artifact["sanity"]["renderer_prompt_matches_api_prompt"] is True
    assert artifact["sanity"]["full_render_prompt_prefix_matches_api"] is True
    assert artifact["sanity"]["tokenization_diverged_count"] == 0

    # Full render = prompt + completion + <eot> (stop_overlap on last assistant turn)
    full_tokens = artifact["render"]["full"]["tokens"]
    assert full_tokens == expected_prompt_ids + expected_completion_ids + [_T["<eot>"]]

    # Audit table partition over provenance
    audit = artifact["audit_table"]
    assert len(audit) == len(full_tokens)

    prov_counts: dict[str, int] = {}
    for row in audit:
        prov_counts[row["provenance"]] = prov_counts.get(row["provenance"], 0) + 1

    assert prov_counts[_PROV_PROMPT] == len(expected_prompt_ids)
    assert prov_counts[_PROV_NATIVE] == len(expected_completion_ids)
    # The trailing <eot> from stop_overlap is the diagnostic position:
    # renderer claims weight 1.0, but the model never emitted it.
    assert prov_counts[_PROV_TRAILING] == 1

    trailing_row = audit[-1]
    assert trailing_row["chunk_source"] == "stop_overlap"
    assert trailing_row["role"] == "assistant"
    assert trailing_row["renderer_claim_weight"] == 1.0
    assert trailing_row["provenance"] == _PROV_TRAILING
    assert trailing_row["token_id"] == _T["<eot>"]


def test_run_probe_flags_tokenization_divergence(toy_renderer):
    """If the renderer re-tokenises the completion differently than the API
    returned, the affected positions are flagged ``tokenization_diverged``."""
    tokenizer = _StubTokenizer()
    messages = [{"role": "user", "content": "hello"}]

    expected_prompt_ids = [_T["<user>"], _T["hello"], _T["<asst>"]]
    # The deployment claims to have emitted ["world", "fine"] for the
    # response text "fine thanks". The renderer re-tokenises "fine thanks"
    # to ["fine", "thanks"], so positions 0 and 1 of the assistant turn
    # disagree.
    api_completion_ids = [_T["world"], _T["fine"]]
    completion_text = "fine thanks"

    client = _StubClient(
        prompt_token_ids=expected_prompt_ids,
        completion_token_ids=api_completion_ids,
        completion_text=completion_text,
    )

    artifact = run_probe(
        renderer_name=toy_renderer,
        tokenizer=tokenizer,
        client=client,
        model="test/model",
        messages=messages,
    )

    assert artifact["sanity"]["tokenization_diverged_count"] >= 1
