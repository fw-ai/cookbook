"""Qwen3.5 TITO certification. Parsing is covered by the Qwen3.8 renderer tests."""

from __future__ import annotations

import pytest

from training.renderer.tito import (
    Qwen35TITORenderer,
    TITORendererCertification,
    build_sidecar_tito_renderer,
)
from training.renderer.tito.qwen35 import (
    QWEN35_RENDERER_NAME,
    QWEN35_TOKENIZER_FINGERPRINT,
)
from training.tests.unit.test_tito_qwen38_renderer import (
    _PROBE_TOOL_COMPLETION,
    _QwenTokenizer,
    _request,
)


def _certification() -> TITORendererCertification:
    return TITORendererCertification(
        certification_id="test-qwen35",
        renderer_names=frozenset({QWEN35_RENDERER_NAME}),
        tokenizer_fingerprint="test",
        renderer_factory=lambda tokenizer, certification: Qwen35TITORenderer(
            tokenizer,
            certification=certification,
        ),
    )


def test_qwen35_parses_qwen_tool_calls_under_its_own_name() -> None:
    tokenizer = _QwenTokenizer()
    renderer = Qwen35TITORenderer(tokenizer, certification=_certification())
    completion = tokenizer.encode(_PROBE_TOOL_COMPLETION)

    parsed = renderer.parse_assistant(_request(), completion, "", "stop")

    assert renderer.renderer_id == QWEN35_RENDERER_NAME
    call = parsed.message["tool_calls"][0]["function"]
    assert call["name"] == "execute_python"
    assert parsed.message["reasoning_content"]


def test_qwen35_malformed_tool_call_names_this_model() -> None:
    tokenizer = _QwenTokenizer()
    renderer = Qwen35TITORenderer(tokenizer, certification=_certification())
    text = "thinking\n</think>\n\n<tool_call>\n<function=execute_python>\n"
    with pytest.raises(ValueError, match="unparsed Qwen3.5 tool-call boundary"):
        renderer.parse_assistant(_request(), tokenizer.encode(text), "", "length")


def test_certification_registry_resolves_qwen3_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "training.renderer.tito.shared._tokenizer_fingerprint",
        lambda _tokenizer: QWEN35_TOKENIZER_FINGERPRINT,
    )
    renderer = build_sidecar_tito_renderer(_QwenTokenizer(), QWEN35_RENDERER_NAME)
    assert isinstance(renderer, Qwen35TITORenderer)
    assert renderer.certification_id == "qwen3.5-35b-a3b-fp8@5a3b7d66-v1"
