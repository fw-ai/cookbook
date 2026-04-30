"""CPU-only renderer ↔ HuggingFace chat-template parity tests.

This is the CI gate for the verifier's CPU layer. Each parametrized
case loads the upstream tokenizer, asks ``apply_chat_template`` for the
canonical token sequence, and asserts byte-for-byte equality against
``renderer.build_generation_prompt``. Tests skip cleanly when the
upstream tokenizer can't be downloaded (network outage, gated repo),
so CI without HF Hub access still completes — but the moment the
tokenizer is available, the test runs and asserts.

Cases marked ``xfail`` track known divergences that the empirical
sweep surfaced (``renderer-verifier-findings.md``); they exist so that
a fix flips the test green automatically and a new regression in the
opposite direction goes red.

The live empirical probe (``training/renderer/verifier/probe.py``) is the
complementary layer: it needs a live deployment, runs nightly, and
emits JSON artifacts. CPU and live answer different questions:

* CPU: does the renderer match upstream HF's canonical tokenization?
* live: does the renderer match what the deployed model actually does?

A renderer can pass one and still fail the other; both are needed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from training.renderer.verifier.utils.hf_parity import (
    HFParityResult,
    compare_renderer_to_hf,
    format_divergence,
)


@dataclasses.dataclass
class _Case:
    case_id: str
    renderer: str
    tokenizer_model: str
    messages: list[dict]
    apply_chat_template_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    xfail_reason: str | None = None  # if set, mark the test xfail with this reason


_SHORT_MSGS = [
    {"role": "system", "content": "Answer with a single integer and nothing else."},
    {"role": "user", "content": "2 + 2 = ?"},
]
_MULTI_TURN_MSGS = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "And 3+3?"},
]


_CASES: list[_Case] = [
    _Case(
        case_id="glm5-single-turn",
        renderer="glm5",
        tokenizer_model="zai-org/GLM-5.1",
        messages=_SHORT_MSGS,
    ),
    _Case(
        case_id="glm5-multi-turn",
        renderer="glm5",
        tokenizer_model="zai-org/GLM-5.1",
        messages=_MULTI_TURN_MSGS,
    ),
    _Case(
        case_id="qwen3-thinking-single-turn",
        renderer="qwen3",
        tokenizer_model="Qwen/Qwen3-8B",
        messages=_SHORT_MSGS,
    ),
    _Case(
        case_id="qwen3-thinking-multi-turn",
        renderer="qwen3",
        tokenizer_model="Qwen/Qwen3-8B",
        messages=_MULTI_TURN_MSGS,
    ),
    _Case(
        case_id="qwen3-disable-thinking",
        renderer="qwen3_disable_thinking",
        tokenizer_model="Qwen/Qwen3-8B",
        messages=_SHORT_MSGS,
        apply_chat_template_kwargs={"enable_thinking": False},
    ),
    _Case(
        case_id="kimi_k25-single-turn",
        renderer="kimi_k25",
        tokenizer_model="moonshotai/Kimi-K2.5",
        messages=_SHORT_MSGS,
    ),
    _Case(
        case_id="minimax_m2-single-turn",
        renderer="minimax_m2",
        tokenizer_model="MiniMaxAI/MiniMax-M2",
        messages=_SHORT_MSGS,
        xfail_reason=(
            "renderer emits an extra '\\n' after <think> in the generation "
            "suffix — surfaced by the empirical sweep against "
            "accounts/fireworks/models/minimax-m2p7. "
            "See workspace_batching/renderer-verifier-findings.md."
        ),
    ),
]


_PARAM = pytest.mark.parametrize("case", _CASES, ids=[c.case_id for c in _CASES])


@_PARAM
@pytest.mark.timeout(180)
def test_renderer_matches_hf_chat_template(case: _Case) -> None:
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)

    try:
        result: HFParityResult = compare_renderer_to_hf(
            renderer_name=case.renderer,
            tokenizer_model=case.tokenizer_model,
            messages=case.messages,
            add_generation_prompt=True,
            apply_chat_template_kwargs=case.apply_chat_template_kwargs,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        # Network outage, gated repo, missing chat_template, or a
        # tokenizer-class registration issue. Skip cleanly so CI without
        # HF Hub access doesn't go red on infra problems.
        pytest.skip(f"tokenizer unavailable for {case.tokenizer_model!r}: {exc}")

    assert result.match, format_divergence(result)
