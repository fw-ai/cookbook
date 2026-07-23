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

from training.utils.rl.rollout.renderer import (
    build_multimodal_completions_prompt_token_ids,
)

from training.renderer.verifier.utils.hf_parity import (
    HFParityResult,
    compare_renderer_to_hf,
    format_divergence,
)
from training.renderer.thinking_trace import get_thinking_trace_model_capability


@dataclasses.dataclass
class _Case:
    case_id: str
    renderer: str
    tokenizer_model: str
    messages: list[dict]
    tokenizer_revision: str | None = None
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
_KIMI_REASONING_MSGS = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "reasoning_content": "2+2 is basic arithmetic.",
        "content": "4.",
    },
    {"role": "user", "content": "And 3+3?"},
]
def _verified_source(alias: str) -> tuple[str, str]:
    capability = get_thinking_trace_model_capability(alias)
    assert capability is not None
    return capability.tokenizer.repo, capability.tokenizer.revision


_GLM51_MODEL, _GLM51_REVISION = _verified_source("zai-org/GLM-5.1")
_GLM52_MODEL, _GLM52_REVISION = _verified_source("zai-org/GLM-5.2")
_QWEN35_MODEL, _QWEN35_REVISION = _verified_source("Qwen/Qwen3.5-35B-A3B")
_QWEN36_MODEL, _QWEN36_REVISION = _verified_source("Qwen/Qwen3.6-27B")
_KIMI25_MODEL, _KIMI25_REVISION = _verified_source("moonshotai/Kimi-K2.5")
_KIMI26_MODEL, _KIMI26_REVISION = _verified_source("moonshotai/Kimi-K2.6")
_KIMI27_MODEL, _KIMI27_REVISION = _verified_source("moonshotai/Kimi-K2.7-Code")
_ONE_PIXEL_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


_CASES: list[_Case] = [
    _Case(
        case_id="glm5-single-turn",
        renderer="glm5_interleaved",
        tokenizer_model=_GLM51_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_GLM51_REVISION,
    ),
    _Case(
        case_id="glm5-multi-turn",
        renderer="glm5_interleaved",
        tokenizer_model=_GLM51_MODEL,
        messages=_MULTI_TURN_MSGS,
        tokenizer_revision=_GLM51_REVISION,
    ),
    _Case(
        case_id="glm_moe_dsa-single-turn",
        renderer="glm_moe_dsa_interleaved",
        tokenizer_model=_GLM52_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_GLM52_REVISION,
        apply_chat_template_kwargs={"reasoning_effort": "max"},
    ),
    _Case(
        case_id="glm_moe_dsa-multi-turn",
        renderer="glm_moe_dsa_interleaved",
        tokenizer_model=_GLM52_MODEL,
        messages=_MULTI_TURN_MSGS,
        tokenizer_revision=_GLM52_REVISION,
        apply_chat_template_kwargs={"reasoning_effort": "max"},
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
        case_id="qwen3_5-thinking-single-turn",
        renderer="qwen3_5_interleaved",
        tokenizer_model=_QWEN35_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_QWEN35_REVISION,
    ),
    _Case(
        case_id="qwen3_5-disable-thinking",
        renderer="qwen3_5_disable_thinking_interleaved",
        tokenizer_model=_QWEN35_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_QWEN35_REVISION,
        apply_chat_template_kwargs={"enable_thinking": False},
    ),
    # Qwen3.6 aliases the qwen3_5 renderer family
    # (see training/renderer/_qwen3_split.py). Validate parity against the
    # 3.6 tokenizer's `apply_chat_template` for default invocation
    # (no `preserve_thinking` kwarg → byte-identical to Qwen3.5).
    _Case(
        case_id="qwen3_6-thinking-single-turn",
        renderer="qwen3_6_interleaved",
        tokenizer_model=_QWEN36_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_QWEN36_REVISION,
    ),
    _Case(
        case_id="qwen3_6-thinking-multi-turn",
        renderer="qwen3_6_interleaved",
        tokenizer_model=_QWEN36_MODEL,
        messages=_MULTI_TURN_MSGS,
        tokenizer_revision=_QWEN36_REVISION,
    ),
    _Case(
        case_id="qwen3_6-disable-thinking",
        renderer="qwen3_6_disable_thinking_interleaved",
        tokenizer_model=_QWEN36_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_QWEN36_REVISION,
        apply_chat_template_kwargs={"enable_thinking": False},
    ),
    # Qwen3.6 preserve-thinking against an assistant message that has
    # NO reasoning content. The official Qwen3.6 chat template's
    # preserve_thinking=true branch emits an empty
    # `<think>\n\n</think>\n\n` wrapper for every historical assistant
    # regardless of reasoning content (the "empty thinking blocks spam
    # context" pattern documented at
    # https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates).
    # `Qwen3_6PreserveThinkingSplitRenderer._assistant_header_suffix`
    # deliberately mirrors that behavior to preserve byte-level
    # train-inference parity with the stock HF chat template.
    _Case(
        case_id="qwen3_6-preserve-thinking-multi-turn",
        renderer="qwen3_6_preserved",
        tokenizer_model=_QWEN36_MODEL,
        messages=_MULTI_TURN_MSGS,
        tokenizer_revision=_QWEN36_REVISION,
        apply_chat_template_kwargs={"preserve_thinking": True},
    ),
    _Case(
        case_id="kimi_k25-single-turn",
        renderer="kimi_k25_interleaved",
        tokenizer_model=_KIMI25_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_KIMI25_REVISION,
    ),
    _Case(
        case_id="kimi_k26-interleaved-single-turn",
        renderer="kimi_k26_interleaved",
        tokenizer_model=_KIMI26_MODEL,
        messages=_SHORT_MSGS,
        tokenizer_revision=_KIMI26_REVISION,
    ),
    _Case(
        case_id="kimi_k27_code-preserve-thinking-multi-turn",
        renderer="kimi_k27_code_preserved",
        tokenizer_model=_KIMI27_MODEL,
        messages=_KIMI_REASONING_MSGS,
        tokenizer_revision=_KIMI27_REVISION,
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
            "See skills/fireworks-training/references/renderer-verification.md."
        ),
    ),
    _Case(
        case_id="minimax_m3-single-turn",
        renderer="minimax_m3",
        tokenizer_model="MiniMaxAI/MiniMax-M3",
        messages=_SHORT_MSGS,
        apply_chat_template_kwargs={"thinking_mode": "adaptive"},
    ),
    _Case(
        case_id="minimax_m3-multi-turn",
        renderer="minimax_m3",
        tokenizer_model="MiniMaxAI/MiniMax-M3",
        messages=_MULTI_TURN_MSGS,
        apply_chat_template_kwargs={"thinking_mode": "adaptive"},
    ),
]


_PARAM = pytest.mark.parametrize("case", _CASES, ids=[c.case_id for c in _CASES])


@_PARAM
@pytest.mark.timeout(180)
def test_renderer_matches_hf_chat_template(case: _Case) -> None:
    if case.xfail_reason:
        pytest.xfail(case.xfail_reason)

    try:
        capability = get_thinking_trace_model_capability(case.tokenizer_model)
        result: HFParityResult = compare_renderer_to_hf(
            renderer_name=case.renderer,
            tokenizer_model=case.tokenizer_model,
            tokenizer_revision=case.tokenizer_revision,
            tokenizer_trust_remote_code=(
                capability.tokenizer.trust_remote_code
                if capability is not None
                else None
            ),
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


@pytest.mark.timeout(180)
def test_qwen3_5_multimodal_adapter_matches_independent_hf_ground_truth() -> None:
    """The serving adapter must transport, not re-render, renderer output.

    Tinker's disable-thinking renderer and HuggingFace's official template
    with ``enable_thinking=False`` are independent producers of the expected
    prompt.  Re-running the HF template with its defaults recreates the
    production regression by appending a different thinking prefix.
    """
    try:
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        import training.renderer  # noqa: F401 (register local renderer overrides)
        from training.utils.supervised import normalize_messages

        tokenizer = get_tokenizer(_QWEN35_MODEL)
        renderer = get_renderer(
            "qwen3_5_disable_thinking",
            tokenizer,
            image_processor=get_image_processor(_QWEN35_MODEL),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"Qwen3.5 tokenizer/image processor unavailable: {exc}")

    renderer_messages = [
        {"role": "system", "content": "Answer briefly."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _ONE_PIXEL_PNG},
                {"type": "text", "text": "Where am I?"},
            ],
        },
    ]
    hf_messages = [
        {"role": "system", "content": "Answer briefly."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _ONE_PIXEL_PNG}},
                {"type": "text", "text": "Where am I?"},
            ],
        },
    ]

    model_input = renderer.build_generation_prompt(
        normalize_messages(renderer_messages),
        role="assistant",
    )
    adapter_tokens, images = build_multimodal_completions_prompt_token_ids(
        renderer_messages,
        model_input,
        tokenizer,
    )
    hf_disabled = tokenizer.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    hf_default = tokenizer.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    disabled_tokens = [int(t) for t in hf_disabled.input_ids]
    default_tokens = [int(t) for t in hf_default.input_ids]

    assert adapter_tokens == disabled_tokens
    assert adapter_tokens != default_tokens
    assert len(images) == 1
    assert images[0].startswith("data:image/jpeg;base64,")
    assert images[0] != _ONE_PIXEL_PNG
    assert tokenizer.decode(adapter_tokens, skip_special_tokens=False).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert tokenizer.decode(default_tokens, skip_special_tokens=False).endswith(
        "<|im_start|>assistant\n<think>\n"
    )
