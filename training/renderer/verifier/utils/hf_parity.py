"""CPU-only renderer ↔ HuggingFace chat-template parity check.

This is the second of two probes in the verifier. The first (``probe.py``)
runs the renderer against a live deployment and records empirical
provenance per token in a JSON artifact. This module compares the
renderer against the *upstream* canonical tokenization shipped on
HuggingFace (``tokenizer.apply_chat_template``) and returns a
structured result the test harness asserts against.

Why have both:

* Live (probe.py) catches gateway↔renderer drift, real-model emission
  shape, and weight intent vs empirical provenance. It needs a
  deployment.
* CPU (this module) catches renderer↔upstream-HF drift independently
  of any serving stack. No deployment, runs every PR. With both
  layers, a "live FAIL" can be triaged into "renderer is wrong" vs
  "gateway is stale" by reading the CPU result.

The output is a small dict, not a JSON artifact — assertion-style
output keeps CPU tests in their natural shape (pytest failures with a
clear first-divergence message).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Importing the cookbook renderer package registers cookbook-local
# renderer names (glm5, gemma4, minimax_m2, nemotron) under
# ``tinker_cookbook.renderers.get_renderer``.
import training.renderer  # noqa: F401
from training.utils.supervised import normalize_messages
from training.utils.supervised import parse_train_on_what
from training.utils.supervised import render_messages_to_datums


@dataclasses.dataclass
class HFParityResult:
    renderer_tokens: list[int]
    hf_tokens: list[int]
    match: bool
    first_divergence_idx: int | None
    renderer_decoded: list[str]
    hf_decoded: list[str]


@dataclasses.dataclass
class HFSupervisedParityResult:
    examples: list[HFParityResult]
    match: bool
    first_mismatch_example: int | None


def _decode_each(tokenizer: Any, ids: list[int]) -> list[str]:
    """Decode token IDs one at a time so the divergence dump is readable."""
    out: list[str] = []
    for i in ids:
        try:
            out.append(tokenizer.decode([int(i)], skip_special_tokens=False))
        except Exception:  # noqa: BLE001 - decode failures shouldn't mask the parity result
            out.append("")
    return out


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where the two token lists diverge.

    Returns the length of the shorter list when one is a strict prefix of
    the other, ``None`` when they are equal.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _hf_tokens(
    tokenizer: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    tools: list[dict] | None = None,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    kwargs.update(apply_chat_template_kwargs or {})
    hf_result = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(hf_result, "input_ids"):
        return [int(t) for t in list(hf_result.input_ids)]  # type: ignore[arg-type]
    return [int(t) for t in list(hf_result)]  # type: ignore[arg-type]


_SPLIT_REQUIRED_TRAINING_MODES = {
    "all_assistant_messages",
    "all_messages",
    "all_tokens",
    "all_user_and_system_messages",
    "customized",
}


def _expected_supervised_prefixes(
    *,
    renderer: Any,
    messages: list[dict],
    train_on_what: Any,
) -> list[list[dict]]:
    train_mode = parse_train_on_what(train_on_what)
    if any(("weight" in m) or ("trainable" in m) for m in messages):
        train_mode = parse_train_on_what("customized")
    if getattr(renderer, "has_extension_property", False):
        return [messages]
    if train_mode.value not in _SPLIT_REQUIRED_TRAINING_MODES:
        return [messages]
    user_message_idxs = [
        idx for idx, message in enumerate(messages) if message.get("role") == "user"
    ]
    if len(user_message_idxs) <= 1:
        return [messages]
    return [
        messages[:next_user_idx]
        for next_user_idx in [*user_message_idxs[1:], len(messages)]
    ]


def _build_result(
    tokenizer: Any,
    renderer_tokens: list[int],
    hf_tokens: list[int],
) -> HFParityResult:
    first_div = _first_divergence(renderer_tokens, hf_tokens)
    return HFParityResult(
        renderer_tokens=renderer_tokens,
        hf_tokens=hf_tokens,
        match=first_div is None,
        first_divergence_idx=first_div,
        renderer_decoded=_decode_each(tokenizer, renderer_tokens),
        hf_decoded=_decode_each(tokenizer, hf_tokens),
    )


def compare_renderer_to_hf(
    *,
    renderer_name: str,
    tokenizer_model: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> HFParityResult:
    """Compare ``renderer.build_generation_prompt`` (or full conversation)
    tokens to ``tokenizer.apply_chat_template`` for the same input.

    The function intentionally has no side effects beyond loading the
    tokenizer — caller decides what to do with a mismatch.
    """
    tokenizer = get_tokenizer(tokenizer_model)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"Tokenizer {tokenizer_model!r} has no chat_template; "
            "HF parity comparison is not meaningful."
        )

    renderer = get_renderer(renderer_name, tokenizer)
    normalized = normalize_messages(messages)

    if add_generation_prompt:
        renderer_input = renderer.build_generation_prompt(normalized, role="assistant")
    else:
        renderer_input, _weights = renderer.build_supervised_example(normalized)
    renderer_tokens = [int(t) for t in renderer_input.to_ints()]

    hf_tokens = _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    return _build_result(tokenizer, renderer_tokens, hf_tokens)


def compare_supervised_rendering_to_hf(
    *,
    renderer_name: str,
    tokenizer_model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    train_on_what: Any = "all_assistant_messages",
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> HFSupervisedParityResult:
    """Compare SFT ``render_messages_to_datums`` outputs to HF tokenization.

    Multi-turn renderers that disaggregate training examples are compared one
    split prefix at a time, which catches missing per-prefix tool declarations
    and thinking-template drift.
    """
    tokenizer = get_tokenizer(tokenizer_model)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"Tokenizer {tokenizer_model!r} has no chat_template; "
            "HF parity comparison is not meaningful."
        )

    renderer = get_renderer(renderer_name, tokenizer)
    rendered_examples = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what=train_on_what,
        tools=tools,
    )
    prefixes = _expected_supervised_prefixes(
        renderer=renderer,
        messages=messages,
        train_on_what=train_on_what,
    )
    if len(rendered_examples) != len(prefixes):
        raise RuntimeError(
            f"rendered example count mismatch: {len(rendered_examples)} != {len(prefixes)}"
        )

    example_results: list[HFParityResult] = []
    for rendered, prefix_messages in zip(rendered_examples, prefixes, strict=True):
        hf_tokens = _hf_tokens(
            tokenizer,
            prefix_messages,
            add_generation_prompt=False,
            tools=tools,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        example_results.append(_build_result(tokenizer, rendered.token_ids, hf_tokens))

    first_mismatch = next(
        (idx for idx, result in enumerate(example_results) if not result.match),
        None,
    )
    return HFSupervisedParityResult(
        examples=example_results,
        match=first_mismatch is None,
        first_mismatch_example=first_mismatch,
    )


def format_divergence(result: HFParityResult, *, context: int = 6) -> str:
    """Render a human-readable failure message for an assertion."""
    if result.match:
        return "tokens match"

    idx = result.first_divergence_idx or 0
    lo = max(0, idx - context)
    hi_r = min(len(result.renderer_tokens), idx + context + 1)
    hi_h = min(len(result.hf_tokens), idx + context + 1)

    lines: list[str] = []
    lines.append(
        f"renderer tokens={len(result.renderer_tokens)}  "
        f"hf tokens={len(result.hf_tokens)}  "
        f"first divergence at idx={idx}"
    )
    lines.append(f"renderer[{lo}:{hi_r}]:")
    for i in range(lo, hi_r):
        marker = "→" if i == idx else " "
        tok = result.renderer_tokens[i] if i < len(result.renderer_tokens) else None
        dec = result.renderer_decoded[i] if i < len(result.renderer_decoded) else ""
        lines.append(f"  {marker} idx={i}  tok={tok}  decoded={dec!r}")
    lines.append(f"hf[{lo}:{hi_h}]:")
    for i in range(lo, hi_h):
        marker = "→" if i == idx else " "
        tok = result.hf_tokens[i] if i < len(result.hf_tokens) else None
        dec = result.hf_decoded[i] if i < len(result.hf_decoded) else ""
        lines.append(f"  {marker} idx={i}  tok={tok}  decoded={dec!r}")
    return "\n".join(lines)


def format_supervised_divergence(
    result: HFSupervisedParityResult,
    *,
    context: int = 6,
) -> str:
    if result.match:
        return "all supervised examples match"
    idx = result.first_mismatch_example or 0
    return f"example {idx} diverged:\n{format_divergence(result.examples[idx], context=context)}"
