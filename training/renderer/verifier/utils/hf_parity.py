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
import json
from typing import Any

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Importing the cookbook renderer package registers cookbook-local
# renderer names (glm5, gemma4, minimax_m2, nemotron) under
# ``tinker_cookbook.renderers.get_renderer``.
import training.renderer  # noqa: F401
from training.utils.supervised import (
    build_tool_prefixed_messages,
    renderer_declares_tools,
)


@dataclasses.dataclass
class HFParityResult:
    renderer_tokens: list[int]
    hf_tokens: list[int]
    match: bool
    first_divergence_idx: int | None
    renderer_decoded: list[str]
    hf_decoded: list[str]


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


def _hf_messages_with_normalized_tool_args(
    messages: list[dict], *, require_mapping: bool
) -> list[dict]:
    """Return a reference copy with a template-compatible argument shape.

    OpenAI-compatible inputs may carry arguments as JSON strings. Current Gemma
    4 templates require deserialized objects, while other renderer parity tests
    intentionally exercise the production JSON-string representation.
    """
    out: list[dict] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            out.append(message)
            continue
        new_calls: list[dict] = []
        for call in tool_calls:
            call = dict(call)
            function = dict(call.get("function") or {})
            if "arguments" in function:
                raw = function["arguments"]
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                function["arguments"] = (
                    parsed if require_mapping else json.dumps(parsed)
                )
            call["function"] = function
            new_calls.append(call)
        message = {**message, "tool_calls": new_calls}
        out.append(message)
    return out


def _hf_messages_with_minimax_m3_reasoning_parts(
    messages: list[dict], *, renderer_name: str
) -> list[dict]:
    """Map normalized M3 thinking parts to its HF ``reasoning_content`` field."""
    if renderer_name != "minimax_m3":
        return messages
    out: list[dict] = []
    for message in messages:
        content = message.get("content")
        if (
            message.get("role") != "assistant"
            or "reasoning_content" in message
            or not isinstance(content, list)
        ):
            out.append(message)
            continue
        reasoning = "".join(
            str(part.get("thinking", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "thinking"
        )
        out.append(
            {**message, **({"reasoning_content": reasoning} if reasoning else {})}
        )
    return out


def compare_renderer_to_hf(
    *,
    renderer_name: str,
    tokenizer_model: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> HFParityResult:
    """Compare ``renderer.build_generation_prompt`` (or full conversation)
    tokens to ``tokenizer.apply_chat_template`` for the same input.

    When ``tools`` is provided, both sides render the tool *declarations*: the
    HF reference receives ``tools=`` and the renderer receives the block from
    ``create_conversation_prefix_with_tools``. This keeps tool-declaration
    formatting inside the same byte-parity contract as everything else, rather
    than exempting it.

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
    # Assemble the renderer input through the SAME production path SFT uses
    # (build_tool_prefixed_messages), so the tool-declaration prefix is applied
    # exactly as training does — no harness-local reimplementation to drift from.
    renderer_messages = build_tool_prefixed_messages(
        messages, renderer=renderer, tools=tools
    )

    if add_generation_prompt:
        renderer_input = renderer.build_generation_prompt(
            renderer_messages, role="assistant"
        )
    else:
        renderer_input, _weights = renderer.build_supervised_example(renderer_messages)
    renderer_tokens = [int(t) for t in renderer_input.to_ints()]

    # Declare tools to HF ONLY when the renderer actually declares them (matches
    # production: renderers without a create_conversation_prefix_with_tools
    # override drop tools, so comparing against an HF prompt that DID declare
    # them would be a false divergence). Set tools AFTER hf_kwargs so a stray
    # ``tools`` key in hf_kwargs can't override the scenario's decision.
    hf_kwargs = dict(apply_chat_template_kwargs or {})
    if tools and renderer_declares_tools(renderer):
        hf_kwargs["tools"] = tools
    else:
        hf_kwargs.pop("tools", None)
    # The July 2026 Gemma 4 and MiniMax M3 templates require deserialized
    # arguments. Preserve the production JSON-string representation for other
    # renderer families.
    hf_messages = _hf_messages_with_normalized_tool_args(
        messages,
        require_mapping=renderer_name in {"gemma4", "gemma4_thinking", "minimax_m3"},
    )
    hf_result = tokenizer.apply_chat_template(
        _hf_messages_with_minimax_m3_reasoning_parts(
            hf_messages,
            renderer_name=renderer_name,
        ),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **hf_kwargs,
    )
    if hasattr(hf_result, "input_ids"):
        hf_tokens = [int(t) for t in list(hf_result.input_ids)]  # type: ignore[arg-type]
    else:
        hf_tokens = [int(t) for t in list(hf_result)]  # type: ignore[arg-type]

    first_div = _first_divergence(renderer_tokens, hf_tokens)
    return HFParityResult(
        renderer_tokens=renderer_tokens,
        hf_tokens=hf_tokens,
        match=first_div is None,
        first_divergence_idx=first_div,
        renderer_decoded=_decode_each(tokenizer, renderer_tokens),
        hf_decoded=_decode_each(tokenizer, hf_tokens),
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
