"""Shared eval helpers for the bucket notebooks (DPO / distillation / multi-LoRA / RuleArena).

Thin wrappers around the Fireworks OpenAI-compatible endpoint plus a couple of
scorers, factored out so every notebook grades before/after fine-tuning the same
way. Mirrors the inference + 503-cold-start-retry pattern from
`eval_before_after.ipynb`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def fw_client(api_key: str, base_url: str = FIREWORKS_BASE_URL):
    """Return an OpenAI SDK client pointed at Fireworks inference."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 8,
    backoff: float = 15.0,
    **kwargs,
) -> Tuple[str, float, Optional[int], Optional[int]]:
    """One chat completion. Returns (text, latency_s, in_tokens, out_tokens).

    Retries 503 cold-starts (min_replica_count=0 deployments return 503 while a
    replica spins up). Latency is measured only for the successful call so cold
    waits are not counted.
    """
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            latency = time.time() - t0
            u = r.usage
            return (
                r.choices[0].message.content or "",
                latency,
                getattr(u, "prompt_tokens", None),
                getattr(u, "completion_tokens", None),
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            is_cold = "503" in msg or "scaling up" in msg or "scaled to zero" in msg
            if is_cold and attempt < max_retries:
                print(f"      cold start (503), waiting {backoff:.0f}s [retry {attempt + 1}/{max_retries}] ...")
                time.sleep(backoff)
                continue
            raise


def strip_think(text: str) -> str:
    """Remove Qwen-style <think>...</think> reasoning blocks."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.DOTALL).strip()


def exact_match(pred: str, gold: str, normalizer: Optional[Callable[[str], str]] = None) -> bool:
    """Normalized exact match. Default normalizer lowercases + collapses whitespace."""
    norm = normalizer or (lambda s: " ".join((s or "").lower().split()))
    return norm(strip_think(pred)) == norm(gold)


# --------------------------------------------------------------------------
# Pairwise LLM judge (for preference / style tasks like DPO)
# --------------------------------------------------------------------------

_PAIRWISE_PROMPT = """You are comparing two AI assistant responses to the same conversation.

CONVERSATION (everything before the final answer):
{context}

RESPONSE A:
{a}

RESPONSE B:
{b}

Which response is better (more helpful, correct, and appropriately styled)? Reply with ONLY a JSON object:
{{"winner": "A" | "B" | "tie", "reasoning": "<one sentence>"}}"""

_PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["winner", "reasoning"],
}


def _ctx_to_text(messages: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(f"{m.get('role', '')}: {content}")
    return "\n".join(parts)[-max_chars:]


def pairwise_judge(
    client,
    judge_model: str,
    context_messages: List[Dict[str, Any]],
    a: str,
    b: str,
    swap_guard: bool = True,
) -> str:
    """Return "A", "B", or "tie". If swap_guard, run both orderings and only
    keep a decisive verdict when the two runs agree after un-swapping (removes
    position bias); otherwise "tie"."""

    def _one(resp_a: str, resp_b: str) -> str:
        prompt = _PAIRWISE_PROMPT.format(context=_ctx_to_text(context_messages), a=resp_a[:3000], b=resp_b[:3000])
        try:
            r = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=None,
                temperature=0.0,
                response_format={"type": "json_object", "schema": _PAIRWISE_SCHEMA},
            )
            obj = json.loads(r.choices[0].message.content or "{}")
            return str(obj.get("winner", "tie")).upper() if obj.get("winner") != "tie" else "tie"
        except Exception:  # noqa: BLE001 - judge must never crash the eval
            return "tie"

    v1 = _one(a, b)
    if not swap_guard:
        return v1
    v2_swapped = _one(b, a)  # A<-b, B<-a; so "A" here means original b won
    v2 = {"A": "B", "B": "A", "tie": "tie"}[v2_swapped]
    return v1 if v1 == v2 else "tie"
