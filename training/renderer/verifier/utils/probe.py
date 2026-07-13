"""Empirical renderer probe.

What it does, in one sentence: render a conversation locally with the
renderer, ask a deployed model to complete the assistant turn, then emit
a token-level audit table that pairs the renderer's claim (chunk source,
training weight) against empirical provenance (was this token actually
emitted by the model, or template-injected).

The audit table is the artifact spec authors read to write
``chunk_rules`` rows. It is *not* a verdict — no pass/fail logic lives
here. Verdict layers (L1, L2) come in follow-up PRs and consume this
same JSON envelope.

The minimal contract is in ``run_probe``. Everything else in this module
is plumbing: chunk attribution, alignment, tokenizer special-token
mapping, and Fireworks API I/O.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
from typing import Any, Iterable, Protocol

import tinker
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import (
    Renderer,
    RenderContext,
    RenderedMessage,
    TrainOnWhat,
)

# Importing the cookbook renderer package registers all cookbook-local
# renderers ("glm5", "gemma4", etc.) under the names exposed by
# ``tinker_cookbook.renderers.get_renderer``.
import training.renderer  # noqa: F401
from training.utils.supervised import normalize_messages

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Provenance vocabulary
_PROV_PROMPT = "prompt_hard_append"
_PROV_NATIVE = "native_generated"
_PROV_TRAILING = "trailing_hard_append"
_PROV_DIVERGED = "tokenization_diverged"


class DispatchError(ValueError):
    """Raised when the probe can't pick a (model, mode) pair from inputs."""


def resolve_dispatch(
    *,
    renderer_name: str,
    model: str | None,
    deployment_id: str | None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    """Pick (model_identifier, dispatch_mode) per the verifier's contract.

    Precedence:

      1. ``model`` and ``deployment_id`` are mutually exclusive — raise.
      2. ``deployment_id`` → resolve via ``DeploymentManager.account_id`` →
         ``accounts/<account>/deployments/<id>``, mode = "deployment".
      3. ``model`` → use as-is, mode = "explicit".
      4. neither → ``DispatchError``. The verifier carries no static
         renderer→model mapping; the caller picks the live Fireworks model
         (the dev server's ``/models`` endpoint and ``SKILL.md`` document
         what's available).

    ``renderer_name`` is accepted for API symmetry with the CLI and the
    /probe handler but doesn't influence resolution any more.
    """
    if model and deployment_id:
        raise DispatchError("`model` and `deployment_id` are mutually exclusive")

    if deployment_id:
        import os as _os

        api_key = api_key or _os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise DispatchError("FIREWORKS_API_KEY not set; cannot resolve deployment_id")
        base_url = base_url or _os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai"
        )
        from fireworks.training.sdk.deployment import DeploymentManager  # noqa: PLC0415

        mgr = DeploymentManager(api_key=api_key, base_url=base_url)
        return f"accounts/{mgr.account_id}/deployments/{deployment_id}", "deployment"

    if model:
        return model, "explicit"

    raise DispatchError(
        "Pass either `model` (e.g. accounts/fireworks/models/glm-5p1) or "
        "`deployment_id`. The verifier no longer ships a renderer→model "
        "default mapping. List available serverless models from the GUI's "
        "model dropdown (populated by /models on the dev server) or with "
        "`Fireworks().models.list(account_id='fireworks', filter=...)`."
    )


class FireworksLikeClient(Protocol):
    """Minimal interface the probe expects from a Fireworks-style client.

    The real client is ``fireworks.Fireworks``; tests pass a stub with the
    same shape so the probe core has no network dependency.
    """

    chat: Any  # exposes .completions.create(**kwargs)


@dataclasses.dataclass
class _ChunkSpan:
    """One contiguous span of tokens attributed to a single render-time source."""

    source: str  # "bos" | "header" | "output" | "stop_overlap" | "generation_suffix"
    msg_idx: int  # -1 for bos / generation_suffix
    role: str | None
    weight: float | None  # filled in from build_supervised_example weights tensor
    tokens: list[int]


def _structural_spans(
    renderer: Renderer,
    messages: list[dict],
    *,
    include_generation_suffix: bool,
    role: str = "assistant",
) -> list[_ChunkSpan]:
    """Walk messages and label each emitted chunk by its structural source.

    This deliberately does NOT compute weights — the only authoritative
    source for ``renderer_claim_weight`` is the renderer's own
    ``build_supervised_example``. Reproducing that logic here would mean
    duplicating any per-renderer customization (e.g. GLM5's split of the
    leading ``<think>`` token to weight 0) and silently disagreeing with
    the renderer the verifier is supposed to be checking against.

    Spans returned here carry ``weight=None`` and are zipped against the
    weights tensor from ``build_supervised_example`` in ``run_probe``.
    """
    spans: list[_ChunkSpan] = []

    bos = list(getattr(renderer, "_bos_tokens", []) or [])
    if bos:
        spans.append(
            _ChunkSpan(source="bos", msg_idx=-1, role=None, weight=None, tokens=bos)
        )

    last_user_idx = max(
        (idx for idx, m in enumerate(messages) if m["role"] == "user"),
        default=-1,
    )
    n = len(messages)

    for idx, msg in enumerate(messages):
        ctx = RenderContext(
            idx=idx,
            is_last=(idx == n - 1),
            prev_message=messages[idx - 1] if idx > 0 else None,
            last_user_index=last_user_idx,
        )
        rendered: RenderedMessage = renderer.render_message(msg, ctx)

        is_last_message = idx == n - 1

        if rendered.header is not None:
            tokens = list(rendered.header.tokens)
            if tokens:
                spans.append(
                    _ChunkSpan(
                        source="header",
                        msg_idx=idx,
                        role=msg["role"],
                        weight=None,
                        tokens=tokens,
                    )
                )

        for chunk in rendered.output:
            tokens = _chunk_tokens(chunk)
            if not tokens:
                continue
            spans.append(
                _ChunkSpan(
                    source="output",
                    msg_idx=idx,
                    role=msg["role"],
                    weight=None,
                    tokens=tokens,
                )
            )

        if is_last_message and rendered.stop_overlap is not None:
            tokens = list(rendered.stop_overlap.tokens)
            if tokens:
                spans.append(
                    _ChunkSpan(
                        source="stop_overlap",
                        msg_idx=idx,
                        role=msg["role"],
                        weight=None,
                        tokens=tokens,
                    )
                )

    if include_generation_suffix:
        suffix_ctx = RenderContext(
            idx=n,
            is_last=True,
            prev_message=messages[-1] if messages else None,
            last_user_index=last_user_idx,
        )
        suffix = list(renderer._get_generation_suffix(role, suffix_ctx))
        if suffix:
            spans.append(
                _ChunkSpan(
                    source="generation_suffix",
                    msg_idx=-1,
                    role=role,
                    weight=None,
                    tokens=suffix,
                )
            )

    return spans


def _chunk_tokens(chunk: tinker.types.ModelInputChunk) -> list[int]:
    """Extract token IDs from a ModelInputChunk, ignoring non-text chunks.

    Image/multimodal chunks have no integer token IDs in the same sense.
    For the probe's scope (text token alignment), we silently skip them
    and emit no audit rows for those positions. The probe is text-only
    in v1; multimodal probing comes with the VL renderer wave.
    """
    tokens = getattr(chunk, "tokens", None)
    if tokens is None:
        return []
    return list(tokens)


def _flatten_spans(spans: Iterable[_ChunkSpan]) -> tuple[list[int], list[_ChunkSpan]]:
    """Return (flat_token_list, per_token_span_ref) parallel arrays.

    The second list points each token at the span it came from, so the
    audit table can read chunk source / weight for any token index.
    """
    flat: list[int] = []
    refs: list[_ChunkSpan] = []
    for span in spans:
        for tok in span.tokens:
            flat.append(tok)
            refs.append(span)
    return flat, refs


def _provenance_per_token(
    *,
    full_len: int,
    prompt_len: int,
    completion_tokens: list[int],
    full_tokens: list[int],
) -> list[str]:
    """Classify each full-render token as prompt / native / trailing / diverged.

    The classification is positional and uses the deployed model's actual
    completion tokens as ground truth for the native span. Anything before
    the prompt boundary is prompt-side hard-append; anything after the
    completion span is template-side trailing hard-append. Mismatches
    inside the expected native span are flagged as ``tokenization_diverged``.
    """
    out: list[str] = []
    completion_end = prompt_len + len(completion_tokens)
    for i in range(full_len):
        if i < prompt_len:
            out.append(_PROV_PROMPT)
        elif i < completion_end:
            j = i - prompt_len
            if j < len(completion_tokens) and full_tokens[i] == completion_tokens[j]:
                out.append(_PROV_NATIVE)
            else:
                out.append(_PROV_DIVERGED)
        else:
            out.append(_PROV_TRAILING)
    return out


def _special_token_map(tokenizer: Any) -> dict[str, str]:
    """Map ``str(token_id) -> decoded`` for tokens flagged as special.

    JSON keys must be strings, hence the str cast on the token id. We
    only emit a row per id we actually know to be special so the artifact
    stays small for large vocabularies.
    """
    out: dict[str, str] = {}
    ids: set[int] = set()

    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tok_id in added:
        ids.add(int(tok_id))

    for attr in (
        "all_special_ids",
        "additional_special_tokens_ids",
    ):
        vals = getattr(tokenizer, attr, None) or []
        for tok_id in vals:
            ids.add(int(tok_id))

    for tok_id in ids:
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:  # noqa: BLE001 - tokenizer-defined behavior varies
            continue
        out[str(tok_id)] = decoded
    return out


def _decode_one(tokenizer: Any, tok_id: int) -> str:
    try:
        return tokenizer.decode([tok_id], skip_special_tokens=False)
    except Exception:  # noqa: BLE001
        return ""


# Fixed API flags the probe always sets so the gateway returns the data the
# alignment step needs. Surfaced verbatim in the artifact (and the React
# viewer) so the human reviewing the probe can see exactly what the
# deployment was asked to do.
PROBE_API_FLAGS: dict[str, Any] = {
    "echo": True,
    "raw_output": True,
    "return_token_ids": True,
}


def _call_completion(
    client: FireworksLikeClient,
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    temperature: float,
    extra_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Hit ``client.chat.completions.create`` with echo + raw_output + return_token_ids.

    Returns a normalised dict with ``prompt_token_ids``,
    ``completion_token_ids``, ``completion_text``, ``stop_reason`` and the
    raw payload. Failures bubble up to the caller; retry logic is
    intentionally not done here so probe runs fail loudly.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **PROBE_API_FLAGS,
    }
    if tools:
        kwargs["tools"] = tools
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    response = client.chat.completions.create(**kwargs)
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)

    prompt_ids = payload.get("prompt_token_ids")
    completion_ids: list[int] = []
    completion_text = ""
    stop_reason = None

    choices = payload.get("choices") or []
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        completion_text = message.get("content") or first.get("text") or ""
        stop_reason = first.get("finish_reason") or first.get("stop_reason")
        raw_output = first.get("raw_output") or {}
        if not prompt_ids:
            prompt_ids = raw_output.get("prompt_token_ids")
        completion_ids = (
            raw_output.get("completion_token_ids")
            or raw_output.get("token_ids")
            or first.get("token_ids")
            or []
        )

    if not prompt_ids:
        raise RuntimeError(
            f"Fireworks response missing prompt_token_ids; payload keys={list(payload)}"
        )

    return {
        "prompt_token_ids": [int(x) for x in prompt_ids],
        "completion_token_ids": [int(x) for x in completion_ids],
        "completion_text": completion_text,
        "stop_reason": stop_reason,
        "raw_payload": payload,
    }


def run_probe(
    *,
    renderer_name: str,
    tokenizer: Any,
    client: FireworksLikeClient,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    extra_completion_kwargs: dict[str, Any] | None = None,
    deployment_id: str | None = None,
    tokenizer_model: str | None = None,
    renderer_config: dict[str, Any] | None = None,
    dispatch_mode: str = "explicit",
) -> dict[str, Any]:
    """Run the empirical probe end to end and return the artifact dict.

    ``dispatch_mode`` ∈ {"serverless", "deployment", "explicit"} records
    how the caller chose ``model`` (default-serverless lookup, --deployment-id
    auto-resolution, or a literal --model override). Recorded in the
    artifact so reviewers can see at a glance what the probe was talking to.

    The artifact is JSON-serialisable. The caller writes it to disk.
    """
    renderer = get_renderer(renderer_name, tokenizer)
    normalized = normalize_messages(messages)

    # 1) Local prompt render
    prompt_input = renderer.build_generation_prompt(normalized, role="assistant")
    renderer_prompt_tokens: list[int] = list(prompt_input.to_ints())

    # 2) Deployed completion (echo=True for prompt parity, raw_output for token IDs)
    api = _call_completion(
        client,
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_kwargs=extra_completion_kwargs,
    )
    api_prompt_tokens = api["prompt_token_ids"]
    completion_tokens = api["completion_token_ids"]
    completion_text = api["completion_text"]
    echo_stripped = False

    # Some Fireworks endpoints honour ``echo=True`` by returning the prompt
    # concatenated in front of the model's actual emission — both in
    # ``completion_token_ids`` and ``message.content``. Detect that and slice
    # so downstream alignment classifies only the model-emitted span as
    # native_generated.
    if (
        completion_tokens
        and len(completion_tokens) >= len(api_prompt_tokens)
        and completion_tokens[: len(api_prompt_tokens)] == api_prompt_tokens
    ):
        completion_tokens = completion_tokens[len(api_prompt_tokens):]
        completion_text = tokenizer.decode(
            completion_tokens, skip_special_tokens=False
        )
        echo_stripped = True

    # 3) Round-trip the completion through the renderer's parser so the
    # assistant turn we hand back to ``build_supervised_example`` is the
    # *structured* message (content / reasoning / tool_calls) — not the
    # model's raw token stream including any trailing stop signal. Without
    # this, Camp A renderers double up the trailing role tag: once because
    # it's embedded in the raw content string, and once because the
    # renderer re-emits it as ``stop_overlap``. parse_response is the
    # renderer's own definition of "what the assistant turn was", so
    # using it here keeps the round-trip non-circular while not papering
    # over real bugs.
    parsed_msg, parse_ok = renderer.parse_response(completion_tokens)
    if parse_ok:
        completion_message: Any = parsed_msg
    else:
        completion_message = {"role": "assistant", "content": completion_text}
    full_messages_raw = list(messages) + [completion_message]
    full_messages = normalize_messages(full_messages_raw)

    # Authoritative source for tokens AND per-token weights: the renderer's
    # own build_supervised_example. This honours per-renderer customization
    # (e.g. GLM5 splits the leading <think> off and assigns it weight 0).
    si_input, weights_tensor = renderer.build_supervised_example(
        full_messages, train_on_what=train_on_what
    )
    full_tokens: list[int] = list(si_input.to_ints())
    weights: list[float] = [float(w) for w in weights_tensor.tolist()]

    # Independently walk render_message to label each token by chunk source.
    # The two walks must produce the same token sequence; if they don't, the
    # renderer's customization changed token identity (not just chunk
    # boundaries) — surface that as a sanity failure rather than silently
    # producing inconsistent attribution.
    struct_spans = _structural_spans(
        renderer,
        full_messages,
        include_generation_suffix=False,
    )
    struct_tokens, span_refs = _flatten_spans(struct_spans)
    structural_token_match = struct_tokens == full_tokens
    if not structural_token_match:
        # Pad span_refs so downstream indexing doesn't crash; the sanity flag
        # tells the human something is off and the audit table will look weird.
        if len(span_refs) < len(full_tokens):
            span_refs = span_refs + [span_refs[-1]] * (len(full_tokens) - len(span_refs))
        else:
            span_refs = span_refs[: len(full_tokens)]

    # 4) Provenance overlay
    prompt_len_for_alignment = len(api_prompt_tokens)
    provenance = _provenance_per_token(
        full_len=len(full_tokens),
        prompt_len=prompt_len_for_alignment,
        completion_tokens=completion_tokens,
        full_tokens=full_tokens,
    )

    # 5) Sanity checks (recorded, not asserted)
    sanity = {
        "renderer_prompt_matches_api_prompt": (
            renderer_prompt_tokens == api_prompt_tokens
        ),
        "renderer_prompt_len": len(renderer_prompt_tokens),
        "api_prompt_len": len(api_prompt_tokens),
        "full_render_prompt_prefix_matches_api": (
            full_tokens[: len(api_prompt_tokens)] == api_prompt_tokens
        ),
        "tokenization_diverged_count": sum(1 for p in provenance if p == _PROV_DIVERGED),
        "echo_prompt_stripped": echo_stripped,
        "completion_token_count": len(completion_tokens),
        "completion_stop_reason": api.get("stop_reason"),
        "parse_response_ok": parse_ok,
        "structural_walk_token_match": structural_token_match,
    }

    # 6) Audit table — chunk_source from structural walk, weight from
    # build_supervised_example weights tensor (authoritative).
    audit: list[dict[str, Any]] = []
    for i, tok_id in enumerate(full_tokens):
        span = span_refs[i]
        decoded = _decode_one(tokenizer, tok_id)
        audit.append(
            {
                "idx": i,
                "token_id": tok_id,
                "decoded": decoded,
                "chunk_source": span.source,
                "msg_idx": span.msg_idx,
                "role": span.role,
                "renderer_claim_weight": weights[i] if i < len(weights) else None,
                "provenance": provenance[i],
            }
        )

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "probe",
        "produced_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "renderer": {
            "name": renderer_name,
            "config": renderer_config or {},
            "train_on_what": str(train_on_what),
        },
        "tokenizer": {
            "model": tokenizer_model or getattr(tokenizer, "name_or_path", None),
            "special_tokens": _special_token_map(tokenizer),
        },
        "deployment": {
            "mode": dispatch_mode,
            "model": model,
            "deployment_id": deployment_id,
            "sampling": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "api_flags": dict(PROBE_API_FLAGS),
            "extra_completion_kwargs": dict(extra_completion_kwargs or {}),
        },
        "input": {
            "messages": messages,
            "tools": tools or [],
        },
        "render": {
            "prompt": {
                "tokens": renderer_prompt_tokens,
                "decoded": [_decode_one(tokenizer, t) for t in renderer_prompt_tokens],
            },
            "full": {
                "tokens": full_tokens,
                "decoded": [_decode_one(tokenizer, t) for t in full_tokens],
            },
        },
        "completion": {
            "text": completion_text,
            "tokens": completion_tokens,
            "stop_reason": api.get("stop_reason"),
        },
        "sanity": sanity,
        "audit_table": audit,
    }
    return artifact


def write_artifact(artifact: dict[str, Any], path: str) -> None:
    """Write the probe artifact as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
        f.write("\n")
