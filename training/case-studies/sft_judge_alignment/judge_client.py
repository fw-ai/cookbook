"""Call a judge and recover both its scores and its *confidence* in them.

Shared by the annotator panel, the prompted baselines, and the fine-tuned judge,
so all three are measured through exactly the same code path.

The calibration half of this case study needs a probability distribution over
1-5 per attribute, not just the argmax. Fireworks returns `top_logprobs` even
when `response_format` constrains generation, so we can read the distribution
straight off the token that produced each score. The wrinkle is locating that
token: reasoning models emit hundreds of tokens before the JSON, and attribute
names are split across token boundaries ('correct' + 'ness'). So we reconstruct
the full text from the token stream, keep a char-offset -> token-index map, and
regex for `"attr": <digit>` in the final JSON object.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

SCALE = [1, 2, 3, 4, 5]

# The OpenAI SDK defaults to a 600s read timeout and 2 retries, so a single stalled
# judge call can hold a batch for half an hour. Bound it: a judgment that slow is
# better recorded as a failure (it gets dropped from the comparison) than waited on.
JUDGE_TIMEOUT_S = 90
JUDGE_MAX_RETRIES = 1


def rubric_schema(attributes: Sequence[str]) -> dict[str, Any]:
    """Structured-output schema pinning the reply to one integer 1-5 per attribute."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rubric_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {a: {"type": "integer", "minimum": 1, "maximum": 5} for a in attributes},
                "required": list(attributes),
                "additionalProperties": False,
            },
        },
    }


@dataclass
class JudgeOutput:
    scores: dict[str, int]
    # Per attribute, a length-5 probability vector over scores 1..5.
    dists: dict[str, list[float]] = field(default_factory=dict)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""
    ok: bool = True
    # True when the distribution is a one-hot fallback rather than real logprobs.
    dist_is_fallback: bool = True
    # True when the reply contained more than one copy of the answer -- a fine-tuned
    # model that has not learned to stop. Parsing survives it; latency and cost do not.
    repeated: bool = False


def _token_offsets(tokens: Sequence[Any]) -> tuple[str, list[int]]:
    """Reconstruct the generated text and the start offset of each token in it."""
    text_parts, offsets, pos = [], [], 0
    for t in tokens:
        offsets.append(pos)
        text_parts.append(t.token)
        pos += len(t.token)
    return "".join(text_parts), offsets


def _dist_from_top_logprobs(tok: Any) -> list[float] | None:
    """Renormalize a token's top-k alternatives over the digits 1-5."""
    alts = getattr(tok, "top_logprobs", None) or []
    probs = {}
    for alt in alts:
        s = alt.token.strip()
        if s in ("1", "2", "3", "4", "5"):
            # A digit can appear more than once in top-k with different leading
            # whitespace; keep the largest mass for each value.
            probs[int(s)] = max(probs.get(int(s), 0.0), math.exp(alt.logprob))
    if not probs:
        return None
    total = sum(probs.values())
    if total <= 0:
        return None
    return [probs.get(s, 0.0) / total for s in SCALE]


def extract_dists(choice: Any, attributes: Sequence[str]) -> dict[str, list[float]]:
    """Pull a 1-5 distribution per attribute off the response's token logprobs."""
    lp = getattr(choice, "logprobs", None)
    tokens = getattr(lp, "content", None) if lp else None
    if not tokens:
        return {}
    text, offsets = _token_offsets(tokens)
    # Restrict to the first *complete* JSON object: that skips digits in a reasoning
    # preamble, and it is also the object `parse_scores` read, so the distribution and
    # the score can never come from different copies of a repeated answer.
    found = find_object(text, attributes)
    if found is None:
        return {}
    start, end, _ = found
    span = text[start:end]
    out: dict[str, list[float]] = {}
    for attr in attributes:
        # The JSON is emitted verbatim, but whitespace and quoting vary by model.
        matches = list(re.finditer(rf'"{re.escape(attr)}"\s*:\s*(\d)', span))
        if not matches:
            continue
        digit_off = start + matches[0].start(1)
        # offsets is sorted; find the token containing this character.
        idx = None
        for i, off in enumerate(offsets):
            if off <= digit_off < off + len(tokens[i].token):
                idx = i
                break
        if idx is None:
            continue
        dist = _dist_from_top_logprobs(tokens[idx])
        if dist:
            out[attr] = dist
    return out


def find_object(text: str, attributes: Sequence[str]) -> tuple[int, int, dict[str, Any]] | None:
    """Locate the FIRST balanced JSON object containing every attribute.

    "First" rather than "last", which is a correctness issue, not a style one. A
    fine-tuned judge that has not learned to stop will emit the same object over and
    over until it hits `max_tokens`, leaving a truncated fragment at the end -- so
    anchoring on the last `{` finds unparseable garbage and throws away a perfectly
    good answer. Scanning forward for the first *complete* object also steps over
    braces in a reasoning preamble, since those will not parse as a dict carrying
    all the attributes.

    Also tolerates a missing final `}`. Using `stop=["}"]` to make a run-on model shut
    up is the right fix, but the API strips the stop string from the reply, so the JSON
    comes back one brace short. Appending it is safe: the brace goes on the end, so every
    character offset before it -- including the score digits -- is unchanged.
    """
    if not text:
        return None
    for candidate in (text, text + "}"):
        for i, ch in enumerate(candidate):
            if ch != "{":
                continue
            depth = 0
            for j in range(i, len(candidate)):
                if candidate[j] == "{":
                    depth += 1
                elif candidate[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(candidate[i:j + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict) and all(a in obj for a in attributes):
                            return i, j + 1, obj
                        break
    return None


def parse_scores(text: str, attributes: Sequence[str]) -> dict[str, int] | None:
    """Pull {attr: int} out of the reply; clamp to 1-5. Safety net under the schema."""
    found = find_object(text, attributes)
    if found is None:
        return None
    raw = found[2]
    out = {}
    for a in attributes:
        try:
            out[a] = min(5, max(1, int(round(float(raw[a])))))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def judge_once(
    client: Any,
    model: str,
    system: str,
    user: str,
    attributes: Sequence[str],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    want_logprobs: bool = True,
    timeout: float = JUDGE_TIMEOUT_S,
    stop: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JudgeOutput:
    """One judge call. Returns scores, per-attribute distributions, latency, tokens.

    `stop=["}"]` ends generation at the close of the JSON object. Use it for a
    fine-tuned judge that runs on past its answer -- it cuts output from the token
    cap to ~25 tokens, which is most of the latency and cost. Do **not** use it with
    a reasoning model: a brace anywhere in the thinking preamble would end the reply
    before the answer exists.
    """
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=rubric_schema(attributes),
        timeout=timeout,
        **({"stop": stop} if stop else {}),
        **(extra or {}),
    )
    if want_logprobs:
        kwargs.update(logprobs=True, top_logprobs=5)

    t0 = time.perf_counter()
    try:
        resp = client.with_options(max_retries=JUDGE_MAX_RETRIES).chat.completions.create(**kwargs)
    except Exception as exc:
        return JudgeOutput(scores={}, latency_s=time.perf_counter() - t0, raw=f"ERROR: {exc}", ok=False)
    latency = time.perf_counter() - t0

    choice = resp.choices[0]
    text = choice.message.content or ""
    scores = parse_scores(text, attributes)
    if scores is None:
        return JudgeOutput(scores={}, latency_s=latency, raw=text, ok=False)

    dists = extract_dists(choice, attributes) if want_logprobs else {}
    usage = getattr(resp, "usage", None)
    repeated = text.count(f'"{attributes[0]}"') > 1 if attributes else False
    out = JudgeOutput(
        repeated=repeated,
        scores=scores,
        dists={a: dists.get(a, one_hot(scores[a])) for a in attributes},
        latency_s=latency,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        raw=text,
        ok=True,
        dist_is_fallback=not dists,
    )
    return out


def one_hot(score: int) -> list[float]:
    return [1.0 if s == score else 0.0 for s in SCALE]


def render_prompt(trace_text: str, task: str) -> str:
    return f"Task given to the agent:\n{task}\n\nAgent trace:\n{trace_text}"


def run_judge(
    records: Sequence[dict[str, Any]],
    client: Any,
    model: str,
    system: str,
    attributes: Sequence[str],
    *,
    temperature: float = 0.0,
    concurrency: int = 8,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], list[JudgeOutput]]:
    """Score every record with one judge configuration.

    Returns `(kept_records, outputs)` aligned by index, with unparseable calls
    dropped from *both* -- so a judge is never credited with agreement on a row
    it failed to answer, and the two lists stay usable as parallel arrays.

    Note on `eval-protocol`: the sibling case studies route rollouts through
    `common/ep_eval.single_turn_eval`, but that helper returns only the message
    text. Three of the four numbers this notebook reports (confidence
    distributions, latency, token cost) live on the raw response, so the judge
    calls go direct to the Fireworks OpenAI-compatible endpoint instead.
    """
    from concurrent.futures import ThreadPoolExecutor

    done = {"n": 0}

    def work(rec):
        from label_panel import build_user_prompt

        out = judge_once(
            client, model, system, build_user_prompt(rec), attributes,
            temperature=temperature, max_tokens=max_tokens, stop=stop,
        )
        done["n"] += 1
        if progress and done["n"] % 25 == 0:
            print(f"  judged {done['n']}/{len(records)}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outs = list(pool.map(work, records))

    kept = [(r, o) for r, o in zip(records, outs) if o.ok]
    if not kept:
        return [], []
    recs, outputs = zip(*kept)
    return list(recs), list(outputs)


def parse_rate(outputs: Sequence[JudgeOutput], attempted: int) -> float:
    return len(outputs) / attempted if attempted else 0.0
