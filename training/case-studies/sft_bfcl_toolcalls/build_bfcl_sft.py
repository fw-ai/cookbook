#!/usr/bin/env python3
"""Build a BFCL-shaped SFT set from xLAM, and hold it out from the benchmark.

Two sources, one schema. Both `Salesforce/xlam-function-calling-60k` (positives)
and `MadeAgents/xlam-irrelevance-7.5k` (negatives) have the same three columns --
`query`, `tools`, `answers` -- each a *stringified* JSON blob. Negatives are just
rows whose `answers` parses to `[]`. So one parser covers both, and the only real
work is schema translation plus leakage filtering.

Schema translation. xLAM describes parameters as a flat name -> spec map with
Python type names (`int`, `str`, `list`) and optionality expressed as either a
`required` flag or the presence of a `default`. Fireworks -- like OpenAI -- wants
JSON Schema. `to_openai_tools()` does that conversion.

Leakage. BFCL ships no train split, and xLAM postdates the decontamination pass
BFCL ran when it built its Live categories (that pass filtered NexusRaven,
Firefunctions, Anyscale and Glaive -- not xLAM, which did not exist yet). The
non-live categories were never filtered at all. So we filter here, and the
notebook prints the counts rather than hiding them.

Usage:
    python build_bfcl_sft.py --out bfcl_sft_train.jsonl --n-positive 12000 --n-negative 1800
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from bfcl_data import load_eval_queries

# xLAM writes Python type names; JSON Schema wants its own. Anything unmapped
# falls through to "string", which is what the BFCL AST checker coerces to for
# its own "any" type as well.
_TYPE_MAP = {
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "double": "number",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "dict": "object",
    "object": "object",
    "any": "string",
    "char": "string",
}

# "List[str]", "list[int]", "Dict[str, Any]" -- take the outer container and,
# for sequences, the inner type so we can emit a usable `items`.
_GENERIC_RE = re.compile(r"^\s*(\w+)\s*\[\s*([^,\]]+)", re.IGNORECASE)

# Refusal targets for the negatives, sampled per row.
#
# This is a pool rather than one fixed string, and that matters more than it
# looks. A single constant repeated across every negative becomes the most
# frequent target in the dataset by a wide margin, and the model learns "emit
# this string" as a high-prior default rather than learning *when* to abstain.
# The failure shows up as refusals on rows that clearly do warrant a call --
# worst on the `multiple` category, where picking one tool from several is the
# hardest discrimination -- while accuracy on genuine irrelevance goes *down*,
# because the refusal has come untethered from the relevance signal.
#
# Keep the pool varied and the negative share small. Selection is keyed off the
# query hash so a given row always renders the same way across runs.
REFUSALS = (
    "I don't have a tool that can do that.",
    "None of the available functions can answer that.",
    "That's outside what these tools can do.",
    "I can't help with that using the tools I have.",
    "There's no function here that covers that request.",
    "None of these tools apply to that question.",
    "I don't have access to a function for that.",
    "That request doesn't match any of the available tools.",
)


# --------------------------------------------------------------------------
# schema translation
# --------------------------------------------------------------------------
def _json_type(raw: str) -> tuple[str, str | None]:
    """Map an xLAM type string to (json_schema_type, item_type_or_None)."""
    raw = (raw or "string").strip()
    generic = _GENERIC_RE.match(raw)
    if generic:
        outer, inner = generic.group(1), generic.group(2)
        mapped = _TYPE_MAP.get(outer.lower(), "string")
        if mapped == "array":
            return mapped, _TYPE_MAP.get(inner.strip().lower(), "string")
        return mapped, None
    return _TYPE_MAP.get(raw.lower(), "string"), None


def to_openai_tools(xlam_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """xLAM tool specs -> the OpenAI `tools` array.

    Returns the *wrapped* form (`{"type": "function", "function": {...}}`),
    which is what both the Fireworks chat endpoint and the cookbook renderer's
    `create_conversation_prefix_with_tools` expect.
    """
    out = []
    for tool in xlam_tools:
        properties, required = {}, []
        for pname, spec in (tool.get("parameters") or {}).items():
            if not isinstance(spec, dict):
                continue
            jtype, item_type = _json_type(spec.get("type", "string"))
            prop: dict[str, Any] = {"type": jtype}
            if spec.get("description"):
                prop["description"] = spec["description"]
            # Only assert `items` when xLAM actually told us the element type.
            # A bare `"type": "list"` carries no inner type, and guessing
            # `string` there is worse than omitting it -- JSON Schema allows an
            # untyped array, whereas a wrong `items` teaches the model to quote
            # numbers and then fails BFCL's type check at eval time.
            if jtype == "array" and item_type:
                prop["items"] = {"type": item_type}
            properties[pname] = prop
            # xLAM marks optionality two ways depending on vintage: an explicit
            # `required` flag, or the presence of a `default`. Honour both.
            if spec.get("required") is True or (
                "required" not in spec and "default" not in spec
            ):
                required.append(pname)
        out.append(
            {
                "type": "function",
                "function": {
                    # BFCL rewrites `.` to `_` at eval time for providers whose
                    # tool-name regex forbids dots (Fireworks is one). Match that
                    # here so train and eval agree on the name.
                    "name": tool["name"].replace(".", "_"),
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return out


def to_chat_row(query: str, tools: list, answers: list) -> dict[str, Any]:
    """One xLAM row -> one `{messages, tools}` SFT row.

    `sft_loop` reads exactly these two keys off each JSONL line
    (see `_render_one_worker` in `training/recipes/sft_loop.py`), so this is
    the whole contract.
    """
    if answers:
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": a["name"].replace(".", "_"),
                        "arguments": json.dumps(a.get("arguments", {})),
                    },
                }
                for i, a in enumerate(answers)
            ],
        }
    else:
        # Keyed off the query so the choice is stable across runs without
        # threading an rng through a pure function.
        pick = zlib.crc32(query.encode("utf-8")) % len(REFUSALS)
        assistant = {"role": "assistant", "content": REFUSALS[pick]}

    return {
        "messages": [{"role": "user", "content": query}, assistant],
        "tools": to_openai_tools(tools),
    }


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def _shingles(text: str, n: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i : i + n] for i in range(max(len(t) - n + 1, 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LeakageFilter:
    """Drop training rows that are too close to a BFCL eval query.

    Two passes, both dependency-free:

    1. Normalized exact match -- an O(1) set lookup, catches verbatim reuse.
    2. Character 5-shingle Jaccard above `threshold`, against eval queries that
       share a rare word. BFCL used ROUGE-L > 0.8 for its own dedupe; shingle
       Jaccard is a cheap stand-in that needs no extra package. The rare-word
       blocking key keeps this near-linear instead of 15k x 1.4k.

    Function-name overlap is deliberately *reported and not filtered*: xLAM and
    BFCL both draw on the RapidAPI namespace, so dropping every shared API would
    gut the training set and would not match any published baseline either.
    """

    def __init__(self, eval_queries: Iterable[str], threshold: float = 0.8):
        self.threshold = threshold
        self.exact: set[str] = set()
        self.by_token: dict[str, list[set[str]]] = {}
        # Words appearing in many eval queries make useless blocking keys, so
        # index each query under its *rarest* words only.
        queries = [q for q in eval_queries if q]
        df = Counter(w for q in queries for w in set(normalize(q).split()))
        for q in queries:
            self.exact.add(normalize(q))
            sh = _shingles(q)
            words = sorted(set(normalize(q).split()), key=lambda w: df[w])[:3]
            for w in words:
                self.by_token.setdefault(w, []).append(sh)

    def is_leaked(self, query: str) -> bool:
        norm = normalize(query)
        if norm in self.exact:
            return True
        sh = _shingles(query)
        seen: list[set[str]] = []
        for w in set(norm.split()):
            seen.extend(self.by_token.get(w, ()))
        return any(jaccard(sh, other) > self.threshold for other in seen)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_xlam(repo_id: str, token: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield parsed `{query, tools, answers}` rows from an xLAM-schema dataset.

    All three columns are stringified JSON on the Hub, so every row costs three
    `json.loads`. Rows that fail to parse are skipped rather than killing the run.
    """
    # lazy: `datasets` is a heavy optional dep, only needed when actually building.
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train", token=token)
    for row in ds:
        try:
            yield {
                "query": row["query"],
                "tools": json.loads(row["tools"]),
                "answers": json.loads(row["answers"]),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            continue


def build(
    eval_queries: list[str],
    n_positive: int,
    n_negative: int,
    *,
    hf_token: str | None = None,
    seed: int = 0,
    positives_repo: str = "Salesforce/xlam-function-calling-60k",
    negatives_repo: str = "MadeAgents/xlam-irrelevance-7.5k",
) -> tuple[list[dict], dict[str, int]]:
    """Return (rows, stats). Stats are printed by the notebook, not swallowed."""
    rng = random.Random(seed)
    leak = LeakageFilter(eval_queries)
    stats = Counter()

    def take(repo: str, want: int, keep_positive: bool, token: str | None) -> list[dict]:
        kept: list[dict] = []
        for row in load_xlam(repo, token=token):
            stats["seen"] += 1
            has_answer = bool(row["answers"])
            if has_answer != keep_positive:
                continue
            if not row["tools"]:
                stats["dropped_no_tools"] += 1
                continue
            if leak.is_leaked(row["query"]):
                stats["dropped_leaked"] += 1
                continue
            kept.append(row)
            if len(kept) >= want * 3:  # oversample, then shuffle-select
                break
        rng.shuffle(kept)
        return kept[:want]

    positives = take(positives_repo, n_positive, True, hf_token)
    negatives = take(negatives_repo, n_negative, False, None)

    stats["positives"] = len(positives)
    stats["negatives"] = len(negatives)
    stats["n_parallel"] = sum(1 for r in positives if len(r["answers"]) > 1)
    stats["n_multi_tool_choice"] = sum(1 for r in positives if len(r["tools"]) > 1)

    rows = [to_chat_row(r["query"], r["tools"], r["answers"]) for r in positives + negatives]
    rng.shuffle(rows)
    return rows, dict(stats)


def tool_name_overlap(rows: list[dict], eval_tool_names: set[str]) -> tuple[int, int]:
    """(shared, total) function names -- reported as a caveat, not filtered."""
    train_names = {
        t["function"]["name"] for r in rows for t in r.get("tools", [])
    }
    return len(train_names & eval_tool_names), len(train_names)


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="bfcl_sft_train.jsonl")
    p.add_argument("--n-positive", type=int, default=12000)
    p.add_argument("--n-negative", type=int, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hf-token", default=None, help="needed for the gated xLAM repo")
    args = p.parse_args()

    rows, stats = build(
        load_eval_queries(), args.n_positive, args.n_negative,
        hf_token=args.hf_token, seed=args.seed,
    )
    write_jsonl(rows, args.out)
    print(json.dumps(stats, indent=2), file=sys.stderr)
    print(f"wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
