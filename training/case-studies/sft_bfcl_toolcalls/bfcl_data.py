#!/usr/bin/env python3
"""Load the BFCL V4 non-live slice, pinned to the leaderboard snapshot.

Why not the HuggingFace dataset: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
still ships only `BFCL_v3_*` files and its viewer is disabled. The canonical V4
data lives in the `bfcl-eval` package / the gorilla repo. We read it from raw
GitHub at a pinned commit so a notebook re-run a month from now scores against the
same rows the published board used.

Note the files are JSON *Lines* despite the `.json` extension.

The seven non-live categories are the whole scoring surface here. `multi_turn`
needs a stateful Python sandbox, and the `agentic` categories (memory, web_search)
need a paid SerpAPI key plus a FAISS index -- neither belongs in a teaching
notebook, and neither is AST-scored.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

# The commit backing the 2025-12-16 leaderboard snapshot (bfcl-eval==2025.12.17).
# Pin it so our numbers stay comparable to the published board.
BFCL_COMMIT = "f7cf735"
BFCL_RAW = (
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_COMMIT}"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# Row counts verified against the pinned commit. Asserting them means a silent
# upstream change fails loudly instead of quietly shifting the denominator.
NON_LIVE_AST = {
    "simple_python": 400,
    "simple_java": 100,
    "simple_javascript": 50,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
}
IRRELEVANCE = {"irrelevance": 240}
NON_LIVE = {**NON_LIVE_AST, **IRRELEVANCE}

# `simple_java` / `simple_javascript` expect values to arrive as strings and are
# scored by language-specific converters. Every open tool-calling training set is
# Python-only, which is why every model on the board scores 20-30 points worse on
# them. Kept in the headline number (the board includes them) but broken out in
# the notebook's per-category table so the gap is visible rather than averaged away.
LANGUAGE = {"simple_java": "java", "simple_javascript": "javascript"}


def _fetch(url: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / url.rsplit("/", 1)[-1]
    if "possible_answer" in url:
        cached = cache_dir / ("answer_" + url.rsplit("/", 1)[-1])
    if not cached.exists():
        with urllib.request.urlopen(url) as resp:
            cached.write_bytes(resp.read())
    return cached.read_text(encoding="utf-8")


def load_category(category: str, cache_dir: str | Path = "./bfcl_data") -> list[dict]:
    """Prompt rows for one category, with the expected row count asserted."""
    text = _fetch(f"{BFCL_RAW}/BFCL_v4_{category}.json", Path(cache_dir))
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    expected = NON_LIVE.get(category)
    if expected is not None and len(rows) != expected:
        raise AssertionError(
            f"{category}: expected {expected} rows at commit {BFCL_COMMIT}, got {len(rows)}. "
            "Upstream data changed -- re-pin BFCL_COMMIT and re-verify the counts."
        )
    return rows


def load_answers(category: str, cache_dir: str | Path = "./bfcl_data") -> dict[str, list]:
    """Ground truth keyed by row id.

    Returns `{}` for `irrelevance`: BFCL ships no possible_answer file for the
    hallucination categories by design, because they are scored structurally --
    the only question is whether *any* parseable call came back.
    """
    if category in IRRELEVANCE:
        return {}
    text = _fetch(f"{BFCL_RAW}/possible_answer/BFCL_v4_{category}.json", Path(cache_dir))
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {r["id"]: r["ground_truth"] for r in rows}


def load_non_live(cache_dir: str | Path = "./bfcl_data") -> dict[str, list[dict]]:
    """All seven non-live categories: 1,390 rows total."""
    return {cat: load_category(cat, cache_dir) for cat in NON_LIVE}


# BFCL's own type vocabulary, which is Python-flavoured rather than JSON Schema.
# `float` matters: JSON Schema has no `float`, and an unconverted one makes the
# endpoint reject the whole tools payload.
_BFCL_TO_JSON_TYPE = {
    "dict": "object", "tuple": "array", "float": "number", "any": "string",
    "String": "string", "long": "integer", "Long": "integer",
    "double": "number", "Double": "number",
    "Array": "array", "ArrayList": "array",
    "HashMap": "object", "Map": "object",
    "Boolean": "boolean", "Integer": "integer", "Float": "number",
    # `char` shows up in some simple_java ground truth. It is not a JSON Schema
    # type, and the dedicated-deployment path 400s the whole request on it
    # ("Error validating JSON Schema: {'type': 'char'} is not valid...") -- the
    # serverless path happens to accept it, which is why only tuned sweeps failed.
    # "string" is what the checker compares against anyway.
    "char": "string", "Char": "string",
}


def _normalize_types(node: Any) -> Any:
    """Recursively rewrite BFCL type names to JSON Schema ones.

    This has to recurse: 16 of the 1,390 non-live rows nest a `dict` type below
    the top level (inside `items`, or inside a nested object's `properties`), and
    a single leftover one is enough for the endpoint to 400 the request.
    """
    if isinstance(node, dict):
        out = {k: _normalize_types(v) for k, v in node.items()}
        raw = out.get("type")
        if isinstance(raw, str) and raw in _BFCL_TO_JSON_TYPE:
            out["type"] = _BFCL_TO_JSON_TYPE[raw]
        return out
    if isinstance(node, list):
        return [_normalize_types(v) for v in node]
    return node


def to_openai_tools(functions: list[dict]) -> list[dict]:
    """BFCL `function` entries -> an OpenAI `tools` array.

    One real gotcha: BFCL writes `"type": "dict"` for the parameter object, which
    is not valid JSON Schema and which the Fireworks endpoint will reject. The
    official harness fixes this in `bfcl_eval.model_handler.utils._cast_to_openai_type`;
    we do the same conversion here so the notebook has no hard dependency on the
    harness just to send a request.
    """
    fixed = []
    for fn in functions:
        params = _normalize_types(fn.get("parameters", {}))
        fixed.append(
            {
                "type": "function",
                "function": {
                    # Fireworks/OpenAI tool names must match ^[a-zA-Z0-9_-]{1,64}$.
                    # BFCL's own `underscore_to_dot` handling assumes this rewrite.
                    "name": fn["name"].replace(".", "_"),
                    "description": fn.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return fixed


def user_prompt(row: dict) -> str:
    """The single user turn. `question` is List[List[message]]; non-live has one turn."""
    return row["question"][0][0]["content"]


def load_eval_queries(cache_dir: str | Path = "./bfcl_data") -> list[str]:
    """Every non-live query, for the leakage filter in build_bfcl_sft.py."""
    return [user_prompt(r) for rows in load_non_live(cache_dir).values() for r in rows]


def eval_tool_names(cache_dir: str | Path = "./bfcl_data") -> set[str]:
    """Every function name BFCL non-live exposes, for the overlap report."""
    return {
        fn["name"].replace(".", "_")
        for rows in load_non_live(cache_dir).values()
        for r in rows
        for fn in r.get("function", [])
    }


if __name__ == "__main__":
    data = load_non_live()
    for cat, rows in data.items():
        print(f"{cat:24s} {len(rows):>5d}")
    print(f"{'TOTAL':24s} {sum(len(r) for r in data.values()):>5d}")
