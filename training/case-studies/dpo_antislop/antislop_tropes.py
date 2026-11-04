"""Mechanical counts of common AI writing habits ("tropes").

This is deliberately a short, readable list of regular expressions, not a
classifier. Each pattern is something a reader can see in the text and argue
about. Counts are reported per text and normalized per 100 words, so a long
LinkedIn post and a short tweet compare on the same footing.

The list is a starting point. Add to it, but keep every entry explainable in
one line, and keep it OUT of the training reward: the experiment is whether
optimizing a detector reduces these on its own.
"""
from __future__ import annotations

import re
from typing import Any

TROPES: dict[str, str] = {
    # punctuation habits
    "em_dash": r"—|(?<=\w)\s--\s(?=\w)",
    # "not X, but Y" / "it's not about X; it's about Y" framing
    "not_x_but_y": r"\b(?:not|isn't|isn’t|aren't|aren’t|wasn't|wasn’t)\b[^.!?\n]{1,60}?\b(?:but|rather)\b",
    # colon-then-reveal headline style: "The result: ..." / "The best part? ..."
    "colon_reveal": r"\b(?:the (?:result|catch|best part|kicker|bottom line|takeaway|difference))\s*[:?]",
    # stock promotional openers and phrases
    "stock_phrase": r"\b(?:game[- ]changer|unlock(?:s|ing)? the (?:power|potential)|in today'?s (?:fast-paced|rapidly evolving)|"
                    r"dive (?:in|into|deep)|let'?s (?:dive|unpack|break (?:it|this) down)|seamless(?:ly)?|cutting[- ]edge|"
                    r"revolutioni[sz](?:e|ing)|supercharge|elevate your|next[- ]level|buckle up|say goodbye to|"
                    r"the future (?:of|is) (?:here|now)|a new era|look no further)\b",
    # triplet with escalating adjectives: "faster, cheaper, better"
    "rule_of_three_adjectives": r"\b\w+er,\s\w+er,\s(?:and\s)?\w+er\b",
    # rhetorical question hook opening a sentence
    "rhetorical_hook": r"(?:^|[.!?]\s+)(?:Ever|Tired of|What if|Imagine|Want to|Ready to)\b[^.!?\n]{0,80}\?",
    # emoji
    "emoji": r"[\U0001F300-\U0001FAFF☀-➿]",
    # hashtag pile (3 or more hashtags)
    "hashtag_pile": r"(?:#\w+\s*){3,}",
    # "Here's the thing" / "Here's why" / "Here's what"
    "heres_the": r"\bhere'?s (?:the thing|why|what|how)\b",
}

_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in TROPES.items()}


def count_tropes(text: str) -> dict[str, int]:
    """Occurrences of each trope in one text."""
    return {k: len(p.findall(text or "")) for k, p in _COMPILED.items()}


def trope_report(text: str) -> dict[str, Any]:
    """Counts, total, per-100-words rate, and whether any trope is present."""
    counts = count_tropes(text)
    words = max(1, len((text or "").split()))
    total = sum(counts.values())
    return {"counts": counts, "total": total, "per_100_words": 100.0 * total / words, "any": total > 0, "words": words}


def summarize(texts: list[str]) -> dict[str, Any]:
    """Aggregate over many texts: mean rate, share with any trope, per-trope share."""
    reps = [trope_report(t) for t in texts]
    n = max(1, len(reps))
    return {
        "n": len(reps),
        "mean_per_100_words": sum(r["per_100_words"] for r in reps) / n,
        "share_with_any": sum(r["any"] for r in reps) / n,
        "share_by_trope": {k: sum(r["counts"][k] > 0 for r in reps) / n for k in TROPES},
    }


if __name__ == "__main__":
    import json
    import sys

    text = sys.stdin.read()
    print(json.dumps(trope_report(text), indent=2))
