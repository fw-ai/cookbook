#!/usr/bin/env python3
"""Seed requests for the lipogram rewrite pipeline: real user questions from public sets.

Sources (Hugging Face): databricks/databricks-dolly-15k (instruction + optional context),
OpenAssistant/oasst1 (English root prompter messages), sentence-transformers/eli5 (questions).
Every seed is deduplicated against all of Rob's lipogram prompts (rl_lipogram/data/*.jsonl:
held-out test, train and the generated pool) by normalized text and word 8-gram overlap, and
against the other seeds. Each seed gets sampled attributes: request type, answer length band and
reader persona, which the draft stage uses to restyle the request.

    python seeds.py --n 3000 --out data/seeds.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
ROB = HERE.parent / "rl_lipogram" / "data"
TYPES = ["how-to", "explanation", "list", "summary", "advice", "factual", "casual chat"]
LENGTHS = {"short": "one or two sentences", "medium": "one paragraph", "long": "three to five short paragraphs"}
LENGTH_WEIGHTS = {"short": 0.3, "medium": 0.45, "long": 0.25}
PERSONAS = ["a curious ten-year-old", "a busy manager who wants the gist", "a university student", "a retired teacher",
            "a software engineer", "a nurse on a night shift", "a small-business owner", "a hobby gardener",
            "a first-time parent", "a high-school science teacher", "a traveller on a budget", "a novelist doing research",
            "a non-native English speaker", "a sports coach", "a home cook", "a journalist on deadline"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def grams(s: str, n: int = 8) -> set[tuple[str, ...]]:
    w = norm(s).split()
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def rob_texts() -> list[str]:
    out = []
    for f in sorted(ROB.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "messages" in r:
                out.append(" ".join(m["content"] for m in r["messages"] if m["role"] == "user"))
            elif "user" in r:
                out.append(r["user"])
    return out


def load_seeds(eli5_sample: int = 20000) -> list[dict]:
    seeds = []
    for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
        ctx = (r["context"] or "").strip()
        if len(ctx) > 1500 or not (15 <= len(r["instruction"]) <= 500):
            continue
        text = r["instruction"].strip() + (f"\n\n{ctx}" if ctx else "")
        seeds.append({"source": f"dolly:{r['category']}", "text": text})
    for r in load_dataset("OpenAssistant/oasst1", split="train"):
        if r["parent_id"] is None and r["role"] == "prompter" and r["lang"] == "en" and 20 <= len(r["text"]) <= 700:
            seeds.append({"source": "oasst1", "text": r["text"].strip()})
    eli5 = load_dataset("sentence-transformers/eli5", split="train")
    for i in random.Random(1).sample(range(len(eli5)), eli5_sample):
        q = eli5[i]["question"].strip()
        if 20 <= len(q) <= 300:
            seeds.append({"source": "eli5", "text": q})
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(HERE / "data" / "seeds.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[], help="seed files whose ids (and texts) must not be reused")
    ap.add_argument("--eli5-sample", type=int, default=20000)
    ap.add_argument("--eli5-cap", type=float, default=0.3, help="max share of seeds from ELI5")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rob = rob_texts()
    rob_norm = {norm(t) for t in rob}
    rob_grams = set().union(*(grams(t) for t in rob))
    pool = load_seeds(a.eli5_sample)
    excluded_ids, excluded_norm = set(), set()
    for f in a.exclude:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            excluded_ids.add(r["id"])
            excluded_norm.add(norm(r["text"]))
    rng.shuffle(pool)
    per_source_cap = {"oasst1": 0.4, "eli5": a.eli5_cap}
    counts: dict[str, int] = {}
    seen_norm: set[str] = set()
    kept, dropped = [], {"rob_exact": 0, "rob_ngram": 0, "dup": 0}
    for s in pool:
        n = norm(s["text"])
        if n in rob_norm:
            dropped["rob_exact"] += 1
            continue
        g = grams(s["text"])
        if g and len(g & rob_grams) / len(g) >= 0.3:
            dropped["rob_ngram"] += 1
            continue
        if n in seen_norm or n in excluded_norm:
            dropped["dup"] += 1
            continue
        fam = s["source"].split(":")[0]
        if fam in per_source_cap and counts.get(fam, 0) >= per_source_cap[fam] * a.n:
            continue
        seen_norm.add(n)
        counts[fam] = counts.get(fam, 0) + 1
        length = rng.choices(list(LENGTH_WEIGHTS), weights=list(LENGTH_WEIGHTS.values()))[0]
        kept.append({"id": hashlib.sha256(n.encode()).hexdigest()[:12], **s, "type": rng.choice(TYPES),
                     "length": length, "length_text": LENGTHS[length], "persona": rng.choice(PERSONAS)})
        if len(kept) >= a.n:
            break
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    meta = {"rows": len(kept), "by_source": counts, "dropped": dropped, "rob_prompts": len(rob),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
