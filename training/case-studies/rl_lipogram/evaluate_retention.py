"""Retention check: does the adapter still answer ordinary exact-answer questions as well as the base?

200 arithmetic and short-factual items with a single exact answer, generated here (no model
calls to build them). Base and adapter each answer every item once at temperature 0 and the
stated effort, with no system prompt. Accuracy is exact match after normalization; the letter
rule is NOT applied, since the question is whether capability survived, not whether the rule
held. Writes summary.json with {"base": acc, "adapter": acc} for check_bar.py.

    python evaluate_retention.py --adapter accounts/pyroworks/models/glm53-lipogram-sft-v2 --effort low --out results/sft-v2-retention-low
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lipogram_rewards import chat  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"
CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Canada": "Ottawa", "Australia": "Canberra", "Brazil": "Brasilia",
            "Egypt": "Cairo", "Kenya": "Nairobi", "Norway": "Oslo", "Peru": "Lima", "Thailand": "Bangkok",
            "Spain": "Madrid", "Italy": "Rome", "Germany": "Berlin", "Poland": "Warsaw", "Chile": "Santiago",
            "Turkey": "Ankara", "Vietnam": "Hanoi", "Greece": "Athens", "Portugal": "Lisbon", "Austria": "Vienna"}
ELEMENTS = {"H": "Hydrogen", "He": "Helium", "O": "Oxygen", "Fe": "Iron", "Au": "Gold", "Ag": "Silver", "Na": "Sodium",
            "K": "Potassium", "C": "Carbon", "N": "Nitrogen", "Cu": "Copper", "Zn": "Zinc", "Pb": "Lead", "Sn": "Tin",
            "Ca": "Calcium", "Mg": "Magnesium", "Al": "Aluminium", "Si": "Silicon", "Cl": "Chlorine", "Ne": "Neon"}


def build_items(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        kind = rng.choice(["add", "mul", "sub", "pct", "capital", "element", "days", "sequence"])
        if kind == "add":
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            items.append({"q": f"What is {a} + {b}? Reply with just the number.", "a": str(a + b)})
        elif kind == "mul":
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            items.append({"q": f"What is {a} × {b}? Reply with just the number.", "a": str(a * b)})
        elif kind == "sub":
            a, b = rng.randint(500, 9999), rng.randint(100, 499)
            items.append({"q": f"What is {a} - {b}? Reply with just the number.", "a": str(a - b)})
        elif kind == "pct":
            p, v = rng.choice([10, 20, 25, 50]), rng.choice([80, 120, 200, 240, 400, 640])
            items.append({"q": f"What is {p}% of {v}? Reply with just the number.", "a": str(p * v // 100)})
        elif kind == "capital":
            c, cap = rng.choice(list(CAPITALS.items()))
            items.append({"q": f"What is the capital of {c}? Reply with just the city name.", "a": cap})
        elif kind == "element":
            sym, name = rng.choice(list(ELEMENTS.items()))
            items.append({"q": f"Which chemical element has the symbol {sym}? Reply with just the name.", "a": name})
        elif kind == "days":
            m, d = rng.choice([("January", 31), ("April", 30), ("June", 30), ("August", 31), ("September", 30), ("December", 31)])
            items.append({"q": f"How many days are in {m}? Reply with just the number.", "a": str(d)})
        else:
            start, step = rng.randint(1, 20), rng.randint(2, 9)
            seq = [start + i * step for i in range(4)]
            items.append({"q": f"What is the next number in the sequence {', '.join(map(str, seq))}? Reply with just the number.", "a": str(start + 4 * step)})
    return items


def norm(s: str) -> str:
    """Exact comparison after trimming a leading 'the answer is' and trailing punctuation. Signs and digits kept."""
    s = s.strip().lower().rstrip(".").strip()
    s = re.sub(r"^(the answer is|answer:)\s*", "", s)
    s = s.replace(",", "") if re.fullmatch(r"[-+]?[\d,]+(\.\d+)?", s) else s
    return re.sub(r"\s+", " ", s)


def is_correct(answer: str, expected: str) -> bool:
    """Exact normalized match only: '120' is not '20', '-42' is not '42', 'Not Paris' is not 'Paris'."""
    return norm(answer) == norm(expected)


def compatible(expected: str) -> bool:
    """An item is constraint-compatible when its exact answer contains no e; the rest are reported separately."""
    return "e" not in expected.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--effort", choices=["low", "max"], default="low")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = build_items(a.n, seed=0)
    arms = {"base": BASE, "adapter": a.adapter}
    jobs = [(arm, i) for arm in arms for i in range(len(items))]

    def one(job):
        arm, i = job
        with httpx.Client() as c:
            d = chat(c, {"model": arms[arm], "messages": [{"role": "user", "content": items[i]["q"]}], "max_tokens": 2048,
                         "temperature": 0.0, "reasoning_effort": a.effort})
        ans = (d["choices"][0]["message"].get("content") or "").strip()
        return {"arm": arm, "i": i, "q": items[i]["q"], "expected": items[i]["a"], "answer": ans,
                "compatible": compatible(items[i]["a"]), "correct": is_correct(ans, items[i]["a"])}

    a.out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "samples.jsonl").open("w") as f:
        recs = []
        for rec in ex.map(one, jobs):
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
    comp = [r for r in recs if r["compatible"]]
    incomp = [r for r in recs if not r["compatible"]]
    n_comp = len({r["i"] for r in comp})
    n_incomp = len({r["i"] for r in incomp})
    acc = {arm: sum(r["correct"] for r in comp if r["arm"] == arm) / max(1, n_comp) for arm in arms}
    acc_incomp = {arm: sum(r["correct"] for r in incomp if r["arm"] == arm) / max(1, n_incomp) for arm in arms}
    (a.out / "summary.json").write_text(json.dumps({"n": len(items), "n_compatible": n_comp, "n_incompatible": n_incomp,
        "effort": a.effort, "adapter": a.adapter, "retention": acc, "retention_incompatible_answers": acc_incomp}, indent=1))
    print(f"retention at effort {a.effort} on {n_comp} constraint-compatible items: base {acc['base']:.1%}  adapter {acc['adapter']:.1%}")
    print(f"items whose exact answer contains an e ({n_incomp}, reported separately, not gated): base {acc_incomp['base']:.1%}  adapter {acc_incomp['adapter']:.1%}")


if __name__ == "__main__":
    main()
