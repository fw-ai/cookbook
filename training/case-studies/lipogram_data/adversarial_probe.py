#!/usr/bin/env python3
"""Probe a served lipogram adapter with prompts that tempt accented, look-alike or escaped e's. Paid calls.

Sets: (A) loanword / accented-name / script-switch / escape traps, (B) held-out seeds never used for data
(data/seeds_p3.jsonl), (C) the rl_lipogram test prompts the adapter found hardest, re-sampled at T=1.0.
Every response is judged with rl_lipogram's llm_judge and audited with audit_unicode.audit.

    python adversarial_probe.py --adapter accounts/research-train/models/arxm-lipo-p2-rl-s100 --out results/adv-s100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
RL = HERE.parent / "rl_lipogram"
sys.path.insert(0, str(RL))
sys.path.insert(0, str(HERE))
from audit_unicode import audit  # noqa: E402
from build_prompts import SYSTEM_PROMPT  # noqa: E402
from lipogram_data import ground_truth, user_messages  # noqa: E402
from lipogram_rewards import chat, llm_judge, lipogram_reward  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"

TRAPS = [
    # French loanwords
    "What's the difference between a café au lait and a latte?",
    "How do I write a strong résumé for a first software engineering job?",
    "Give me a step-by-step recipe for crème brûlée.",
    "How do I sauté mushrooms so they brown instead of going soggy?",
    "What is the difference between a fiancé and a fiancée?",
    "Give me five clichés to avoid in a cover letter, and what to write instead.",
    "What does 'touché' mean and when do people say it?",
    "Explain what a coup d'état is and give two historical examples.",
    "What is déjà vu and what do scientists think causes it?",
    "What is crème fraîche and what can I substitute for it?",
    "Use the word 'protégé' in three example sentences.",
    "What is an exposé in journalism? Name a famous one.",
    "At a French restaurant, what do entrée, plat principal and dessert mean?",
    "What is pâté made of and how is it served?",
    "Explain the phrase 'vis-à-vis' with an example.",
    "What happens during a rugby mêlée (scrum)?",
    "How do I make a proper French crêpe batter?",
    "What should I wear to a soirée?",
    "What is the Champs-Élysées and why is it famous?",
    "What is a fête in France, and how is the Fête de la Musique celebrated?",
    "How do the French celebrate Noël?",
    "What does 'naïve' mean, and is 'naive' without the dots also correct?",
    "Where does the word 'doppelgänger' come from?",
    "What's the best time of year to visit Zürich, and what should I see?",
    "How spicy is a jalapeño compared with a habanero?",
    "What is a piñata and where does the tradition come from?",
    "What is the difference between a café and a bistro?",
    # Accented names and brands
    "Who is Beyoncé and what are her most famous albums?",
    "How does evolution work in Pokémon games?",
    "Who was René Descartes and what did 'cogito, ergo sum' mean?",
    "What was Pelé famous for?",
    "Name three films starring Renée Zellweger.",
    "Who were the Brontë sisters and what did each write?",
    "What is Émile Zola known for, and what was 'J'accuse'?",
    "Who is Michael Bublé?",
    "What did Kurt Gödel prove with his incompleteness theorems?",
    "What is Nietzsche's idea of the Übermensch?",
    "Tell me about the band Mötley Crüe.",
    "What is Häagen-Dazs and where was it founded?",
    "Who was Marie Curie's husband Pierre, and what did they discover together?",
    "Who painted 'Les Demoiselles d'Avignon' and why was it shocking?",
    "Who is Chloë Grace Moretz?",
    "Who is the footballer Kylian Mbappé?",
    # Requests that ask for the accented or foreign form directly
    "Spell the word 'café' exactly as it is written in French, with the accent.",
    "Write 'résumé' with its accents and explain why the accents are there.",
    "List five French desserts using their proper French spelling with accents.",
    "Write the full name of the Pokémon franchise exactly as it is trademarked.",
    "How do you say 'hello' and 'thank you' in Russian? Write them in Cyrillic.",
    "Write the Greek letter epsilon and explain what it means in calculus.",
    "Translate 'the tea is ready' into French.",
    "Write one sentence in Spanish and one in German about the weather.",
    "What is the HTML entity for the letter é?",
    "Write a Python line that prints the word Hello.",
    "What is Euler's number e, and what is it approximately equal to?",
    "Write the JSON object {\"name\": \"Renee\", \"role\": \"engineer\"} for me.",
    "Write a SQL statement that creates an index named idx_users_email on users(email).",
    "Write a regular expression that matches release tags like rel-1.2 or rel-12.34.",
    "Give me the exact shell command to install the latest version of a Go tool with go install.",
    "Write a short haiku about Zürich in winter.",
    "Write a two-line toast for my friend René's birthday.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard-from", type=Path, default=HERE / "results/p2-rl-s100-served-low/samples.jsonl")
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--n-hard", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    seeds = [json.loads(l) for l in open(HERE / "data/seeds_p3.jsonl")]
    seeds = random.Random(7).sample(seeds, a.n_seeds)
    test = [json.loads(l) for l in open(RL / "data/lipogram_test.jsonl")]
    by_prompt: dict[int, list[bool]] = {}
    for line in open(a.hard_from):
        r = json.loads(line)
        if r["arm"] == "adapter":
            by_prompt.setdefault(r["prompt_index"], []).append(r["success"])
    hard = sorted(by_prompt, key=lambda i: (sum(by_prompt[i]), i))[: a.n_hard]

    jobs = []
    for i, q in enumerate(TRAPS):
        item = {"set": "trap", "id": f"trap-{i}", "question": q, "messages": [{"role": "user", "content": q}], "facts": ""}
        jobs += [(item, a.adapter, False, d, 0.0 if d == 0 else 0.7) for d in range(3)]
        jobs.append((item, BASE, True, 0, 0.0))
    for s in seeds:
        item = {"set": "heldout_seed", "id": s["id"], "question": s["text"], "messages": [{"role": "user", "content": s["text"]}], "facts": ""}
        jobs += [(item, a.adapter, False, d, 0.0 if d == 0 else 0.7) for d in range(2)]
    for i in hard:
        gt = ground_truth(test[i])
        item = {"set": "hard_test", "id": f"test-{i}", "question": gt["question"], "messages": user_messages(test[i]),
                "facts": gt.get("facts", "")}
        jobs += [(item, a.adapter, False, d + 1, 1.0) for d in range(3)]

    def one(job):
        item, model, prompted, draw, temp = job
        msgs = ([{"role": "system", "content": SYSTEM_PROMPT}] if prompted else []) + item["messages"]
        with httpx.Client() as client:
            res = chat(client, {"model": model, "messages": msgs, "max_tokens": 8192, "temperature": temp, "reasoning_effort": "low"})
            choice = res["choices"][0]
            answer = (choice["message"].get("content") or "").strip()
            judge = llm_judge(client, item["question"], answer, item["facts"])
        _, free, answered, rate = lipogram_reward(answer, judge, phase="strict")
        return {"set": item["set"], "id": item["id"], "arm": "base+prompt" if prompted else "adapter", "draw": draw,
                "temperature": temp, "question": item["question"], "response": answer, "finish_reason": choice.get("finish_reason"),
                "e_free": free, "answered": answered if judge else None, "completeness": (judge or {}).get("answers"),
                "success": bool(free and answered and choice.get("finish_reason") == "stop"), "e_rate": rate,
                "words": len(answer.split()), "audit": audit(answer, item["question"])}

    print(f"{len(jobs)} calls")
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "samples.jsonl").open("w") as f:
        for n, rec in enumerate(ex.map(one, jobs), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(n, flush=True)
    print("done")


if __name__ == "__main__":
    main()
