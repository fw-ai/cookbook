"""Build the lipogram prompt set from inputs this gallery already has.

No model calls. Three families, group-disjoint train/test:

- ``caveman``: the Caveman task questions (a question plus a facts block); the facts give the
  judge something to check faithfulness against.
- ``rewrite``: Humanizer source passages, asked to be restated for a reader. Long enough that a
  single slipped "e" is likely, which is where the adapter has to earn its keep.
- ``plain``: short open questions written here, with no context, in three length bands. These
  are the demo prompts and the off-distribution check.

Each row: ``{"messages": [system, user], "ground_truth": json}`` like the other RL studies.
The system prompt states the constraint, so the prompted-base arm is a fair comparison; the
adapter is evaluated with and without it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent

SYSTEM_PROMPT = (
    "Answer the user without using the letter \"e\" anywhere in your final answer. Not even once, in any word. "
    "Your separate reasoning may use any letters: plan, check and revise before giving the final answer. "
    "Answer usefully in English, using natural paraphrases. The rule applies to all final-answer text, "
    "including headings, quotations and code fences. Do not disguise letters with escapes or look-alikes."
)

PLAIN = [
    ("short", "What is a LoRA adapter?"),
    ("short", "Why is the sky blue?"),
    ("short", "How do I boil an egg?"),
    ("short", "What does HTTP 404 mean?"),
    ("short", "Name three uses for a Raspberry Pi."),
    ("short", "What is the capital of Australia?"),
    ("short", "How does a bicycle stay upright?"),
    ("short", "What is a prime number?"),
    ("short", "Why do leaves change color in autumn?"),
    ("short", "What is the difference between RAM and disk?"),
    ("short", "How do vaccines work?"),
    ("short", "What is compound interest?"),
    ("medium", "Explain to a new engineer what a LoRA adapter is and why a team would use one."),
    ("medium", "Describe how a hash map works and when you would pick one over a list."),
    ("medium", "Give me a plan for a first week of running, for someone who has not run before."),
    ("medium", "Explain what a TLS certificate is and what happens when one expires."),
    ("medium", "What should I check before merging a pull request? Give a short list."),
    ("medium", "Explain the difference between supervised fine-tuning and reinforcement learning."),
    ("medium", "How does a credit score work and what moves it?"),
    ("medium", "Describe the water cycle for a ten year old."),
    ("medium", "What is technical debt and how do teams pay it down?"),
    ("medium", "Explain why floating point arithmetic can give surprising results."),
    ("medium", "What makes a good commit message? Give an example."),
    ("medium", "How do I choose between SQL and a document database for a new app?"),
    ("long", "Write a short guide to setting up a Python project from scratch: environment, dependencies, tests and a first script."),
    ("long", "Explain how a web request travels from a browser to a server and back, naming each stage."),
    ("long", "Describe the causes of the 2008 financial crisis in a few paragraphs."),
    ("long", "Write an incident postmortem template and explain what goes in each section."),
    ("long", "Explain how large language models are trained, from data to a chat assistant."),
    ("long", "Give a beginner's introduction to version control with git: what it tracks, branches, merges and why it matters."),
    ("long", "Describe how to plan a two week trip to Japan on a modest budget."),
    ("long", "Explain photosynthesis in detail, including what goes in, what comes out and why it matters for the planet."),
    ("long", "Write a product announcement for a new log tailing CLI for Kubernetes, about 200 words."),
    ("long", "Explain how public key cryptography makes online shopping safe."),
    ("long", "Describe the job of a site reliability engineer and a typical on-call week."),
    ("long", "Write a cover letter for a junior data analyst role at a small company."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-caveman", type=int, default=24)
    ap.add_argument("--test-rewrite", type=int, default=24)
    ap.add_argument("--test-plain", type=int, default=12)
    ap.add_argument("--extra", default="", help="generated_prompts.jsonl from generate_prompts.py; each (topic, family) is a group")
    ap.add_argument("--test-extra-groups", type=int, default=40, help="held-out (topic, family) groups from --extra")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for t in (json.loads(line) for line in (STUDIES / "sft_caveman/data/caveman_tasks.jsonl").read_text().splitlines() if line.strip()):
        facts = "\n".join(f"- {f}" for f in t["facts"]) if isinstance(t["facts"], list) else str(t["facts"])
        user = f"Task information:\n{facts}\n\n{t['question']}"
        rows.append({"family": "caveman", "group_id": f"cv-{t['group_id']}", "user": user, "facts": facts,
                     "question": t["question"], "source_id": t["id"]})
    for r in (json.loads(line) for line in (STUDIES / "sft_humanizer/data/humanizer_inputs.jsonl").read_text().splitlines() if line.strip()):
        if len(r["source"].split()) > 220:
            continue
        user = f"Restate this for {r['audience']}, keeping every fact:\n\n{r['source']}"
        rows.append({"family": "rewrite", "group_id": f"hz-{r['group_id']}", "user": user, "facts": r["source"],
                     "question": f"Restate for {r['audience']}", "source_id": r["id"]})
    for i, (band, q) in enumerate(PLAIN):
        rows.append({"family": "plain", "group_id": f"pl-{i}", "user": q, "facts": "", "question": q,
                     "source_id": f"plain-{i:02d}", "band": band})
    if a.extra:
        seen = set()
        for k, g in enumerate(json.loads(line) for line in Path(a.extra).read_text().splitlines() if line.strip()):
            if g["user"] in seen:
                continue
            seen.add(g["user"])
            rows.append({"family": "gen-" + g["family"], "group_id": f"gen-{g['topic']}|{g['family']}", "user": g["user"],
                         "facts": "", "question": g["user"], "source_id": f"gen-{k:05d}"})

    # Group-disjoint split: hold out whole groups per family.
    by_family: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_family.setdefault(r["family"], {}).setdefault(r["group_id"], []).append(r)
    want = {"caveman": a.test_caveman, "rewrite": a.test_rewrite, "plain": a.test_plain}
    gen_fams = sorted(f for f in by_family if f.startswith("gen-"))
    for f in gen_fams:
        want[f] = 10 ** 9   # resolved below as whole groups
    train, test = [], []
    gen_groups = sorted(g for fam in gen_fams for g in by_family[fam])
    rng.shuffle(gen_groups)
    gen_held = set(gen_groups[: a.test_extra_groups])
    for fam, groups in by_family.items():
        gids = sorted(groups)
        rng.shuffle(gids)
        if fam.startswith("gen-"):
            held = [g for g in gids if g in gen_held]
        else:
            held, n = [], 0
            for g in gids:
                if n >= want[fam]:
                    break
                held.append(g)
                n += len(groups[g])
        for g in gids:
            (test if g in held else train).extend(groups[g])

    def row(r: dict) -> dict:
        gt = {k: r[k] for k in ("family", "group_id", "question", "facts", "source_id")}
        if "band" in r:
            gt["band"] = r["band"]
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["user"]}],
                "ground_truth": json.dumps(gt)}

    rng.shuffle(train)
    (out / "lipogram_train.jsonl").write_text("".join(json.dumps(row(r)) + "\n" for r in train))
    (out / "lipogram_test.jsonl").write_text("".join(json.dumps(row(r)) + "\n" for r in test))
    fam = lambda rs: {f: sum(1 for r in rs if r["family"] == f) for f in sorted({r["family"] for r in rows})}  # noqa: E731
    print(f"train {len(train)} {fam(train)}  test {len(test)} {fam(test)}  -> {out}")


if __name__ == "__main__":
    main()
