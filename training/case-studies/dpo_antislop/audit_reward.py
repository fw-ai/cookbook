#!/usr/bin/env python3
"""Reward headroom audit for the Anti-slop RL environment. No training pool involved.

Before spending an RL run, answer two questions about the reward the run will climb:

  headroom   Sample the base policy G times on N training briefs through the public inference
             API, score every sample exactly as antislop_rl.py does (format gate -> judge
             eligibility -> GPTZero P(human)), and report the reward distribution, the share of
             groups GRPO would keep (reward spread > 0), and the trope rate. If most groups have
             no spread, RL has no signal.
  hacks      Score hand-written degenerate posts: empty, the article's first sentences copied,
             an emoji wall, a hashtag pile, a one-word post. Anything that scores high and
             passes the gate is a hole RL will find.

Usage:
    export FIREWORKS_API_KEY=... GPT_ZERO_API_KEY=...
    python audit_reward.py --briefs 16 --group 8 --out results/<date>/reward-audit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from antislop_tropes import trope_report  # noqa: E402
from build_antislop_pack import (  # noqa: E402
    ELIGIBILITY_MIN,
    SYSTEM_PROMPT,
    brief_prompt,
    chat,
    gptzero_human,
    llm_judge,
    meets_format,
    read_jsonl,
)

POLICY_MODEL = "accounts/fireworks/models/glm-5p3-flash"


def score_one(content: str, b: dict, key: str) -> dict:
    """Mirror of ServerlessAntislopRL._score_one, with the intermediate verdicts kept."""
    tr = trope_report(content)
    rec = {"text": content, "tropes_per_100w": tr["per_100_words"], "format_ok": bool(content.strip()) and meets_format(content, b["platform"]),
           "judge": None, "eligible": False, "p_human": None, "reward": 0.0}
    if not rec["format_ok"]:
        return rec
    with httpx.Client() as c:
        jud = llm_judge(c, content, b["platform"], b["topic"], b["article"])
        rec["judge"] = jud
        if jud is None or any(jud[k] < ELIGIBILITY_MIN for k in ("faithful", "coherent", "on_format", "specific")):
            return rec
        hit = gptzero_human(c, content, key)
    if hit is None:
        return rec
    rec.update(eligible=True, p_human=float(hit[0]), reward=float(hit[0]))
    return rec


def hack_posts(b: dict) -> dict[str, str]:
    art = b["article"]
    first = " ".join(art.split(". ")[:2])[:270]
    return {
        "empty": "",
        "one_word": "Fireworks.",
        "copied_excerpt": first,
        "emoji_wall": "🔥🚀✨💡🎉 " * 8,
        "hashtag_pile": f"{b['topic'].title()} #AI #LLM #Fireworks #ML #DevTools #GenAI #Startup #Tech",
        "topic_only": f"New on the Fireworks blog: {b['topic']}.",
        "plain_human": f"we wrote up {b['topic']}. it's on the blog if you're into that sort of thing",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--briefs", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results" / time.strftime("%Y-%m-%d") / "reward-audit"))
    a = ap.parse_args()
    gz = os.environ.get("GPT_ZERO_API_KEY") or os.environ.get("GPTZERO_API_KEY")
    if not gz:
        raise SystemExit("set GPT_ZERO_API_KEY")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    briefs = [b for b in read_jsonl(HERE / "pack" / "antislop_briefs.jsonl") if b["split"] == "train" and b["platform"] == "tweet"]
    rng = random.Random(a.seed)
    briefs = rng.sample(briefs, min(a.briefs, len(briefs)))
    print(f"{len(briefs)} tweet briefs x {a.group} samples from base; plus {len(hack_posts(briefs[0]))} hack probes x 4 briefs")

    # ---- headroom: sample the base policy
    def sample(job):
        b, i = job
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": brief_prompt(b["platform"], b["topic"], b["article"])}]
        with httpx.Client() as c:
            d = chat(c, {"model": POLICY_MODEL, "messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature,
                         "reasoning_effort": "low"})
        ch = d["choices"][0]
        return b["id"], i, (ch["message"].get("content") or "").strip(), ch.get("finish_reason")

    jobs = [(b, i) for b in briefs for i in range(a.group)]
    with ThreadPoolExecutor(a.concurrency) as ex:
        samples = list(ex.map(sample, jobs))
    bmap = {b["id"]: b for b in briefs}

    def score_sample(s):
        bid, i, text, fin = s
        rec = score_one(text, bmap[bid], gz)
        rec.update(brief_id=bid, sample=i, finish=fin, kind="sample")
        return rec

    with ThreadPoolExecutor(a.concurrency) as ex:
        scored = list(ex.map(score_sample, samples))

    # ---- hacks: 4 briefs x each probe
    hack_jobs = [(b, name, text) for b in briefs[:4] for name, text in hack_posts(b).items()]

    def score_hack(j):
        b, name, text = j
        rec = score_one(text, b, gz)
        rec.update(brief_id=b["id"], probe=name, kind="hack")
        return rec

    with ThreadPoolExecutor(a.concurrency) as ex:
        hacks = list(ex.map(score_hack, hack_jobs))

    with (out / "samples.jsonl").open("w") as f:
        for r in scored + hacks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary
    rewards = [r["reward"] for r in scored]
    elig = [r for r in scored if r["eligible"]]
    groups = {}
    for r in scored:
        groups.setdefault(r["brief_id"], []).append(r["reward"])
    spread = [g for g in groups.values() if max(g) - min(g) > 1e-9]
    gate_fail = sum(not r["format_ok"] for r in scored)
    judge_fail = sum(r["format_ok"] and not r["eligible"] for r in scored)
    summary = {
        "briefs": len(briefs), "group": a.group, "samples": len(scored),
        "format_gate_fail_rate": gate_fail / len(scored), "judge_fail_rate": judge_fail / len(scored), "eligible_rate": len(elig) / len(scored),
        "reward_mean": st.mean(rewards), "reward_mean_eligible": st.mean(r["reward"] for r in elig) if elig else None,
        "reward_quantiles_eligible": [round(q, 3) for q in st.quantiles([r["reward"] for r in elig], n=4)] if len(elig) > 3 else None,
        "groups_with_spread": f"{len(spread)}/{len(groups)}", "mean_group_range": st.mean(max(g) - min(g) for g in groups.values()),
        "tropes_per_100w_mean": st.mean(r["tropes_per_100w"] for r in scored),
        "truncated": sum(r["finish"] != "stop" for r in scored),
        "hacks": {},
    }
    for name in hack_posts(briefs[0]):
        hs = [h for h in hacks if h["probe"] == name]
        summary["hacks"][name] = {"eligible": sum(h["eligible"] for h in hs), "of": len(hs),
                                  "max_reward": max(h["reward"] for h in hs), "format_ok": sum(h["format_ok"] for h in hs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"**Headroom** ({len(briefs)} briefs x {a.group} samples, base policy, T={a.temperature})", "",
             "| metric | value |", "|---|---|",
             f"| format gate fail | {summary['format_gate_fail_rate']:.0%} |", f"| judge fail (eligibility) | {summary['judge_fail_rate']:.0%} |",
             f"| eligible | {summary['eligible_rate']:.0%} |", f"| reward mean (all / eligible) | {summary['reward_mean']:.3f} / {summary['reward_mean_eligible'] or 0:.3f} |",
             f"| eligible reward quartiles | {summary['reward_quantiles_eligible']} |", f"| groups with spread | {summary['groups_with_spread']} |",
             f"| mean group range | {summary['mean_group_range']:.3f} |", f"| tropes / 100 words | {summary['tropes_per_100w_mean']:.2f} |",
             f"| truncated | {summary['truncated']} |", "", "**Hack probes** (4 briefs each)", "", "| probe | passes gate | eligible | max reward |", "|---|---|---|---|"]
    for name, h in summary["hacks"].items():
        lines.append(f"| {name} | {h['format_ok']}/{h['of']} | {h['eligible']}/{h['of']} | {h['max_reward']:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
