#!/usr/bin/env python3
"""Three-arm evaluation for the Anti-slop adapter on held-out briefs.

Arms (any number, named on the command line):

  base      the base model with the task prompt
  prompted  the base model with the task prompt plus an explicit ban on the
            listed tropes (the "could we have just prompted it" control)
  adapter   the DPO adapter, served under any chat-completions model name

Every arm answers the same held-out briefs with the same generation settings.
Each text is scored four ways, none of which was the training reward's
ranking signal except the detector:

  detector   GPTZero P(human), with the detector version recorded
  tropes     mechanical counts from antislop_tropes.py, per 100 words
  format     meets the exact task spec (tweet <= 280 chars; LinkedIn 60-200 words)
  faithful   the four-axis judge's faithfulness score (0-1)

Usage:
  export FIREWORKS_API_KEY=... GPT_ZERO_API_KEY=...
  python evaluate_antislop.py --split test --platforms tweet \\
      --arm base=accounts/fireworks/models/glm-5p3-flash \\
      --arm prompted=accounts/fireworks/models/glm-5p3-flash:ban-tropes \\
      --arm adapter="accounts/<acct>/models/<adapter>#accounts/<acct>/deployments/<dep>" \\
      --out results/eval-test

Readers can run the first two arms with only a Fireworks key and a GPTZero key.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from antislop_tropes import TROPES, trope_report  # noqa: E402
from build_antislop_pack import (  # noqa: E402
    GEN_REASONING_EFFORT,
    GEN_TEMPERATURE,
    SYSTEM_PROMPT,
    api_key,
    brief_prompt,
    chat,
    gptzero_human,
    llm_judge,
    meets_format,
    read_jsonl,
)

BAN_TROPES_SUFFIX = (
    "\n\nStyle rules: no em dashes. Do not use 'not X but Y' framing. No emoji. No hashtags. "
    "No stock phrases such as 'game changer', 'dive in', 'unlock', 'seamless', 'cutting-edge' or "
    "'the future is here'. Do not open with a rhetorical question. Write like a specific person, not a brand."
)

PROMPT_VARIANTS = {"": "", "ban-tropes": BAN_TROPES_SUFFIX}


EDITOR_SYSTEM = ("You are an editor. Rewrite the text so it reads naturally for its audience without changing what it "
                 "says. Keep every fact, number, name, condition, hedge and quotation. Return only the revised text.")
EDITOR_AUDIENCE = "readers of a tech company's social feed"


def parse_arm(spec: str) -> tuple[str, str, str]:
    """'name=model', 'name=model:variant', or 'name=writer>editor' (chain: writer drafts, editor rewrites)."""
    name, rest = spec.split("=", 1)
    if ">" in rest:
        writer, editor = rest.split(">", 1)
        return name.strip(), writer.strip(), "chain:" + editor.strip()
    model, variant = (rest.rsplit(":", 1) + [""])[:2] if ":" in rest.split("#")[-1] else (rest, "")
    if variant not in PROMPT_VARIANTS:
        raise SystemExit(f"unknown prompt variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    return name.strip(), model.strip(), variant


def edit_text(client: httpx.Client, editor: str, text: str, max_tokens: int) -> str:
    d = chat(client, {"model": editor, "max_tokens": max_tokens, "temperature": 0.0, "reasoning_effort": "low",
                      "messages": [{"role": "system", "content": EDITOR_SYSTEM},
                                   {"role": "user", "content": f"Audience: {EDITOR_AUDIENCE}\n\nText:\n{text}"}]})
    return (d["choices"][0]["message"].get("content") or "").strip()


def generate(client: httpx.Client, model: str, brief: dict, variant: str, samples: int, max_tokens: int) -> list[dict]:
    editor = variant[6:] if variant.startswith("chain:") else None
    if editor:
        variant = ""
    body = {"model": model, "max_tokens": max_tokens, "temperature": GEN_TEMPERATURE,
            "reasoning_effort": GEN_REASONING_EFFORT,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT + PROMPT_VARIANTS[variant]},
                         {"role": "user", "content": brief_prompt(brief["platform"], brief["topic"], brief["article"])}]}
    out = []
    for i in range(samples):
        d = chat(client, body)
        c = d["choices"][0]
        text = (c["message"].get("content") or "").strip()
        rec = {"text": text, "finish": c.get("finish_reason"), "completion_tokens": (d.get("usage") or {}).get("completion_tokens")}
        if editor and text:
            rec["draft_before_edit"] = text
            rec["text"] = edit_text(client, editor, text, max_tokens)
        out.append(rec)
    return out


_URL_RE = re.compile(r"https?://\S+|\b[\w-]+\.(?:ai|com|io|org|net)/\S*")


def invented_urls(text: str, brief: dict) -> list[str]:
    """URLs in the post that are not in the article: the writer made them up."""
    src = brief.get("article") or ""
    return [u for u in _URL_RE.findall(text) if u.rstrip(".,)") not in src]


def score_text(client: httpx.Client, text: str, brief: dict, gptzero_key: str | None) -> dict:
    rep = trope_report(text)
    hit = gptzero_human(client, text, gptzero_key) if (gptzero_key and text.strip()) else None
    jud = llm_judge(client, text, brief["platform"], brief["topic"], brief["article"]) if text.strip() else None
    return {"detector": None if hit is None else hit[0], "detector_version": None if hit is None else hit[1],
            "tropes_per_100w": rep["per_100_words"], "trope_counts": rep["counts"], "any_trope": rep["any"],
            "words": rep["words"], "chars": len(text), "format_ok": meets_format(text, brief["platform"]),
            "faithful": None if jud is None else jud["faithful"], "judge": jud,
            "invented_urls": invented_urls(text, brief)}


def aggregate(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    det = [r["detector"] for r in rows if r["detector"] is not None]
    fai = [r["faithful"] for r in rows if r["faithful"] is not None]
    return {
        "n": len(rows),
        "detector_mean": st.mean(det) if det else None,
        "detector_median": st.median(det) if det else None,
        "detector_ge_0.5": (sum(d >= 0.5 for d in det) / len(det)) if det else None,
        "tropes_per_100w": st.mean(r["tropes_per_100w"] for r in rows),
        "share_any_trope": sum(r["any_trope"] for r in rows) / n,
        "share_by_trope": {k: sum(r["trope_counts"][k] > 0 for r in rows) / n for k in TROPES},
        "format_ok": sum(r["format_ok"] for r in rows) / n,
        "faithful_mean": st.mean(fai) if fai else None,
        "faithful_ge_0.8": (sum(f >= 0.8 for f in fai) / len(fai)) if fai else None,
        "mean_words": st.mean(r["words"] for r in rows),
        "share_invented_url": sum(bool(r.get("invented_urls")) for r in rows) / n,
        "detector_failures": sum(r["detector"] is None for r in rows),
        "judge_failures": sum(r["faithful"] is None for r in rows),
    }


def markdown_table(summary: dict) -> str:
    lines = ["| arm | platform | n | detector mean | detector >= 0.5 | tropes /100w | any trope | em dash | emoji | format ok | faithful >= 0.8 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, per_pf in summary.items():
        for pf, a in per_pf.items():
            f = lambda v, pct=False: "n/a" if v is None else (f"{v:.0%}" if pct else f"{v:.3f}")
            lines.append(f"| {arm} | {pf} | {a['n']} | {f(a['detector_mean'])} | {f(a['detector_ge_0.5'], True)} | "
                         f"{a['tropes_per_100w']:.2f} | {f(a['share_any_trope'], True)} | {f(a['share_by_trope']['em_dash'], True)} | "
                         f"{f(a['share_by_trope']['emoji'], True)} | {f(a['format_ok'], True)} | {f(a['faithful_ge_0.8'], True)} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--briefs", default=str(HERE / "pack" / "antislop_briefs.jsonl"))
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--platforms", default="tweet,linkedin")
    ap.add_argument("--arm", action="append", required=True, help="name=model or name=model:ban-tropes (repeatable)")
    ap.add_argument("--samples", type=int, default=2, help="generations per brief per arm")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-briefs", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results" / "eval"))
    a = ap.parse_args()

    api_key()
    gz = os.environ.get("GPT_ZERO_API_KEY")
    if not gz:
        print("GPT_ZERO_API_KEY not set: detector columns will be n/a", flush=True)
    pfs = {p.strip() for p in a.platforms.split(",")}
    briefs = [b for b in read_jsonl(Path(a.briefs)) if b["split"] == a.split and b["platform"] in pfs and b["article"]]
    if a.max_briefs:
        briefs = briefs[: a.max_briefs]
    arms = [parse_arm(s) for s in a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(briefs)} {a.split} briefs x {len(arms)} arms x {a.samples} samples", flush=True)

    jobs = [(name, model, variant, b) for name, model, variant in arms for b in briefs]
    t0 = time.time()

    def run(job):
        name, model, variant, b = job
        with httpx.Client() as client:
            gens = generate(client, model, b, variant, a.samples, a.max_tokens)
            rows = []
            for g in gens:
                s = score_text(client, g["text"], b, gz)
                rows.append({"arm": name, "model": model, "variant": variant, "brief_id": b["id"],
                             "platform": b["platform"], **g, **s})
        return rows

    samples_path = out / "samples.jsonl"
    all_rows: list[dict] = []
    with samples_path.open("w") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for i, rows in enumerate(ex.map(run, jobs), 1):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            all_rows.extend(rows)
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)

    summary = {}
    for name, _, _ in arms:
        summary[name] = {}
        for pf in sorted(pfs):
            rows = [r for r in all_rows if r["arm"] == name and r["platform"] == pf]
            if rows:
                summary[name][pf] = aggregate(rows)
    versions = sorted({r["detector_version"] for r in all_rows if r.get("detector_version")})
    (out / "summary.json").write_text(json.dumps({"split": a.split, "samples_per_brief": a.samples,
                                                  "arms": {n: {"model": m, "variant": v} for n, m, v in arms},
                                                  "detector_versions": versions, "summary": summary}, indent=2) + "\n")
    table = markdown_table(summary)
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\ndetector versions: {versions or 'n/a'}\nwrote {out}")


if __name__ == "__main__":
    main()
