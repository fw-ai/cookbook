#!/usr/bin/env python3
"""Evaluate Humanizer adapters against the held-out inputs.

Arms are chat-completions model names with an optional prompt mode suffix:
  name=model            student prompt (the two-sentence editor instruction)
  name=model:plain      "Rewrite this to read naturally. Return only the text."
  name=model:skill      the full Humanizer skill as the system prompt (the teacher)

Every arm edits every held-out input greedily. Each output gets the pack
filter's own checks: preservation (must_preserve, protected, no new numbers
or URLs, length ratio, no preamble) and the judge (meaning_preserved,
no_additions, natural, on_request). Then an IDEMPOTENCE pass: each arm edits
its own output once more; the character similarity between the two says
whether the editor leaves good prose alone.

Headline metrics per arm:
  preserve_rate   share of outputs with no mechanical problem
  pass_rate       preserve AND judge >= 0.8 on all four axes
  over_edit_good  share of class-good inputs changed by more than 15%
  idem_median     median similarity between first and second pass
  idem_changed    share of second passes that changed > 10%

Before any paid call the judge is checked against hand-written known answers
(`calibrate_judge.py`). With --draws N > 1 every input is edited N times at
temperature 0.7 and the summary adds pass@k and pass^k for k = 1..N.

Usage:
  export FIREWORKS_API_KEY=...
  python evaluate_pack.py --split test --arm base_plain=...:plain --arm base_short=... \
      --arm skill=...:skill --arm adapter=accounts/<acct>/models/<id>#<deployment> --out results/<date>/eval
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_humanizer_pack import (  # noqa: E402
    GOOD_MIN_SIMILARITY,
    JUDGE_MIN,
    api_key,
    chat,
    extract_final,
    judge,
    preservation_checks,
    read_jsonl,
    similarity,
    skill_body,
    student_messages,
)

PLAIN_SYSTEM = "Rewrite this to read naturally. Return only the text."


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k draws passes, given c of n passed (unbiased, Codex-style)."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Chance that all k draws pass: the consistency number."""
    if c < k:
        return 0.0
    p = 1.0
    for i in range(k):
        p *= (c - i) / (n - i)
    return p


def pass_curves(rows: list[dict], draws: int) -> dict:
    """Mean pass@k and pass^k over fixtures for k = 1..draws, from rows that carry id/draw/pass."""
    by_id: dict[str, list[bool]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(bool(r["pass"]))
    ks = list(range(1, draws + 1))
    at = [st.mean(pass_at_k(len(v), sum(v), k) for v in by_id.values()) for k in ks]
    pw = [st.mean(pass_pow_k(len(v), sum(v), k) for v in by_id.values()) for k in ks]
    return {"k": ks, "pass_at": at, "pass_pow": pw}


def run_calibration() -> None:
    import subprocess
    print("judge calibration (known answers)...")
    if subprocess.run([sys.executable, str(HERE / "calibrate_judge.py")]).returncode:
        raise SystemExit("judge calibration failed; pass --no-calibrate to override")


def parse_arm(spec: str) -> tuple[str, str, str]:
    name, rest = spec.split("=", 1)
    mode = "student"
    for suffix in (":plain", ":skill"):
        if rest.endswith(suffix):
            mode, rest = suffix[1:], rest[: -len(suffix)]
    return name.strip(), rest.strip(), mode


def messages_for(mode: str, audience: str, text: str) -> list[dict]:
    if mode == "plain":
        return [{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": text}]
    if mode == "skill":
        return [{"role": "system", "content": skill_body() + "\n\nEmbedded mode: return only the final text."},
                {"role": "user", "content": f"Embedded mode: reply with ONLY the final rewritten text. No draft, no list of "
                                            f"patterns, no notes.\n\nAudience: {audience}\n\nText:\n{text}"}]
    return student_messages(audience, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-inputs", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--no-idem", action="store_true")
    ap.add_argument("--draws", type=int, default=1, help="samples per fixture; >1 reports pass@k and pass^k")
    ap.add_argument("--temperature", type=float, default=None, help="default 0.0 for one draw, 0.7 for several")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the known-answer judge gate")
    ap.add_argument("--out", default=str(HERE / "results" / time.strftime("%Y-%m-%d") / "pack-eval"))
    a = ap.parse_args()
    api_key()
    if not a.no_calibrate:
        run_calibration()
    temperature = a.temperature if a.temperature is not None else (0.0 if a.draws == 1 else 0.7)
    inputs = read_jsonl(HERE / "pack" / f"humanizer_{a.split}.jsonl")
    if a.max_inputs:
        inputs = inputs[: a.max_inputs]
    arms = [parse_arm(s) for s in a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(inputs)} {a.split} inputs x {len(arms)} arms" + ("" if a.no_idem else " + idempotence pass"))

    def edit(model: str, mode: str, audience: str, text: str) -> tuple[str, str]:
        with httpx.Client() as c:
            d = chat(c, {"model": model, "max_tokens": a.max_tokens, "temperature": temperature, "reasoning_effort": "low",
                         "messages": messages_for(mode, audience, text)})
        ch = d["choices"][0]
        raw = ch["message"].get("content") or ""
        return (extract_final(raw) if mode == "skill" else raw.strip()), ch.get("finish_reason") or ""

    def run(job):
        name, model, mode, r, draw = job
        text, fin = edit(model, mode, r["audience"], r["source"])
        probs = preservation_checks(text, r)
        if fin != "stop":
            probs.append("truncated")
        with httpx.Client() as c:
            j = judge(c, text, r) if text.strip() else None
        judge_ok = j is not None and all(v >= JUDGE_MIN for v in j.values())
        sim = similarity(r["source"], text)
        rec = {"arm": name, "model": model, "mode": mode, "id": r["id"], "draw": draw, "cls": r["cls"], "genre": r["genre"], "text": text,
               "finish": fin, "similarity": round(sim, 3), "length_ratio": round(len(text.split()) / max(1, len(r["source"].split())), 3),
               "problems": probs, "judge": j, "preserve": not probs, "pass": (not probs) and judge_ok,
               "em_dash_added": "em dash added" in probs,
               "over_edit": r["cls"] == "good" and sim < GOOD_MIN_SIMILARITY}
        if not a.no_idem and text.strip():
            again, _ = edit(model, mode, r["audience"], text)
            rec["idem_similarity"] = round(similarity(text, again), 3)
        return rec

    jobs = [(n, m, mode, r, d) for n, m, mode in arms for r in inputs for d in range(a.draws)]
    rows: list[dict] = []
    t0 = time.time()
    with (out / "samples.jsonl").open("w") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for i, rec in enumerate(ex.map(run, jobs), 1):
            rows.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)

    summary = {}
    hdr = "| arm | preserve | pass | em dash added | judge meaning | judge natural | over-edit (good) | idem median | idem changed >10% | median length ratio |"
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for name, _, _ in arms:
        rs = [r for r in rows if r["arm"] == name]
        n = max(1, len(rs))
        js = [r["judge"] for r in rs if r["judge"]]
        good = [r for r in rs if r["cls"] == "good"]
        idem = [r["idem_similarity"] for r in rs if "idem_similarity" in r]
        s = {"n": len(rs), "preserve_rate": sum(r["preserve"] for r in rs) / n, "pass_rate": sum(r["pass"] for r in rs) / n,
             "em_dash_added_rate": sum(r.get("em_dash_added", False) for r in rs) / n,
             "judge": {k: (st.mean(j[k] for j in js) if js else None) for k in ("meaning_preserved", "no_additions", "natural", "on_request")},
             "over_edit_good": (sum(r["over_edit"] for r in good) / len(good)) if good else None,
             "idem_median": st.median(idem) if idem else None, "idem_changed": (sum(x < 0.9 for x in idem) / len(idem)) if idem else None,
             "median_length_ratio": st.median(r["length_ratio"] for r in rs),
             "by_cls": {c: {"n": sum(r["cls"] == c for r in rs), "pass_rate": sum(r["pass"] for r in rs if r["cls"] == c) / max(1, sum(r["cls"] == c for r in rs))}
                        for c in sorted({r["cls"] for r in rs})}}
        if a.draws > 1:
            s["curves"] = pass_curves(rs, a.draws)
        summary[name] = s
        f = lambda v, fmt: ("n/a" if v is None else fmt.format(v))
        lines.append(f"| {name} | {s['preserve_rate']:.0%} | {s['pass_rate']:.0%} | {s['em_dash_added_rate']:.0%} | {f(s['judge']['meaning_preserved'], '{:.2f}')} | "
                     f"{f(s['judge']['natural'], '{:.2f}')} | {f(s['over_edit_good'], '{:.0%}')} | {f(s['idem_median'], '{:.2f}')} | "
                     f"{f(s['idem_changed'], '{:.0%}')} | {s['median_length_ratio']:.2f} |")
    if a.draws > 1:
        lines += ["", f"pass@k / pass^k over {a.draws} draws at temperature {temperature}:", "| arm | " + " | ".join(f"k={k}" for k in range(1, a.draws + 1)) + " |",
                  "|---|" + "---|" * a.draws]
        for name, _, _ in arms:
            cv = summary[name]["curves"]
            lines.append(f"| {name} | " + " | ".join(f"{at:.0%} / {pw:.0%}" for at, pw in zip(cv["pass_at"], cv["pass_pow"])) + " |")
    table = "\n".join(lines)
    (out / "summary.json").write_text(json.dumps({"split": a.split, "draws": a.draws, "temperature": temperature, "arms": {n: {"model": m, "mode": mode} for n, m, mode in arms},
                                                  "summary": summary}, indent=2) + "\n")
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
