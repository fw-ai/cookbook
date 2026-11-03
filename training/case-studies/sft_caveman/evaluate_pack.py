#!/usr/bin/env python3
"""Evaluate Caveman adapters against the held-out test fixtures.

Arms are chat-completions model names. Typical:
  base       accounts/fireworks/models/glm-5p3-flash[#deployment]
  skill      the same base with the Caveman skill prompt appended   (name=model:skill)
  adapter    accounts/<acct>/models/<adapter>#accounts/<acct>/deployments/<dep>

Every arm answers every test fixture with the STUDENT prompt (no skill, except
the explicit :skill arm), greedy, same token cap. Each answer is checked the
way the pack filter checked training targets:

  literals   all required literals present (negation family for 'not')
  numbers    every number is in the input or the expected answer
  words      length of the visible answer
  judge      supported / complete / on_request from the pack's judge model

The headline is "pass": literals and numbers clean AND judge >= 0.8 on all
three axes. Report pass rate, mean words, and the judge means per arm.

Before any paid call the judge is checked against hand-written known answers
(`calibrate_judge.py`). With --draws N > 1 every fixture is sampled N times at
temperature 0.7 and the summary adds pass@k (any of k draws passes) and pass^k
(all k pass) for k = 1..N; pass^k is the consistency claim.

Usage:
  export FIREWORKS_API_KEY=...
  python evaluate_pack.py --arm base=... --arm skill=...:skill --arm adapter=... --out results/<date>/eval
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
from build_caveman_pack import (  # noqa: E402
    JUDGE_MIN,
    SKILL_SUFFIX,
    api_key,
    chat,
    judge,
    literal_checks,
    read_jsonl,
    skill_body,
)


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


def parse_arm(spec: str) -> tuple[str, str, bool]:
    name, rest = spec.split("=", 1)
    use_skill = rest.endswith(":skill")
    model = rest[:-6] if use_skill else rest
    return name.strip(), model.strip(), use_skill


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--draws", type=int, default=1, help="samples per fixture; >1 reports pass@k and pass^k")
    ap.add_argument("--temperature", type=float, default=None, help="default 0.0 for one draw, 0.7 for several")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the known-answer judge gate")
    ap.add_argument("--out", default=str(HERE / "results" / time.strftime("%Y-%m-%d") / "pack-eval"))
    ap.add_argument("--pack-dir", type=Path, default=HERE / "release" / "v2c", help="where caveman_{dev,test}.jsonl live (defaults to the bundled v2c release)")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "max"])
    ap.add_argument("--no-system-prompt", action="store_true",
                    help="send the user turn alone (for adapters trained with --drop-system-prompt); the :skill arm keeps its prompt")
    ap.add_argument("--limit", type=int, default=0, help="first N fixtures only")
    a = ap.parse_args()
    api_key()
    if not a.no_calibrate:
        run_calibration()
    temperature = a.temperature if a.temperature is not None else (0.0 if a.draws == 1 else 0.7)
    fixtures = read_jsonl(a.pack_dir / f"caveman_{a.split}.jsonl")
    if a.limit:
        fixtures = fixtures[: a.limit]
    arms = [parse_arm(s) for s in a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "\n\n" + skill_body() + SKILL_SUFFIX
    print(f"{len(fixtures)} {a.split} fixtures x {len(arms)} arms")

    def run(job):
        name, model, use_skill, fx, draw = job
        msgs = [dict(m) for m in fx["student_messages"]]
        if use_skill:
            msgs[0]["content"] += suffix
        elif a.no_system_prompt:
            msgs = [m for m in msgs if m["role"] != "system"]
        with httpx.Client() as c:
            d = chat(c, {"model": model, "max_tokens": a.max_tokens, "temperature": temperature, "reasoning_effort": a.effort, "messages": msgs}, stage="eval")
            ch = d["choices"][0]
            text = (ch["message"].get("content") or "").strip()
            probs = literal_checks(text, fx)
            if ch.get("finish_reason") != "stop":
                probs.append("truncated")   # the pack filter applies the same rule
            j = judge(c, text, fx) if text else None
        judge_ok = j is not None and all(v >= JUDGE_MIN for v in j.values())
        return {"arm": name, "model": model, "skill": use_skill, "id": fx["id"], "draw": draw, "family": fx["family"], "text": text,
                "words": len(text.split()), "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
                "finish": ch.get("finish_reason"), "problems": probs, "judge": j,
                "judge_pass": judge_ok, "pass": (not probs) and judge_ok}

    jobs = [(n, m, s, fx, d) for n, m, s in arms for fx in fixtures for d in range(a.draws)]
    rows: list[dict] = []
    t0 = time.time()
    with (out / "samples.jsonl").open("w") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for i, r in enumerate(ex.map(run, jobs), 1):
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)

    summary = {}
    lines = ["| arm | pass | judge pass | literals+numbers clean | judge supported | judge complete | judge on_request | mean words | median words |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, _, _ in arms:
        rs = [r for r in rows if r["arm"] == name]
        n = max(1, len(rs))
        js = [r["judge"] for r in rs if r["judge"]]
        s = {"n": len(rs), "pass_rate": sum(r["pass"] for r in rs) / n, "clean_rate": sum(not r["problems"] for r in rs) / n,
             "judge_pass_rate": sum(r.get("judge_pass", False) for r in rs) / n,
             "judge": {k: (st.mean(j[k] for j in js) if js else None) for k in ("supported", "complete", "on_request")},
             "mean_words": st.mean(r["words"] for r in rs), "median_words": st.median(r["words"] for r in rs),
             "by_family": {fam: {"n": sum(r["family"] == fam for r in rs), "pass_rate": (sum(r["pass"] for r in rs if r["family"] == fam) / max(1, sum(r["family"] == fam for r in rs)))}
                           for fam in sorted({r["family"] for r in rs})}}
        if a.draws > 1:
            s["curves"] = pass_curves(rs, a.draws)
        summary[name] = s
        jj = s["judge"]
        lines.append(f"| {name} | {s['pass_rate']:.0%} | {s['judge_pass_rate']:.0%} | {s['clean_rate']:.0%} | {jj['supported']:.2f} | {jj['complete']:.2f} | {jj['on_request']:.2f} | {s['mean_words']:.1f} | {s['median_words']:.0f} |")
    if a.draws > 1:
        lines += ["", f"pass@k / pass^k over {a.draws} draws at temperature {temperature}:", "| arm | " + " | ".join(f"k={k}" for k in range(1, a.draws + 1)) + " |",
                  "|---|" + "---|" * a.draws]
        for name, _, _ in arms:
            cv = summary[name]["curves"]
            lines.append(f"| {name} | " + " | ".join(f"{at:.0%} / {pw:.0%}" for at, pw in zip(cv["pass_at"], cv["pass_pow"])) + " |")
    table = "\n".join(lines)
    (out / "summary.json").write_text(json.dumps({"split": a.split, "draws": a.draws, "temperature": temperature, "effort": a.effort, "no_system_prompt": a.no_system_prompt, "pack_dir": str(a.pack_dir), "n_fixtures": len(fixtures), "arms": {n: {"model": m, "skill": s} for n, m, s in arms}, "summary": summary}, indent=2) + "\n")
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
