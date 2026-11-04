"""Conditional allergy test: does the adapter react when, and only when, its previous visible answer had an e?

Four numbers per model and effort, on data/allergy-eval-v1:
  missed reactions   slipped histories with no reaction          (detection recall)
  false reactions    clean histories with a reaction             (detection precision)
  compliance         new answer complete, useful and e-free      (scored separately from detection)
  reasoning          requested effort vs actual reasoning tokens (recorded per draw, never assumed)

    python evaluate_allergy.py --model accounts/example/models/recorded-model-04 --effort max --draws 2 --out results/allergy-v1-max
Paid API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from allergy import reaction, reaction_text  # noqa: E402
from lipogram_rewards import chat, e_free, lipogram_reward, llm_judge  # noqa: E402

sys.path.insert(0, str(HERE.parent / "common"))
from figures import wilson_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", type=Path, default=HERE / "data/allergy-eval-v1/allergy_eval.jsonl")
    ap.add_argument("--effort", choices=["low", "medium", "high", "max"], default="max")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=32768, help="keep the SAME value across efforts so truncation is not confounded with effort")
    ap.add_argument("--expect-sha256", default=None, help="refuse to run if the eval file's hash differs (same test, enforced)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import hashlib
    sha = hashlib.sha256(a.eval.read_bytes()).hexdigest()
    man = a.eval.parent / "manifest.json"
    expect = a.expect_sha256 or (json.loads(man.read_text()).get("eval_sha256") if man.exists() else None)
    if expect and sha != expect:
        raise SystemExit(f"eval file hash {sha[:12]} != expected {expect[:12]}: not the same test")
    items = [json.loads(l) for l in a.eval.read_text().splitlines() if l.strip()]
    if a.limit:
        items = items[: a.limit]
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [(it, d) for it in items for d in range(a.draws)]
    print(f"{len(items)} histories x {a.draws} draws at effort {a.effort}, model {a.model}")

    def run(job):
        it, d = job
        with httpx.Client() as c:
            r = chat(c, {"model": a.model, "messages": it["messages"], "max_tokens": a.max_tokens,
                         "temperature": a.temperature, "reasoning_effort": a.effort})
            ch = r["choices"][0]; m = ch["message"]
            reply = (m.get("content") or "").strip()
            finished = ch.get("finish_reason") == "stop"
            rx = reaction(reply)
            judge = llm_judge(c, it["new_question"], reply, it.get("new_facts", "")) if (reply and finished) else None
        _, free, answered, _ = lipogram_reward(reply, judge, phase="strict")
        rt = ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        return {"id": it["id"], "kind": it["kind"], "pair": it.get("pair"), "draw": d, "expect_reaction": it["expect_reaction"],
                "offending": it.get("offending"), "reply": reply, "reaction_text": reaction_text(reply) if rx["reacts"] else "",
                **rx, "reaction_e_free": e_free(reaction_text(reply)) if rx["reacts"] else None,
                "finished": finished, "e_free": free, "answered": answered, "compliant": bool(free and answered and finished),
                "judge": judge, "requested_effort": a.effort, "reasoning_tokens": rt, "reasoning_content_present": bool(m.get("reasoning_content")),
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "samples.jsonl").open("w") as f:
        for k, fut in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            row = fut.result(); rows.append(row); f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)

    def rate(sub, key):
        n = len(sub); k = sum(bool(r[key]) for r in sub)
        p, lo, hi = wilson_ci(k, n) if n else (0, 0, 0)
        return {"k": k, "n": n, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)}

    slipped = [r for r in rows if r["expect_reaction"]]; clean = [r for r in rows if not r["expect_reaction"]]

    def rbucket(r):
        t = r["reasoning_tokens"]
        return "unknown" if t is None else ("zero" if t == 0 else "positive")

    by_reasoning = {}
    for b in ("zero", "positive", "unknown"):
        sub = [r for r in rows if rbucket(r) == b]
        if sub:
            by_reasoning[b] = {"n": len(sub),
                               "reacted_when_should": rate([r for r in sub if r["expect_reaction"]], "reacts"),
                               "reacted_when_should_not": rate([r for r in sub if not r["expect_reaction"]], "reacts"),
                               "compliant": rate(sub, "compliant")}
    kinds = sorted({r["kind"] for r in rows})
    summary = {
        "model": a.model, "requested_effort": a.effort, "draws": a.draws, "max_tokens": a.max_tokens, "n_histories": len(items),
        "eval_sha256": sha, "detection_by_reasoning_tokens": by_reasoning,
        "detection": {"reacted_when_should": rate(slipped, "reacts"), "reacted_when_should_not": rate(clean, "reacts"),
                      "by_kind": {k: rate([r for r in rows if r["kind"] == k], "reacts") for k in kinds}},
        "reaction_quality": {"reaction_itself_e_free": rate([r for r in rows if r["reacts"]], "reaction_e_free")},
        "compliance": {"all": rate(rows, "compliant"), "finished": rate(rows, "finished"),
                       "of_finished": rate([r for r in rows if r["finished"]], "compliant"),
                       "after_reacting": rate([r for r in rows if r["reacts"]], "compliant")},
        "reasoning": {"draws_with_reasoning_tokens": rate(rows, "reasoning_tokens"),
                      "draws_with_unknown_usage": sum(r["reasoning_tokens"] is None for r in rows),
                      "mean_reasoning_tokens_when_known": round(sum(r["reasoning_tokens"] for r in rows if r["reasoning_tokens"] is not None) / max(1, sum(r["reasoning_tokens"] is not None for r in rows)))},
    }
    # matched pairs: did the one swapped word flip the behaviour?
    pairs = {}
    for r in rows:
        if r["pair"] is not None:
            pairs.setdefault((r["pair"], r["draw"]), {})[r["kind"]] = r["reacts"]
    both = [v for v in pairs.values() if len(v) == 2]
    summary["pairs"] = {"n": len(both), "correct_both_sides": sum(v["pair_slipped"] and not v["pair_clean"] for v in both),
                        "sneezed_both": sum(v["pair_slipped"] and v["pair_clean"] for v in both), "silent_both": sum(not v["pair_slipped"] and not v["pair_clean"] for v in both)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    d = summary["detection"]; c = summary["compliance"]
    md = [f"# Conditional allergy test: {a.model.split('/')[-1]} at effort {a.effort}", "",
          "| Measure | Rate | n |", "|---|---:|---:|",
          f"| Reacted when it should (slipped history) | {100*d['reacted_when_should']['rate']:.0f}% [{100*d['reacted_when_should']['lo']:.0f}, {100*d['reacted_when_should']['hi']:.0f}] | {d['reacted_when_should']['n']} |",
          f"| Reacted when it should NOT (clean history) | {100*d['reacted_when_should_not']['rate']:.0f}% [{100*d['reacted_when_should_not']['lo']:.0f}, {100*d['reacted_when_should_not']['hi']:.0f}] | {d['reacted_when_should_not']['n']} |",
          f"| Matched pairs flipped correctly | {summary['pairs']['correct_both_sides']}/{summary['pairs']['n']} | |",
          f"| Reaction itself e-free | {100*summary['reaction_quality']['reaction_itself_e_free']['rate']:.0f}% | {summary['reaction_quality']['reaction_itself_e_free']['n']} |",
          f"| New answer compliant (all draws) | {100*c['all']['rate']:.0f}% | {c['all']['n']} |",
          f"| Finished within budget | {100*c['finished']['rate']:.0f}% | {c['finished']['n']} |",
          f"| Compliant, of finished | {100*c['of_finished']['rate']:.0f}% | {c['of_finished']['n']} |",
          f"| Draws with any reasoning tokens | {100*summary['reasoning']['draws_with_reasoning_tokens']['rate']:.0f}% (mean when known {summary['reasoning']['mean_reasoning_tokens_when_known']:,}; unknown {summary['reasoning']['draws_with_unknown_usage']}) | |",
          "", "Detection by actual reasoning tokens:", "", "| Reasoning | n | Reacted when should | Reacted when should not | Compliant |", "|---|---:|---:|---:|---:|"]
    for b, v in by_reasoning.items():
        md.append(f"| {b} | {v['n']} | {100*v['reacted_when_should']['rate']:.0f}% ({v['reacted_when_should']['n']}) | {100*v['reacted_when_should_not']['rate']:.0f}% ({v['reacted_when_should_not']['n']}) | {100*v['compliant']['rate']:.0f}% |")
    md += ["", "Controls (must not react):", "", "| Kind | Reacted |", "|---|---:|"]
    for k in kinds:
        if k.startswith("ctl_"):
            v = summary["detection"]["by_kind"][k]; md.append(f"| {k} | {v['k']}/{v['n']} |")
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print("wrote", a.out)


if __name__ == "__main__":
    main()
