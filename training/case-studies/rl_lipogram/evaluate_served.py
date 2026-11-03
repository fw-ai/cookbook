"""Four-arm serverless evaluation. Running this script makes paid API calls.

Primary metric: complete, e-free AND useful responses / all requested draws.
Judge failures remain unknown and are reported separately; they cannot improve
that conservative success rate. Bootstrap intervals resample whole prompts.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from build_prompts import SYSTEM_PROMPT
from lipogram_data import ground_truth, user_messages
from lipogram_rewards import chat, llm_judge, lipogram_reward

BASE = "accounts/fireworks/models/glm-5p3-flash"
HERE = Path(__file__).resolve().parent


def summarize(rows):
    n = len(rows)
    known = [r for r in rows if r["completeness"] is not None]
    mean = lambda key, rs: sum(r[key] for r in rs) / len(rs) if rs else None
    return {"n": n, "e_free": mean("e_free", rows), "success": mean("success", rows),
            "answered": mean("answered", known), "completeness": mean("completeness", known),
            "judge_failures": n - len(known), "truncated": sum(r["truncated"] for r in rows),
            "e_rate": mean("e_rate", rows), "words": mean("words", rows),
            "reasoning_tokens": mean("reasoning_tokens", [r for r in rows if r.get("reasoning_tokens") is not None]),
            "generation_seconds": mean("generation_seconds", [r for r in rows if r.get("generation_seconds") is not None])}


def paired_interval(rows, metric="success"):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if not r["greedy"]:
            by[r["prompt_index"]][r["arm"]].append(r[metric])
    diffs = [sum(v["adapter"]) / len(v["adapter"]) - sum(v["base+prompt"]) / len(v["base+prompt"])
             for v in by.values() if v["adapter"] and v["base+prompt"]]
    if not diffs:
        return None
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(4000))
    return {"delta": sum(diffs) / len(diffs), "lo": boots[100], "hi": boots[3899], "prompt_groups": len(diffs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", type=Path, default=HERE / "data/lipogram_test.jsonl")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--effort", choices=["low", "max"], default="low")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N prompts (0 = all); for diagnostics")
    ap.add_argument("--no-mask-arm", action="store_true", help="skip the constrained-decoding (logit-mask) baseline arm")
    ap.add_argument("--out", type=Path, required=True, help="New directory, e.g. results/sft-low")
    a = ap.parse_args()
    if min(a.draws, a.max_tokens, a.concurrency) < 1:
        ap.error("draws, max-tokens and concurrency must be positive")
    rows = [json.loads(line) for line in a.tasks.read_text().splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        ap.error("Empty task set")
    arms = {"base+prompt": (BASE, True), "base": (BASE, False),
            "adapter+prompt": (a.adapter, True), "adapter": (a.adapter, False)}
    # Constrained-decoding floor: the base model with every token containing an e masked out. Guaranteed
    # e-free by construction; the question it answers is how much coherence the mask costs, which is
    # what an adapter has to beat. Needs the cached tokenizer; skipped (with a note) if it is absent.
    e_mask = {}
    if not a.no_mask_arm:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3-Flash", local_files_only=True)
            special = set(tok.all_special_ids)
            e_mask = {str(i): -100 for i in range(len(tok)) if i not in special and ("e" in tok.decode([i]) or "E" in tok.decode([i]))}
            arms["base+mask"] = (BASE, False)
            print(f"base+mask arm: {len(e_mask)} tokens masked")
        except Exception as exc:  # noqa: BLE001
            print(f"base+mask arm skipped: {exc}")
    jobs = [(arm, i, d) for arm in arms for i in range(len(rows)) for d in range(a.draws + 1)]
    a.out.mkdir(parents=True, exist_ok=False)
    meta = {"adapter": a.adapter, "base": BASE, "draws": a.draws, "effort": a.effort,
            "max_tokens": a.max_tokens, "sample_temperature": 0.7, "system_prompt": SYSTEM_PROMPT,
            "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(), "created_at": time.time()}
    (a.out / "manifest.json").write_text(json.dumps(meta, indent=2))

    def one(job):
        arm, i, draw = job
        model, prompted = arms[arm]
        row = rows[i]
        gt = ground_truth(row)
        msgs = ([{"role": "system", "content": SYSTEM_PROMPT}] if prompted else []) + user_messages(row)
        with httpx.Client() as client:
            started = time.perf_counter()
            body = {"model": model, "messages": msgs, "max_tokens": a.max_tokens,
                    "temperature": 0.0 if draw == 0 else 0.7, "reasoning_effort": a.effort}
            if arm == "base+mask":
                body["logit_bias"] = e_mask
            result = chat(client, body)
            generation_seconds = time.perf_counter() - started
            choice = result["choices"][0]
            answer = (choice["message"].get("content") or "").strip()
            judge = llm_judge(client, gt["question"], answer, gt.get("facts", ""))
        reward, free, answered, rate = lipogram_reward(answer, judge, phase="strict")
        complete = choice.get("finish_reason") == "stop"
        return {"arm": arm, "prompt_index": i, "source_id": gt["source_id"], "draw": draw,
                "greedy": draw == 0, "family": gt["family"], "band": gt.get("band", "unspecified"),
                "e_free": free, "answered": answered if judge is not None else None,
                "success": bool(free and answered and complete),
                "completeness": (judge or {}).get("answers"), "e_rate": rate, "reward": reward,
                "finish_reason": choice.get("finish_reason"), "truncated": choice.get("finish_reason") == "length",
                "words": len(answer.split()), "response": answer,
                "reasoning_tokens": ((result.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "usage": result.get("usage"), "generation_seconds": generation_seconds}

    print(f"{len(jobs)} generation calls, plus judging")
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "samples.jsonl").open("w") as f:
        for record in ex.map(one, jobs):
            f.write(json.dumps(record) + "\n")
            f.flush()
    recs = [json.loads(line) for line in (a.out / "samples.jsonl").read_text().splitlines()]
    summary = {**meta, "arms": {}}
    lines = ["| Arm | Greedy success | Sampled success | Sampled e-free | Judge failures |",
             "|---|---:|---:|---:|---:|"]
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        greedy = summarize([r for r in rs if r["greedy"]])
        sampled = summarize([r for r in rs if not r["greedy"]])
        summary["arms"][arm] = {"greedy": greedy, "sampled": sampled,
            "by_family": {f: summarize([r for r in rs if r["family"] == f]) for f in sorted({r["family"] for r in rs})},
            "by_band": {b: summarize([r for r in rs if r["band"] == b]) for b in sorted({r["band"] for r in rs})}}
        lines.append(f"| {arm} | {greedy['success']:.1%} | {sampled['success']:.1%} | {sampled['e_free']:.1%} | {sampled['judge_failures']} |")
    ci = paired_interval(recs)
    summary["paired_adapter_minus_base_prompt_success"] = ci
    lines.append(f"\nJoint success, adapter without instruction minus prompted base: {ci['delta']:+.1%} "
                 f"(95% paired bootstrap interval {ci['lo']:+.1%} to {ci['hi']:+.1%}).")
    lines.append("Success requires a complete, e-free, useful answer. Unknown judge verdicts do not count as successes.")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
