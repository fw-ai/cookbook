"""The decode-time mask grid: does a trained adapter write better than the base under a hard "no e" constraint?

Arms are explicit (model, prompt?, mask?). The mask is logit_bias=-100 on every token whose text contains an e or
an e look-alike (the same set the checker counts), applied to the whole generation: the API has no "visible answer
only" mask, so at effort low (little or no reasoning) this is close to the intended contract and at higher efforts it
constrains the reasoning too. Reported per arm, all draws in the denominator:
  forbidden     any forbidden character in the visible answer (0 by construction for masked arms, verified)
  finished      finish_reason == stop
  nonempty      at least MIN_WORDS words
  repetitive    the answer repeats a 4-gram more than REPEAT_LIMIT times (the masked loop failure)
  useful        finished, nonempty, not repetitive, judged complete (>= 0.5)
  success       forbidden-free AND useful
  reasoning     actual reasoning tokens (never assumed from the effort setting)

    python evaluate_masked.py --effort low --draws 2 --max-tokens 4096 --out results/mask-grid-low \
        --arm base+prompt=BASE:prompt --arm base+prompt+mask=BASE:prompt:mask \
        --arm ao-v5=accounts/pyroworks/models/glm53-lipogram-ao-v5 --arm ao-v5+mask=accounts/pyroworks/models/glm53-lipogram-ao-v5:mask \
        --arm trace-v1=accounts/pyroworks/models/glm53-lipogram-trace-allergy-v1 --arm trace-v1+mask=accounts/pyroworks/models/glm53-lipogram-trace-allergy-v1:mask
Paid API calls.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_prompts import SYSTEM_PROMPT  # noqa: E402
from lipogram_data import ground_truth, user_messages  # noqa: E402
from lipogram_rewards import _canonical, chat, e_free, llm_judge, non_latin_letters  # noqa: E402

sys.path.insert(0, str(HERE.parent / "common"))
from figures import wilson_ci  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"
MIN_WORDS = 15
REPEAT_LIMIT = 3


def build_mask() -> dict[str, int]:
    """Every tokenizer id whose decoded text canonicalises to something containing an e (ASCII or look-alike)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3-Flash", local_files_only=True)
    special = set(tok.all_special_ids)
    mask = {}
    for i in range(len(tok)):
        if i in special:
            continue
        text = tok.decode([i])
        # e, any e look-alike, or any non-Latin letter: the three things the checker counts as a slip
        if "e" in _canonical(text).lower() or "e" in unicodedata.normalize("NFKD", text).lower() or non_latin_letters(text):
            mask[str(i)] = -100
    return mask


def repetitive(text: str) -> bool:
    words = text.lower().split()
    grams = collections.Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return bool(grams) and grams.most_common(1)[0][1] > REPEAT_LIMIT


def parse_arm(spec: str) -> tuple[str, str, bool, bool]:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    model = BASE if parts[0] == "BASE" else parts[0]
    return name, model, "prompt" in parts[1:], "mask" in parts[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=model[:prompt][:mask]; model BASE = the base model")
    ap.add_argument("--tasks", type=Path, default=HERE / "data/sft-v2/validation.jsonl")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--effort", choices=["low", "medium", "high", "max"], default="low")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = [parse_arm(s) for s in a.arm]
    rows = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    mask = build_mask() if any(m for *_, m in arms) else {}
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "manifest.json").write_text(json.dumps({"arms": {n: {"model": m, "prompt": p, "mask": k} for n, m, p, k in arms},
        "effort": a.effort, "draws": a.draws, "max_tokens": a.max_tokens, "temperature": a.temperature, "masked_tokens": len(mask),
        "mask_note": "logit_bias -100 on every token containing an e or e look-alike; applies to reasoning too (API has no answer-only mask)",
        "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(), "n_prompts": len(rows)}, indent=2))
    jobs = [(n, m, p, k, i, d) for n, m, p, k in arms for i in range(len(rows)) for d in range(a.draws)]
    print(f"{len(arms)} arms x {len(rows)} prompts x {a.draws} draws at effort {a.effort}; mask covers {len(mask)} tokens")

    def one(job):
        name, model, prompted, masked, i, d = job
        gt = ground_truth(rows[i])
        msgs = ([{"role": "system", "content": SYSTEM_PROMPT}] if prompted else []) + user_messages(rows[i])
        body = {"model": model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature, "reasoning_effort": a.effort}
        if masked:
            body["logit_bias"] = mask
        with httpx.Client() as c:
            r = chat(c, body)
            ch = r["choices"][0]; ans = (ch["message"].get("content") or "").strip()
            finished = ch.get("finish_reason") == "stop"
            rep = repetitive(ans); nonempty = len(ans.split()) >= MIN_WORDS
            judge = llm_judge(c, gt["question"], ans, gt.get("facts", "")) if (finished and nonempty and not rep) else None
        useful = bool(finished and nonempty and not rep and judge and judge.get("answers", 0) >= 0.5)
        free = e_free(ans)
        return {"arm": name, "prompt_index": i, "source_id": gt["source_id"], "family": gt["family"], "draw": d, "response": ans,
                "forbidden": not free, "finished": finished, "nonempty": nonempty, "repetitive": rep, "judge": judge,
                "useful": useful, "success": bool(free and useful), "words": len(ans.split()),
                "reasoning_tokens": ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}

    out_rows = []; t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "samples.jsonl").open("w") as f:
        for k, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            row = fut.result(); out_rows.append(row); f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if k % 50 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)

    def rate(sub, key):
        n = len(sub); c = sum(bool(r[key]) for r in sub); p, lo, hi = wilson_ci(c, n) if n else (0, 0, 0)
        return {"k": c, "n": n, "rate": round(p, 3), "lo": round(lo, 3), "hi": round(hi, 3)}

    summary = {}
    for n, *_ in arms:
        sub = [r for r in out_rows if r["arm"] == n]
        summary[n] = {k: rate(sub, k) for k in ("forbidden", "finished", "nonempty", "repetitive", "useful", "success")}
        summary[n]["reasoning"] = {"draws_with_tokens": rate(sub, "reasoning_tokens")["rate"],
                                   "mean_when_known": round(sum(r["reasoning_tokens"] for r in sub if r["reasoning_tokens"] is not None) / max(1, sum(r["reasoning_tokens"] is not None for r in sub)))}
        summary[n]["mean_words"] = round(sum(r["words"] for r in sub) / max(1, len(sub)), 1)
    (a.out / "summary.json").write_text(json.dumps({"effort": a.effort, "max_tokens": a.max_tokens, "draws": a.draws, "summary": summary}, indent=2))
    md = [f"# Mask grid at effort {a.effort}, {len(rows)} prompts x {a.draws} draws, {a.max_tokens} tokens", "",
          "| Arm | Forbidden char | Finished | Non-empty | Repetitive | Useful | **Success** | Words | Reasoning tok (share>0) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, *_ in arms:
        s = summary[n]
        md.append(f"| {n} | {100*s['forbidden']['rate']:.0f}% | {100*s['finished']['rate']:.0f}% | {100*s['nonempty']['rate']:.0f}% | {100*s['repetitive']['rate']:.0f}% | "
                  f"{100*s['useful']['rate']:.0f}% | **{100*s['success']['rate']:.0f}%** [{100*s['success']['lo']:.0f}, {100*s['success']['hi']:.0f}] | {s['mean_words']} | {s['reasoning']['mean_when_known']:,} ({100*s['reasoning']['draws_with_tokens']:.0f}%) |")
    md += ["", "Success = no forbidden character AND finished, non-empty, non-repetitive, judged complete. Masked arms should show 0% forbidden;",
           "if not, the mask has a hole. The comparison that isolates the trained weights is adapter+mask vs base+prompt+mask on Useful."]
    (a.out / "summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md)); print("wrote", a.out)


if __name__ == "__main__":
    main()
