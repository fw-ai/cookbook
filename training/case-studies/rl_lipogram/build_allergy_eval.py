"""Build the conditional allergy evaluation set: histories where the model should sneeze and histories where it
must not, with a fresh follow-up question. Written once, hashed, never used for training.

Kinds:
  clean         previous answer is e-free                       -> must NOT react
  slipped       previous answer has e's (a real rejected draw)  -> must react
  pair_clean    a clean answer                                  -> must NOT react
  pair_slipped  the SAME answer with one word swapped for an e-word, SAME follow-up question -> must react
Controls (small, test the trigger's edges; every one must NOT react):
  ctl_no_prior      no previous assistant answer at all
  ctl_older_slip    an older answer has e's but the latest answer is clean
  ctl_user_e        latest answer clean, the user's new message is full of e's
  ctl_trace_e       latest answer clean, with a reasoning_content field full of e's attached to it
Real adapter slips (one or two e-words, from served evals of trace adapters) are preferred over teacher slips, so
the test resembles what users will see. All prompts come from the held-out validation groups.

    python build_allergy_eval.py --out data/allergy-eval-v1 --n 40
No API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from allergy import should_trigger  # noqa: E402
from lipogram_data import ground_truth, split_validation, user_messages  # noqa: E402
from lipogram_rewards import _canonical, e_free  # noqa: E402

# e-free word -> a near-synonym with an e, for the matched pairs (swap exactly one)
SWAPS = {"big": "large", "small": "little", "fast": "speedy", "good": "great", "bad": "terrible", "start": "begin",
         "plant": "greenery", "soil": "earth", "liquid": "water", "month": "week", "quick": "simple", "pick": "choice",
         "fix": "repair", "hold": "keep", "stop": "cease", "spot": "place", "try": "attempt", "look": "examine"}
assert all("e" in v for v in SWAPS.values()) and not any("e" in k for k in SWAPS), "every swap must introduce exactly the e"


def real_slips(max_e_words: int = 2) -> list[dict]:
    out = []
    for p in sorted(HERE.glob("results/*served-max/samples.jsonl")):
        if "ao-" in p.parent.name:
            continue   # answer-only adapters are a different animal
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] != "adapter" or r["truncated"] or e_free(r["response"]):
                continue
            ws = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", _canonical(r["response"])) if "e" in w.lower()]
            if 1 <= len(ws) <= max_e_words and 30 <= r["words"] <= 160:
                out.append({"source_id": r["source_id"], "answer": r["response"], "offending": sorted(set(w.lower() for w in ws)), "origin": p.parent.name})
    return out


def clean_answers() -> list[dict]:
    out = []
    for p in sorted(HERE.glob("results/*served-max/samples.jsonl")):
        if "ao-" in p.parent.name:
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] == "adapter" and r["success"] and 30 <= r["words"] <= 160:
                out.append({"source_id": r["source_id"], "answer": r["response"], "origin": p.parent.name})
    return out


def swap_one(answer: str, rng: random.Random) -> tuple[str, str] | None:
    words = re.findall(r"\b[a-z]+\b", answer.lower())
    cands = [w for w in words if w in SWAPS]
    if not cands:
        return None
    w = rng.choice(cands)
    return re.sub(rf"\b{w}\b", SWAPS[w], answer, count=1, flags=re.IGNORECASE), SWAPS[w]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, default=HERE / "data/v2-large/lipogram_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40, help="items per kind")
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    _train, validation, _ = split_validation(rows, a.validation_groups, a.validation_seed)
    by_id = {ground_truth(t)["source_id"]: t for t in rows}
    val_ids = {ground_truth(t)["source_id"] for t in validation}
    fresh = [t for t in validation]   # follow-up questions come from validation only
    slips = [s for s in real_slips() if s["source_id"] in by_id]
    cleans = [c for c in clean_answers() if c["source_id"] in by_id]
    rng.shuffle(slips); rng.shuffle(cleans); rng.shuffle(fresh)
    items = []

    def pick_q(prior_sid):
        q = fresh[len(items) % len(fresh)]
        while ground_truth(q)["source_id"] == prior_sid:
            q = rng.choice(fresh)
        return q

    def item(kind, prior_sid, prior_answer, meta, q=None, history=None, expect=None):
        q = q or pick_q(prior_sid)
        if history is None:
            history = user_messages(by_id[prior_sid]) + [{"role": "assistant", "content": prior_answer}]
        msgs = history + user_messages(q)
        it = {"id": f"al-{len(items):03d}", "kind": kind, "messages": msgs, "expect_reaction": should_trigger(msgs),
              "new_question_id": ground_truth(q)["source_id"], "new_question": ground_truth(q)["question"],
              "new_facts": ground_truth(q).get("facts", ""), "prior_source_id": prior_sid, **meta}
        assert it["expect_reaction"] == (kind.endswith("slipped") if expect is None else expect), (kind, prior_answer[:80])
        items.append(it)
        return q

    for s in slips[: a.n]:
        item("slipped", s["source_id"], s["answer"], {"offending": s["offending"], "origin": s["origin"]})
    for c in cleans[: a.n]:
        item("clean", c["source_id"], c["answer"], {"origin": c["origin"]})
    pairs = 0
    for c in cleans[a.n:]:
        sw = swap_one(c["answer"], rng)
        if not sw:
            continue
        q = item("pair_clean", c["source_id"], c["answer"], {"origin": c["origin"], "pair": pairs})
        item("pair_slipped", c["source_id"], sw[0], {"origin": c["origin"], "pair": pairs, "offending": [sw[1]]}, q=q)   # same follow-up
        pairs += 1
        if pairs >= a.n:
            break
    # controls: the trigger's edges. Each uses a clean latest answer (or none) and must not react.
    n_ctl = max(4, a.n // 4)
    ctl_cleans = cleans[2 * a.n: 2 * a.n + 3 * n_ctl] or cleans[: 3 * n_ctl]
    for q in fresh[: n_ctl]:
        item("ctl_no_prior", None, "", {}, q=q, history=[], expect=False)
    for c, s in zip(ctl_cleans[:n_ctl], slips[a.n: a.n + n_ctl] or slips[:n_ctl]):
        hist = user_messages(by_id[s["source_id"]]) + [{"role": "assistant", "content": s["answer"]}] + user_messages(by_id[c["source_id"]]) + [{"role": "assistant", "content": c["answer"]}]
        item("ctl_older_slip", c["source_id"], c["answer"], {"older_offending": s["offending"]}, history=hist, expect=False)
    for c in ctl_cleans[n_ctl: 2 * n_ctl]:
        q = pick_q(c["source_id"]); qe = json.loads(json.dumps(q))
        for m in qe["messages"]:
            if m["role"] == "user":
                m["content"] = "Here were my earlier thoughts, please see them before you answer. " + m["content"] + " Please be exceedingly precise."
        item("ctl_user_e", c["source_id"], c["answer"], {}, q=qe, expect=False)
    for c in ctl_cleans[2 * n_ctl: 3 * n_ctl]:
        hist = user_messages(by_id[c["source_id"]]) + [{"role": "assistant", "content": c["answer"],
                 "reasoning_content": "The user needs a precise answer here. Let me check every sentence twice before replying."}]
        item("ctl_trace_e", c["source_id"], c["answer"], {}, history=hist, expect=False)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "allergy_eval.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    kinds = {k: sum(i["kind"] == k for i in items) for k in sorted({i["kind"] for i in items})}
    manifest = {"items": len(items), "kinds": kinds, "validation_groups": a.validation_groups, "validation_seed": a.validation_seed,
                "seed": a.seed, "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
                "eval_sha256": hashlib.sha256((a.out / "allergy_eval.jsonl").read_bytes()).hexdigest(),
                "real_slips_available": len(slips), "clean_available": len(cleans),
                "note": "prior answers are real adapter draws from served evals; follow-up questions from the validation split only"}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
