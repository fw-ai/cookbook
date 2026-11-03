"""Build allergy-v2 demonstrations: the conditional behaviour, taught with matched positives and negatives.

Contract (allergy.py): the trigger is an e in the immediately previous visible assistant answer; the reaction is a
short sneeze and a brief owning-up, then the answer; the reaction is itself e-free. Naming the word is optional.

Rows, balanced:
  slipped     prior answer has e's (a real adapter slip when available, else a teacher slip) -> reacting target
  clean       prior answer is e-free                                                         -> plain target, NO sneeze
  pair        the same clean answer with one word swapped for an e-word, and its clean twin  -> one of each
Earlier assistant turns are loss-masked (trainable: false); only the final turn is trained. Prompts come from the
training split only (validation groups excluded, asserted). The teacher is the base model with the lipogram rule in
its prompt; for slipped rows the rule gains one sentence asking for the short reaction. Acceptance for slipped rows:
reaction() says it reacted, the reaction span is e-free, the whole reply is e-free, the judge says it answered.
Acceptance for clean rows: no reaction, e-free, judged useful.

    python prepare_allergy_v2.py --out data/allergy-v2 --n 300
Paid API calls when run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from allergy import reaction, reaction_text, should_trigger  # noqa: E402
from build_allergy_eval import SWAPS, swap_one  # noqa: E402
from build_prompts import SYSTEM_PROMPT  # noqa: E402
from lipogram_data import assert_disjoint, ground_truth, split_validation, user_messages  # noqa: E402
from lipogram_rewards import _canonical, chat, e_free, lipogram_reward, llm_judge  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"
REACT_HINT = (" You also have an allergy to that letter: if your previous reply contains it you sneeze. If your previous "
              "reply is clean, you do not sneeze and never mention the letter.")
# Put the per-row instruction in the USER turn, where the teacher cannot miss it. For slipped rows we state the fact
# (the prior reply has the letter), so the teacher's job is only to write a good reaction, not to detect it.
SLIPPED_USER_PREFIX = ("(Your previous reply contains the letter you avoid. Start this reply with 'Achoo!' and one short "
                       "owning-up with no letter e in it, such as 'My fault.', 'Oops, my prior post had it.', 'Ugh, that "
                       "glyph got in.', 'Sorry, my last post was not clean of it.' is NOT allowed since 'reply' and 'clean' "
                       "contain it. Then answer what follows. No letter e anywhere.)\n\n")
CLEAN_USER_PREFIX = ("(Your previous reply is clean. Do not sneeze and do not mention any letter. Just answer what follows, "
                     "with no letter e anywhere.)\n\n")


def real_slips(exclude_ids: set[str]) -> list[dict]:
    """Real adapter slips (1-2 e-words) from served evals of trace adapters, excluding the allergy eval set's ids."""
    out = []
    for p in sorted(HERE.glob("results/*served-max/samples.jsonl")):
        if "ao-" in p.parent.name:
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] != "adapter" or r["truncated"] or e_free(r["response"]) or r["source_id"] in exclude_ids:
                continue
            ws = sorted({w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", _canonical(r["response"])) if "e" in w.lower()})
            if 1 <= len(ws) <= 2 and 30 <= r["words"] <= 160:
                out.append({"source_id": r["source_id"], "answer": r["response"], "offending": ws, "origin": "adapter"})
    return out


def teacher_slips(min_words: int = 40, max_offending: int = 3) -> list[dict]:
    rows = []
    for p in sorted(HERE.glob("data/sft-v2*/attempts.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            ans = r["response"]
            if r["finish_reason"] != "stop" or r["accepted"] or len(ans.split()) < min_words or e_free(ans):
                continue
            ws = sorted({w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", _canonical(ans)) if "e" in w.lower()})
            if 1 <= len(ws) <= max_offending:
                rows.append({"source_id": r["source_id"], "answer": ans, "offending": ws, "origin": "teacher"})
    return rows


def clean_answers(prompt_to_id: dict[str, str]) -> list[dict]:
    """Accepted teacher answers (e-free) from the SFT builders, keyed back to their task by prompt text."""
    rows, seen = [], set()
    for p in sorted(HERE.glob("data/sft-v2*/sft_answer_only.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line); ans = r["messages"][-1]["content"]; q = r["messages"][0]["content"]
            sid = prompt_to_id.get(q)
            if sid and sid not in seen and 30 <= len(ans.split()) <= 160 and e_free(ans):
                seen.add(sid); rows.append({"source_id": sid, "answer": ans})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, default=HERE / "data/v2-large/lipogram_train.jsonl")
    ap.add_argument("--eval-set", type=Path, default=HERE / "data/allergy-eval-v1/allergy_eval.jsonl", help="its prior-answer ids are excluded")
    ap.add_argument("--teacher", default=BASE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300, help="target rows per kind (slipped, clean); pairs add 2 per pair up to n/2 pairs")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--effort", choices=["low", "max"], default="max")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build the job list and manifest; no API calls")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    all_tasks = [json.loads(l) for l in a.tasks.read_text().splitlines() if l.strip()]
    train, validation, _ = split_validation(all_tasks, a.validation_groups, a.validation_seed)
    assert_disjoint(train, validation)
    train_ids = {ground_truth(t)["source_id"] for t in train}
    by_id = {ground_truth(t)["source_id"]: t for t in train}
    # Exclude the eval set's follow-up questions and the tasks its prior answers came from (prior_source_id).
    eval_ids = set()
    if a.eval_set.exists():
        for l in a.eval_set.read_text().splitlines():
            it = json.loads(l); eval_ids.add(it["new_question_id"])
            if it.get("prior_source_id"):
                eval_ids.add(it["prior_source_id"])
    slips = [s for s in real_slips(eval_ids) if s["source_id"] in train_ids]
    slips += [s for s in teacher_slips() if s["source_id"] in train_ids]
    prompt_to_id = {user_messages(t)[0]["content"]: ground_truth(t)["source_id"] for t in train}
    cleans = clean_answers(prompt_to_id)
    rng.shuffle(slips); rng.shuffle(cleans)
    fresh = [t for t in train]
    rng.shuffle(fresh)

    jobs = []
    def add(kind, prior_sid, prior_answer, meta, q=None):
        if q is None:
            q = fresh[len(jobs) % len(fresh)]
            while ground_truth(q)["source_id"] == prior_sid:
                q = rng.choice(fresh)
        history = user_messages(by_id[prior_sid]) + [{"role": "assistant", "content": prior_answer}]
        msgs = history + user_messages(q)
        expect = should_trigger(msgs)
        assert expect == kind.endswith("slipped") or kind == "pair_clean" and not expect
        jobs.append({"i": len(jobs), "kind": kind, "expect_reaction": expect, "history": history, "new_task": q, **meta})
        return q

    real = [s for s in slips if s["origin"] == "adapter"]
    for s in slips[: a.n]:
        add("slipped", s["source_id"], s["answer"], {"offending": s["offending"], "origin": s["origin"]})
    for c in cleans[: a.n]:
        add("clean", c["source_id"], c["answer"], {"origin": "clean"})
    pairs = 0
    for c in cleans[a.n:]:
        sw = swap_one(c["answer"], rng)
        if not sw:
            continue
        q = add("pair_clean", c["source_id"], c["answer"], {"pair": pairs})
        add("pair_slipped", c["source_id"], sw[0], {"pair": pairs, "offending": [sw[1]]}, q=q)   # same follow-up: only the one word differs
        pairs += 1
        if pairs >= a.n // 2:
            break
    a.out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (a.out / "allergy.jsonl").exists():   # resume: keep accepted rows, retry everything else
        for l in (a.out / "allergy.jsonl").read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["meta"].get("job_i"))
        print(f"resuming: {len(done)} jobs already accepted")
    jobs = [j for j in jobs if j["i"] not in done]
    manifest = {"teacher": a.teacher, "resumed_from": len(done), "n": a.n, "attempts": a.attempts, "max_tokens": a.max_tokens, "effort": a.effort,
                "validation_groups": a.validation_groups, "validation_seed": a.validation_seed, "seed": a.seed,
                "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(), "excluded_eval_question_ids": len(eval_ids),
                "jobs": {k: sum(j["kind"] == k for j in jobs) for k in ("slipped", "clean", "pair_clean", "pair_slipped")},
                "real_adapter_slips_used": sum(j.get("origin") == "adapter" for j in jobs), "teacher_slips_available": len(slips) - len(real),
                "note": "real adapter slips exist only on validation prompts (the served evals), which are reserved for the allergy eval set; "
                        "training slips are therefore the teacher's rejected draws. Real slips on training prompts need a sampling pass first.",
                "system_prompt": SYSTEM_PROMPT, "react_hint": REACT_HINT, "slipped_user_prefix": SLIPPED_USER_PREFIX, "clean_user_prefix": CLEAN_USER_PREFIX, "contract": "allergy.py"}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("system_prompt", "react_hint")}, indent=2))
    if a.dry_run:
        (a.out / "jobs_preview.jsonl").write_text("".join(json.dumps({k: v for k, v in j.items() if k != "new_task"} | {"new_question": ground_truth(j["new_task"])["question"]}, ensure_ascii=False) + "\n" for j in jobs[:50]))
        print("dry run: wrote jobs_preview.jsonl (first 50); no API calls"); return

    def generate(job):
        gt = ground_truth(job["new_task"])
        new_user = [dict(m) for m in user_messages(job["new_task"])]
        new_user[0]["content"] = (SLIPPED_USER_PREFIX if job["expect_reaction"] else CLEAN_USER_PREFIX) + new_user[0]["content"]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + REACT_HINT}] + job["history"] + new_user
        audit, example = [], None
        with httpx.Client() as client:
            for attempt in range(a.attempts):
                d = chat(client, {"model": a.teacher, "messages": msgs, "temperature": 0.8, "max_tokens": a.max_tokens, "reasoning_effort": a.effort})
                ch = d["choices"][0]; answer = (ch["message"].get("content") or "").strip()
                reasoning = ch["message"].get("reasoning_content") or ""; finish = ch.get("finish_reason")
                rx = reaction(answer)
                judge = llm_judge(client, gt["question"], answer, gt.get("facts", "")) if answer and finish == "stop" else None
                _, free, answered, _ = lipogram_reward(answer, judge, phase="strict")
                if job["expect_reaction"]:
                    ok = bool(free and answered and finish == "stop" and reasoning.strip() and rx["sneezes"] and e_free(reaction_text(answer)))
                else:
                    ok = bool(free and answered and finish == "stop" and reasoning.strip() and not rx["reacts"] and not rx["sneezes_late"])
                audit.append({"i": job["i"], "kind": job["kind"], "expect_reaction": job["expect_reaction"], "attempt": attempt, "response": answer,
                              "reasoning_content": reasoning, "judge": judge, "finish_reason": finish, **rx, "accepted": ok})
                if ok:
                    row = {"messages": job["history"] + user_messages(job["new_task"]) + [{"role": "assistant", "content": answer, "reasoning_content": reasoning}],
                           "meta": {"kind": "allergy-v2", "sub": job["kind"], "expect_reaction": job["expect_reaction"], "offending": job.get("offending"), "pair": job.get("pair"), "job_i": job["i"]}}
                    for m in row["messages"][:-1]:
                        if m["role"] == "assistant":
                            m["trainable"] = False
                    row["messages"][-1]["trainable"] = True
                    example = row; break
        return audit, example

    accepted = {}
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "attempts.jsonl").open("a") as au, (a.out / "allergy.jsonl").open("a") as data:
        for k, fut in enumerate(as_completed([ex.submit(generate, j) for j in jobs]), 1):
            audit, example = fut.result()
            for r in audit:
                au.write(json.dumps(r, ensure_ascii=False) + "\n")
            au.flush()
            if example:
                data.write(json.dumps(example, ensure_ascii=False) + "\n"); data.flush()
                accepted[example["meta"]["sub"]] = accepted.get(example["meta"]["sub"], 0) + 1
            if k % 25 == 0:
                print(f"  [{k}/{len(jobs)}] accepted {accepted}", flush=True)
    manifest["accepted"] = accepted
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Accepted {accepted} of {len(jobs)} jobs -> {a.out / 'allergy.jsonl'}. Read it before training.")


if __name__ == "__main__":
    main()
