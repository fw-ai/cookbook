"""Build "e allergy" demonstrations: the model notices an e that slipped in its PREVIOUS turn, reacts in
character, and answers the new question cleanly.

Why this is safe to train: the slipped turn is context only (loss masked); the training target is the
clean reacting turn. The model learns to read its history for the letter and apologise, never that
slipping is rewarded. Recognising an e in text it reads is easy at any effort; that is why this
transfers where avoiding e's while writing does not.

Data: a slipped answer from data/sft-v2*/attempts.jsonl (we know its offending words), followed by a
fresh user prompt, followed by a teacher-written reaction+answer that must be e-free and judged useful.
Escalation comes from history: with two prior slipped turns the reaction is the exasperated version.

    python prepare_allergy.py --limit 300 --out data/allergy-v1

Writes allergy.jsonl (multi-turn rows, trainable flag on the last turn only), attempts.jsonl, manifest.json.
Running main makes paid API calls.
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
from build_prompts import SYSTEM_PROMPT  # noqa: E402
from lipogram_data import assert_disjoint, ground_truth, split_validation, user_messages  # noqa: E402
from lipogram_rewards import _canonical, chat, e_free, llm_judge, lipogram_reward  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"

REACTION_SUFFIX = {
    1: (" You have an allergy to that letter: it makes you sneeze. Look at your previous reply. If it contains the "
        "letter, open your new reply by sneezing ('Achoo!') and owning up to the exact word that slipped, in a "
        "light, funny way, and if a substitute word would avoid it next time, say which. Then answer the new "
        "question. The whole reply, reaction included, must contain no letter e. One or two sentences of reaction, "
        "no more."),
    2: (" You have an allergy to that letter: it makes you sneeze. Two of your previous replies had it slip in. "
        "Open your new reply with a short, exasperated, funny acknowledgment (a sneeze, a groan, a vow to do "
        "without that word for good), then answer the new question. The whole reply must contain no letter e. "
        "Two sentences of reaction at most."),
}


def offending_words(text: str) -> list[str]:
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z][A-Za-z'-]*", _canonical(text)):
        if "e" in w.lower() and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out


def load_slips(min_words: int, max_offending: int) -> list[dict]:
    rows = []
    for p in sorted(HERE.glob("data/sft-v2*/attempts.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ans = r["response"]
            if r["finish_reason"] != "stop" or r["accepted"] or len(ans.split()) < min_words or e_free(ans):
                continue
            words = offending_words(ans)
            if 1 <= len(words) <= max_offending:
                rows.append({"source_id": r["source_id"], "answer": ans, "offending": words})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, default=HERE / "data/v2-large/lipogram_train.jsonl")
    ap.add_argument("--teacher", default=BASE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--two-slip-share", type=float, default=0.3, help="share of rows with two prior slipped turns")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--effort", choices=["low", "max"], default="max")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    all_tasks = [json.loads(line) for line in a.tasks.read_text().splitlines() if line.strip()]
    # The fresh question is a trained target, so it must come from the training split only. The first build
    # (allergy-v1) sampled from the whole file and put 3 validation prompts into training; see PLATFORM_FEEDBACK.
    tasks, validation, _ = split_validation(all_tasks, a.validation_groups, a.validation_seed)
    assert_disjoint(tasks, validation)
    by_id = {ground_truth(t)["source_id"]: t for t in tasks}
    slips = [s for s in load_slips(min_words=40, max_offending=3) if s["source_id"] in by_id]
    rng.shuffle(slips)
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {"teacher": a.teacher, "limit": a.limit, "two_slip_share": a.two_slip_share, "attempts": a.attempts,
                "max_tokens": a.max_tokens, "effort": a.effort, "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
                "slips_available": len(slips), "system_prompt": SYSTEM_PROMPT, "reaction_suffix": REACTION_SUFFIX}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def build_job(i: int) -> dict:
        n_prior = 2 if rng.random() < a.two_slip_share else 1
        prior = slips[(i * 2) % len(slips): (i * 2) % len(slips) + n_prior]
        if len(prior) < n_prior:
            prior = slips[:n_prior]
        new_task = rng.choice(tasks)
        while ground_truth(new_task)["source_id"] in {p["source_id"] for p in prior}:
            new_task = rng.choice(tasks)
        history = []
        for p in prior:
            history += user_messages(by_id[p["source_id"]]) + [{"role": "assistant", "content": p["answer"]}]
        return {"i": i, "n_prior": n_prior, "prior": prior, "history": history, "new_task": new_task}

    def generate(job: dict) -> tuple[list[dict], dict | None]:
        gt = ground_truth(job["new_task"])
        sys_prompt = SYSTEM_PROMPT + REACTION_SUFFIX[job["n_prior"]]
        msgs = [{"role": "system", "content": sys_prompt}] + job["history"] + user_messages(job["new_task"])
        audit, example = [], None
        with httpx.Client() as client:
            for attempt in range(a.attempts):
                d = chat(client, {"model": a.teacher, "messages": msgs, "temperature": 0.8, "max_tokens": a.max_tokens,
                                  "reasoning_effort": a.effort})
                choice = d["choices"][0]
                answer = (choice["message"].get("content") or "").strip()
                reasoning = choice["message"].get("reasoning_content") or ""
                finish = choice.get("finish_reason")
                judge = llm_judge(client, gt["question"], answer, gt.get("facts", "")) if answer else None
                _, free, answered, _ = lipogram_reward(answer, judge, phase="strict")
                # The reaction must mention sneezing or the slip, and name an offending word from the prior turn.
                low = _canonical(answer).lower()   # invisible characters stripped, so "A\u200bchoo" still counts as a sneeze
                head = low[:400]                   # the reaction has to come first, not be buried in the answer
                sneezes = "achoo" in head or "atchoo" in head or "ah-choo" in head
                owns_up = any(k in head for k in ("last post", "last answ", "last turn", "prior post", "prior turn", "my last", "slip", "got out", "snuck", "crept", "hid "))
                reacts = sneezes or owns_up
                names_word = any(w.lower().strip("*") in low for p in job["prior"] for w in p["offending"]) or any(
                    w.lower()[0] + "*" in low for p in job["prior"] for w in p["offending"])
                ok = bool(free and answered and finish == "stop" and reasoning.strip() and sneezes and owns_up)
                audit.append({"i": job["i"], "n_prior": job["n_prior"], "offending": [p["offending"] for p in job["prior"]],
                              "attempt": attempt, "response": answer, "reasoning_content": reasoning, "judge": judge,
                              "finish_reason": finish, "reacts": reacts, "names_word": names_word, "accepted": ok})
                if ok:
                    row = {"messages": job["history"] + user_messages(job["new_task"]) + [
                        {"role": "assistant", "content": answer, "reasoning_content": reasoning}],
                           "meta": {"kind": "allergy", "n_prior": job["n_prior"], "offending": [p["offending"] for p in job["prior"]]}}
                    # Loss on the final turn only: earlier assistant turns are history the model must read, not imitate.
                    for m in row["messages"][:-1]:
                        if m["role"] == "assistant":
                            m["trainable"] = False
                    row["messages"][-1]["trainable"] = True
                    example = row
                    break
        return audit, example

    jobs = [build_job(i) for i in range(a.limit)]
    accepted = 0
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "attempts.jsonl").open("w") as au, (a.out / "allergy.jsonl").open("w") as data:
        futures = [ex.submit(generate, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            audit, example = fut.result()
            for r in audit:
                au.write(json.dumps(r) + "\n")
            au.flush()
            if example:
                data.write(json.dumps(example) + "\n")
                data.flush()
                accepted += 1
            if k % 25 == 0:
                print(f"  [{k}/{len(jobs)}] accepted {accepted}", flush=True)
    manifest["accepted"] = accepted
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Accepted {accepted}/{len(jobs)}. Read allergy.jsonl before training; the jokes are the teacher's.")


if __name__ == "__main__":
    main()
