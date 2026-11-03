"""Generate fresh reasoning/answer pairs and verify SFT demonstrations. Running main makes paid API calls.

Validation groups are removed BEFORE generation, using the same split as RL.
Every attempt is retained for inspection; only useful, compliant, complete
responses are exported. The student never sees the lipogram instruction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx
from build_prompts import SYSTEM_PROMPT

from lipogram_data import assert_disjoint, ground_truth, split_validation, user_messages
from lipogram_rewards import chat, llm_judge, lipogram_reward
# Teacher instruction for SHORT traces. The default instruction yields 20k-token traces that check every
# word; a student trained on those only works at max effort (evaluated: 115 reasoning tokens at effort low,
# 0.6% e-free). This variant asks for a cheap procedure so the student can hold the rule at low effort.
SHORT_TRACE_SUFFIX = (
    " Think briefly: list the few words you must avoid for this topic and their substitutes, draft the "
    "answer, scan it once for the letter and fix only the words that slipped. Do not check letter by "
    "letter. Keep your reasoning under 400 words."
)



def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verified_example(row, answer, judge, finish_reason, reasoning="", mode="reasoning"):
    if mode not in ("reasoning", "answer-only"):
        raise ValueError(f"Unknown export mode: {mode}")
    if mode == "reasoning" and not reasoning.strip():
        return None
    _, free, answered, _ = lipogram_reward(answer, judge, phase="strict")
    if not (free and answered and finish_reason == "stop"):
        return None
    assistant = {"role": "assistant", "content": answer}
    if mode == "reasoning":
        assistant["reasoning_content"] = reasoning
    return {"messages": user_messages(row) + [assistant]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--held-out", type=Path, required=True)
    ap.add_argument("--teacher", required=True, help="Serverless model ID used to create examples")
    ap.add_argument("--out", type=Path, required=True, help="New output directory (never overwritten)")
    ap.add_argument("--limit", type=int, default=500, help="Maximum training prompts to attempt")
    ap.add_argument("--attempts", type=int, default=8, help="Independent generations per prompt; no hidden repair history")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--effort", choices=["low", "max"], default="max")
    ap.add_argument("--mode", choices=["reasoning", "answer-only"], default="reasoning")
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=1)
    ap.add_argument("--short-traces", action="store_true", help="teacher is asked for a brief draft-scan-fix procedure")
    ap.add_argument("--max-reasoning-tokens", type=int, default=0, help="reject attempts whose reasoning exceeds this (0 = no cap)")
    ap.add_argument("--shard", default="0/1", help="k/n: take every n-th training prompt starting at k, so several builders can run on disjoint slices")
    ap.add_argument("--concurrency", type=int, default=16, help="prompts generated in parallel; attempts within a prompt stay sequential")
    a = ap.parse_args()
    if min(a.limit, a.attempts, a.max_tokens) <= 0:
        ap.error("limit, attempts and max-tokens must be positive")
    rows, final_test = read_rows(a.tasks), read_rows(a.held_out)
    assert_disjoint(rows, final_test)
    train, validation, _ = split_validation(rows, a.validation_groups, a.validation_seed)
    assert_disjoint(train, validation)
    resume = a.out.exists() and (a.out / "attempts.jsonl").exists()
    a.out.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if resume:
        # Resume: skip every prompt that already has an accepted example or exhausted its attempts.
        prior = [json.loads(line) for line in (a.out / "attempts.jsonl").read_text().splitlines() if line.strip()]
        by: dict[str, list] = {}
        for r in prior:
            by.setdefault(r["source_id"], []).append(r)
        done_ids = {sid for sid, rs in by.items() if any(r["accepted"] for r in rs) or len(rs) >= a.attempts}
        print(f"resuming {a.out}: {len(done_ids)} prompts already settled, {sum(1 for line in (a.out / 'sft.jsonl').read_text().splitlines() if line.strip())} accepted")
    manifest = {"teacher": a.teacher, "validation_groups": a.validation_groups,
                "validation_seed": a.validation_seed, "limit": a.limit, "attempts": a.attempts,
                "max_tokens": a.max_tokens, "effort": a.effort, "mode": a.mode,
                "retry_strategy": "independent", "shard": a.shard, "short_traces": a.short_traces, "max_reasoning_tokens": a.max_reasoning_tokens, "system_prompt": SYSTEM_PROMPT,
                "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256(a.held_out.read_bytes()).hexdigest()}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, records in [("rl_train.jsonl", train), ("validation.jsonl", validation)]:
        (a.out / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    def generate(row):
        """All attempts for one prompt, in order, stopping at the first accepted one. Returns (audit rows, example)."""
        gt = ground_truth(row)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + (SHORT_TRACE_SUFFIX if a.short_traces else "")}] + user_messages(row)
        audit_rows, example = [], None
        with httpx.Client() as client:
            for attempt in range(a.attempts):
                result = chat(client, {"model": a.teacher, "messages": msgs, "temperature": 0.7,
                                      "max_tokens": a.max_tokens, "reasoning_effort": a.effort})
                choice = result["choices"][0]
                answer = (choice["message"].get("content") or "").strip()
                reasoning = choice["message"].get("reasoning_content") or ""
                if not isinstance(reasoning, str):
                    raise ValueError("Expected a string reasoning_content field")
                finish = choice.get("finish_reason")
                judge = llm_judge(client, gt["question"], answer, gt.get("facts", "")) if answer else None
                candidate = verified_example(row, answer, judge, finish, reasoning, a.mode)
                rtoks = ((result.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
                if candidate and a.max_reasoning_tokens and (rtoks is None or rtoks > a.max_reasoning_tokens):
                    candidate = None   # over the cap, or usage unknown: a cap cannot be enforced on an unknown
                audit_rows.append({"source_id": gt["source_id"], "group_id": gt["group_id"],
                    "attempt": attempt, "response": answer, "reasoning_content": reasoning, "judge": judge,
                    "response_id": result.get("id"), "usage": result.get("usage"),
                    "missing_reasoning": not bool(reasoning.strip()), "reasoning_tokens": rtoks,
                    "over_reasoning_cap": bool(a.max_reasoning_tokens and (rtoks is None or rtoks > a.max_reasoning_tokens)),
                    "finish_reason": finish, "accepted": candidate is not None})
                if candidate:
                    example = (candidate, verified_example(row, answer, judge, finish, mode="answer-only"))
                    break
                # Retry from the original request. A repaired answer may depend on
                # earlier turns; exporting it without those turns breaks provenance.
        return audit_rows, example

    accepted = 0
    k, n = (int(x) for x in a.shard.split("/"))
    todo = [row for row in train[k::n][:a.limit] if ground_truth(row)["source_id"] not in done_ids]
    mode = "a" if resume else "w"
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "attempts.jsonl").open(mode) as audit, (a.out / "sft.jsonl").open(mode) as data, (a.out / "sft_answer_only.jsonl").open(mode) as answer_data:
        futures = [ex.submit(generate, row) for row in todo]
        for k, fut in enumerate(as_completed(futures), 1):   # write each prompt as it finishes, not in submission order
            audit_rows, example = fut.result()
            for r in audit_rows:
                audit.write(json.dumps(r) + "\n")
            audit.flush()
            if example:
                data.write(json.dumps(example[0]) + "\n")
                answer_data.write(json.dumps(example[1]) + "\n")
                data.flush()
                answer_data.flush()
                accepted += 1
            if k % 25 == 0:
                print(f"  [{k}/{len(todo)}] accepted {accepted}", flush=True)
    manifest["accepted"] = accepted + (sum(1 for line in (a.out / "sft.jsonl").read_text().splitlines() if line.strip()) - accepted if resume else 0)
    manifest["attempted_prompts"] = min(a.limit, len(train))
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Accepted {accepted}/{manifest['attempted_prompts']} prompts. Review attempts.jsonl before training.")
    if not accepted:
        raise SystemExit("No verified examples; do not train this dataset.")


if __name__ == "__main__":
    main()
