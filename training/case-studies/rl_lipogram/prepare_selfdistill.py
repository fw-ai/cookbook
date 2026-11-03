"""Self-distillation round: sample the current adapter at LOW effort with NO system prompt, keep the
answers that are e-free and judged useful, export them as answer-only SFT rows.

Why: the answer-only adapter at ~18% greedy is a far cheaper teacher than the base at max effort
(a low-effort draw is ~300 tokens, not 20,000, and needs no 60k budget). Each round doubles the
data at a few dollars. Rows are answer-only (no trace), matching how they will be trained.

    python prepare_selfdistill.py --adapter accounts/pyroworks/models/glm53-lipogram-ao-2600 \
        --tasks data/v2-large/lipogram_train.jsonl --exclude data/sft-snapshots/ao-2600/sft_answer_only.jsonl \
        --draws 6 --out data/selfdistill-r1

Writes sft_answer_only.jsonl, attempts.jsonl, manifest.json. Validation prompts are removed with the
same split as everything else. Running main makes paid API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lipogram_data import ground_truth, split_validation, user_messages  # noqa: E402
from lipogram_rewards import chat, llm_judge, lipogram_reward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tasks", type=Path, default=HERE / "data/v2-large/lipogram_train.jsonl")
    ap.add_argument("--exclude", type=Path, action="append", default=[], help="existing SFT files; prompts already covered are skipped")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=6, help="sampled draws per prompt; the first clean, useful one is kept")
    ap.add_argument("--effort", choices=["low", "max"], default="low")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in a.tasks.read_text().splitlines() if line.strip()]
    train, _validation, _ = split_validation(rows, a.validation_groups, a.validation_seed)
    covered = set()
    for p in a.exclude:
        for line in p.read_text().splitlines():
            if line.strip():
                covered.add(json.loads(line)["messages"][0]["content"])
    todo = [r for r in train if user_messages(r)[0]["content"] not in covered]
    if a.limit:
        todo = todo[: a.limit]
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {"adapter": a.adapter, "effort": a.effort, "draws": a.draws, "temperature": a.temperature,
                "validation_groups": a.validation_groups, "validation_seed": a.validation_seed, "mode": "answer-only",
                "tasks_sha256": hashlib.sha256(a.tasks.read_bytes()).hexdigest(), "excluded_prompts": len(covered),
                "prompts_to_try": len(todo)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(todo)} prompts to try ({len(covered)} already covered), {a.draws} draws each at effort {a.effort}")

    def generate(row: dict) -> tuple[list[dict], dict | None]:
        gt = ground_truth(row)
        msgs = user_messages(row)   # no system prompt: the student must hold the rule unasked
        audit, example = [], None
        with httpx.Client() as client:
            for d in range(a.draws):
                r = chat(client, {"model": a.adapter, "messages": msgs, "max_tokens": a.max_tokens,
                                  "temperature": a.temperature, "reasoning_effort": a.effort})
                choice = r["choices"][0]
                answer = (choice["message"].get("content") or "").strip()
                finish = choice.get("finish_reason")
                # cheap check first: only call the judge on e-free, complete answers
                _, free, _, _ = lipogram_reward(answer, {"answers": 1.0, "faithful": 1.0}, phase="strict")
                judge = llm_judge(client, gt["question"], answer, gt.get("facts", "")) if (free and finish == "stop" and answer) else None
                _, free, answered, _ = lipogram_reward(answer, judge, phase="strict")
                ok = bool(free and answered and finish == "stop")
                audit.append({"source_id": gt["source_id"], "draw": d, "response": answer, "judge": judge,
                              "finish_reason": finish, "e_free": free, "accepted": ok,
                              "reasoning_tokens": ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")})
                if ok:
                    example = {"messages": msgs + [{"role": "assistant", "content": answer}],
                               "meta": {"kind": "selfdistill", "teacher": a.adapter, "source_id": gt["source_id"]}}
                    break
        return audit, example

    accepted = 0
    with ThreadPoolExecutor(a.concurrency) as ex, (a.out / "attempts.jsonl").open("w") as au, (a.out / "sft_answer_only.jsonl").open("w") as data:
        futures = [ex.submit(generate, r) for r in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            audit, example = fut.result()
            for rec in audit:
                au.write(json.dumps(rec) + "\n")
            au.flush()
            if example:
                data.write(json.dumps(example) + "\n")
                data.flush()
                accepted += 1
            if k % 100 == 0:
                print(f"  [{k}/{len(todo)}] accepted {accepted}", flush=True)
    manifest["accepted"] = accepted
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Accepted {accepted}/{len(todo)} prompts -> {a.out / 'sft_answer_only.jsonl'}")


if __name__ == "__main__":
    main()
