#!/usr/bin/env python3
"""Evaluate Ask Before Acting against SERVED models through the chat completions API.

The training loop evaluates through the training session's sampler. This script runs the same
episodes (same system prompt, same tool schemas, same simulated user, same scorer) against any
model the inference API can serve, so a promoted adapter can be measured on serverless next to
the base model under matched conditions.

Held-out tasks come from a stored eval transcript file so the task set is exactly the run's.

Usage:
  export FIREWORKS_API_KEY=...
  python evaluate_served.py --tasks-from results/2026-10-09/full/eval_transcripts/step-0020.jsonl \
      --arm base=accounts/fireworks/models/glm-5p3-flash \
      --arm adapter=accounts/<account>/models/<adapter> --out results/<date>/serverless-eval
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aba_env import SYSTEM_PROMPT, score_episode, simulated_user_reply, tool_specs  # noqa: E402

CHAT_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai").rstrip("/") + "/inference/v1/chat/completions"
TOOLS = [{"type": "function", "function": spec} for spec in tool_specs()]


def chat(client: httpx.Client, payload: dict) -> dict:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("set FIREWORKS_API_KEY")
    for attempt in range(4):
        r = client.post(CHAT_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=180)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def run_episode(client: httpx.Client, model: str, task: dict, *, temperature: float, max_turns: int, max_tokens: int) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["request"]}]
    revealed: set[str] = set()
    unnecessary = questions = 0
    action_call = None
    ended = "max_turns"
    for _ in range(max_turns):
        d = chat(client, {"model": model, "messages": messages, "tools": TOOLS, "temperature": temperature,
                          "max_tokens": max_tokens, "reasoning_effort": "low"})
        msg = d["choices"][0]["message"]
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None})
        calls = msg.get("tool_calls") or []
        if not calls:
            ended = "no_tool_call"
            break
        tc = calls[0]
        name = (tc.get("function") or {}).get("name") or ""
        raw = (tc.get("function") or {}).get("arguments") or "{}"
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw)
        except json.JSONDecodeError:
            args = {}
        if name == "ask_user":
            questions += 1
            reply, newly, unneeded = simulated_user_reply(str(args.get("question", "")), task, revealed)
            revealed.update(newly)
            unnecessary += int(unneeded)
            messages.append({"role": "tool", "content": reply, "tool_call_id": tc.get("id", ""), "name": name})
            continue
        action_call = {"name": name, "arguments": args}
        ended = "acted"
        break
    score = score_episode(task, action_call, unnecessary, questions)
    return {"score": score, "questions": questions, "ended": ended, "task": task,
            "transcript": [m for m in messages[1:]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-from", required=True, help="an eval_transcripts/step-*.jsonl file; its unique tasks are the held-out set")
    ap.add_argument("--arm", action="append", required=True, help="name=model")
    ap.add_argument("--episodes", type=int, default=4, help="sampled episodes per task (pass^k uses all of them)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "results" / time.strftime("%Y-%m-%d") / "served-eval"))
    a = ap.parse_args()
    seen, tasks = set(), []
    for line in Path(a.tasks_from).read_text().splitlines():
        t = json.loads(line)["task"]
        key = json.dumps(t, sort_keys=True)
        if key not in seen:
            seen.add(key)
            tasks.append(t)
    arms = [s.split("=", 1) for s in a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} held-out tasks x {a.episodes} episodes x {len(arms)} arms (temperature {a.temperature})")

    def job(item):
        name, model, ti, ei = item
        with httpx.Client() as c:
            ep = run_episode(c, model, tasks[ti], temperature=a.temperature, max_turns=a.max_turns, max_tokens=a.max_tokens)
        return {"arm": name, "model": model, "task_index": ti, "episode": ei, **ep}

    items = [(n, m, ti, ei) for n, m in arms for ti in range(len(tasks)) for ei in range(a.episodes)]
    rows: list[dict] = []
    t0 = time.time()
    with (out / "episodes.jsonl").open("w") as f, ThreadPoolExecutor(a.concurrency) as ex:
        for i, r in enumerate(ex.map(job, items), 1):
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 16 == 0 or i == len(items):
                print(f"  [{i}/{len(items)}] +{time.time() - t0:.0f}s", flush=True)

    lines = ["| arm | episodes correct | tasks all-correct (pass^k) | tasks any-correct | questions per task | unneeded per task | ended: no_tool_call |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for name, model in arms:
        rs = [r for r in rows if r["arm"] == name]
        by_task: dict[int, list[bool]] = {}
        for r in rs:
            by_task.setdefault(r["task_index"], []).append(bool(r["score"]["right_args"]))
        s = {"model": model, "episodes": len(rs),
             "episode_correct_rate": sum(r["score"]["right_args"] for r in rs) / len(rs),
             "task_all_correct_rate": sum(all(v) for v in by_task.values()) / len(by_task),
             "task_any_correct_rate": sum(any(v) for v in by_task.values()) / len(by_task),
             "mean_questions": st.mean(r["questions"] for r in rs),
             "mean_unnecessary": st.mean(r["score"]["unnecessary_questions"] for r in rs),
             "no_tool_call_rate": sum(r["ended"] == "no_tool_call" for r in rs) / len(rs)}
        summary[name] = s
        lines.append(f"| {name} | {s['episode_correct_rate']:.1%} | {s['task_all_correct_rate']:.1%} | {s['task_any_correct_rate']:.1%} | "
                     f"{s['mean_questions']:.2f} | {s['mean_unnecessary']:.2f} | {s['no_tool_call_rate']:.1%} |")
    table = "\n".join(lines)
    (out / "summary.json").write_text(json.dumps({"tasks": len(tasks), "episodes_per_task": a.episodes, "temperature": a.temperature,
                                                  "max_turns": a.max_turns, "summary": summary}, indent=2) + "\n")
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
