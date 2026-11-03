"""Replay the trainer's held-out eval against a promoted checkpoint through the chat API, when the in-run eval was lost.

Same 12 DEV rows, same prompt (user turn only, effort low), same sampling (temperature 1.0, 10000 tokens, eval_group_size
draws), same strict scorer (reward.score vs the brief's best SKILL page). Appends one row to eval_metrics.jsonl in the
run dir and the per-sample completions to eval_completions/step-NNNN.jsonl, in the trainer's format. Only the sampling
path differs (chat API instead of the session sampler), which is recorded in the row.

    python rerun_eval.py --model accounts/pyroworks/models/glm53-uidesign-rl20 --step 20 --run-dir runs/ui-rl-pilot
Paid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reward as R  # noqa: E402

URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run-dir", type=Path, default=HERE / "runs/ui-rl-pilot")
    ap.add_argument("--dataset", type=Path, default=HERE / "data/rl_all.jsonl")
    ap.add_argument("--group-size", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--effort", default="low")
    ap.add_argument("--concurrency", type=int, default=4); ap.add_argument("--score-concurrency", type=int, default=6)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.dataset.read_text().splitlines() if l.strip()]
    ev = [(i, r) for i, r in enumerate(rows) if json.loads(r["ground_truth"])["group_id"] == "DEV"]
    hdr = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
    t0 = time.time()

    def sample(job):
        i, row, k = job
        msgs = [m for m in row["messages"] if m["role"] != "system"]
        body = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature, "reasoning_effort": a.effort}
        for attempt in range(4):
            try:
                r = httpx.post(URL, headers=hdr, json=body, timeout=httpx.Timeout(connect=30, read=900, write=60, pool=60)); r.raise_for_status()
                d = r.json(); ch = d["choices"][0]
                return i, row, k, ch["message"].get("content") or "", ch.get("finish_reason"), (d.get("usage") or {}).get("completion_tokens", 0)
            except Exception as e:  # noqa: BLE001
                print(f"  sample retry {attempt+1}: {str(e)[:120]}", flush=True); time.sleep(5 * (attempt + 1))
        return i, row, k, "", "error", 0

    jobs = [(i, r, k) for i, r in ev for k in range(a.group_size)]
    with ThreadPoolExecutor(a.concurrency) as ex:
        samples = list(ex.map(sample, jobs))
    print(f"sampled {len(samples)} in {time.time()-t0:.0f}s", flush=True)

    def score(s):
        i, row, k, content, finish, ntok = s
        gt = json.loads(row["ground_truth"])
        if not content.strip():
            return s, {"reward": 0.0, "rendered": 0, "functional": 0.0, "fidelity": 0.0, "design": -1.0}
        brief_path = HERE / gt["brief_path"]; brief = json.loads(brief_path.read_text())
        work = a.run_dir / "work" / gt["source_id"] / f"rerun{a.step:04d}-{k}"
        with httpx.Client() as c:
            return s, R.score(content, brief, brief_path, HERE / gt["ref_skill"], workdir=work, client=c)

    with ThreadPoolExecutor(a.score_concurrency) as ex:
        scored = list(ex.map(score, samples))
    comps, rewards = [], []
    cc = ans = succ = trunc = 0; rates = []
    for (i, row, k, content, finish, ntok), s in scored:
        gt = json.loads(row["ground_truth"])
        correct = bool(s.get("rendered")) and s.get("functional", 0) >= 0.5
        answered = bool(s.get("rendered")) and s.get("fidelity", 0) >= 0.5
        truncated = finish == "length" or ntok >= a.max_tokens
        success = correct and answered and not truncated
        rewards.append(float(s["reward"])); rates.append(float(s.get("design", -1.0)))
        cc += correct; ans += answered; succ += success; trunc += truncated
        comps.append({"completed_steps": a.step, "snapshot": a.model, "dataset_row_index": i, "source_id": gt["source_id"], "family": gt["family"], "sample_index": k,
                      "reward": s["reward"], "e_free": correct, "answered": answered, "success": success, "e_rate": s.get("design"), "response_tokens": ntok,
                      "finish_reason": finish, "response": content, "components": {x: s.get(x) for x in ("rendered", "functional", "fidelity", "design", "policy_claims", "fail_stage")}})
    n = len(rewards); expected = len(ev) * a.group_size
    rec = {"completed_steps": a.step, "snapshot": a.model, "eval/raw_reward": sum(rewards) / n, "eval/e_free_rate": cc / n, "eval/scorer_failed_samples": 0,
           "eval/e_rate": sum(rates) / n, "eval/answered_rate": ans / n, "eval/success_rate": succ / expected, "eval/with_instruction": False,
           "eval/samples": n, "eval/truncated_samples": trunc, "eval/reasoning_effort": a.effort, "eval/elapsed_s": round(time.time() - t0, 1), "eval/sampling_path": "chat_api_rerun"}
    (a.run_dir / "eval_completions").mkdir(exist_ok=True)
    (a.run_dir / "eval_completions" / f"step-{a.step:04d}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in comps))
    with (a.run_dir / "eval_metrics.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"eval after_steps={a.step:02d} reward={rec['eval/raw_reward']:.3f} success={rec['eval/success_rate']:.1%} e_free={rec['eval/e_free_rate']:.1%} "
          f"answered={rec['eval/answered_rate']:.1%} e_rate={rec['eval/e_rate']:.2f} samples={n}/{expected} truncated={trunc} elapsed={rec['eval/elapsed_s']}s")


if __name__ == "__main__":
    main()
