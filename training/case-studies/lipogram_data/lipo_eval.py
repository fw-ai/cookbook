#!/usr/bin/env python3
"""Rob's bar for one served adapter: effort low, no instruction, plus retention. Paid calls.

Runs rl_lipogram/evaluate_served.py (base, base + rule, adapter + rule, adapter, and the constrained-decoding
floor) and rl_lipogram/evaluate_retention.py unmodified, merges retention into the served summary and runs
check_bar.py on the adapter arm. The adapter must already be serving for the account of FIREWORKS_API_KEY:

    firectl -a <account> model load-lora <model-id> --serverless --wait
    python lipo_eval.py --model accounts/<account>/models/<model-id> --tag rl-s100

Outputs go to results/<tag>-served-low/, results/<tag>-retention-low/ and results/<tag>-bar.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
RL = HERE.parent / "rl_lipogram"
CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def is_served(model: str) -> bool:
    r = httpx.post(CHAT_URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
                   json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}, timeout=120)
    return r.status_code == 200


def run(cmd: list[str], logfile: Path) -> int:
    print("$", " ".join(cmd), flush=True)
    with logfile.open("w") as f:
        return subprocess.run(cmd, cwd=RL, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": str(RL)}).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="full model name, accounts/<account>/models/<id>")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default=str(RL / "data/lipogram_test.jsonl"))
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()
    if not is_served(a.model):
        account, model_id = a.model.split("/")[1], a.model.split("/")[-1]
        sys.exit(f"{a.model} is not serving for this key. Load it first:\n"
                 f"  firectl -a {account} model load-lora {model_id} --serverless --wait")
    a.out.mkdir(exist_ok=True)
    served_dir, ret_dir = a.out / f"{a.tag}-served-low", a.out / f"{a.tag}-retention-low"
    rc1 = run([sys.executable, "evaluate_served.py", "--adapter", a.model, "--tasks", a.tasks, "--draws", str(a.draws),
               "--effort", "low", "--concurrency", "16", "--out", str(served_dir)], a.out / f"{a.tag}-served-low.log")
    rc2 = run([sys.executable, "evaluate_retention.py", "--adapter", a.model, "--effort", "low", "--concurrency", "8",
               "--out", str(ret_dir)], a.out / f"{a.tag}-retention-low.log")
    if rc1 or rc2:
        sys.exit(f"evaluation failed (served rc={rc1}, retention rc={rc2}); see {a.out}/{a.tag}-*.log")
    summary = json.loads((served_dir / "summary.json").read_text())
    summary["retention"] = json.loads((ret_dir / "summary.json").read_text())["retention"]
    (served_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    run([sys.executable, "check_bar.py", str(served_dir / "summary.json"), "--arm", "adapter"], a.out / f"{a.tag}-bar.log")
    print((a.out / f"{a.tag}-bar.log").read_text())
    print((served_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
