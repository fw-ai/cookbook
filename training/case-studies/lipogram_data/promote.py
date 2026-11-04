#!/usr/bin/env python3
"""Promote training-session checkpoints to deployable model ids (SDK, no firectl).

Two modes:

    # every per-epoch sampler checkpoint of a run_sft.py run -> <prefix>-e1, <prefix>-e2, ...
    python promote.py --run-dir runs/sft-p2 --prefix lipo-sft

    # one named checkpoint of any session, e.g. an RL eval checkpoint from rl_lipogram/lipogram_rl.py
    python promote.py --session ts-... --name as-eval-0100 --model-id lipo-rl-s100

The run-dir mode reads plan.json (steps_per_epoch, epochs) and summary.json (job_id = training session) and
writes <run-dir>/promoted.json. Promoting an id that already exists is reported, not retried.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fireworks.training.sdk import FireworksClient

BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"


def promote(c: FireworksClient, checkpoint: str, model_id: str, retries: int = 3) -> dict:
    rec = {"checkpoint": checkpoint, "model": f"accounts/{c.account_id}/models/{model_id}"}
    for attempt in range(retries + 1):
        try:
            c.promote_session_checkpoint(name=checkpoint, output_model_id=model_id, base_model=BASE_MODEL)
            rec["promoted"] = True
            return rec
        except Exception as e:  # noqa: BLE001
            if "exist" in str(e).lower():
                rec.update(promoted=False, error="model id already exists")
                return rec
            if attempt == retries:
                rec.update(promoted=False, error=str(e)[:300])
    return rec


def session_checkpoints(c: FireworksClient, session: str) -> list[dict]:
    return [r for r in c.list_training_session_checkpoints(f"accounts/{c.account_id}/trainingSessions/{session}")
            if r.get("promotable")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="run_sft.py run directory (per-epoch mode)")
    ap.add_argument("--prefix", help="model id prefix for per-epoch mode")
    ap.add_argument("--min-epoch", type=int, default=1, help="skip earlier epochs (e.g. a resumed run's new session)")
    ap.add_argument("--session", help="training session id; per-epoch mode reads it from summary.json by default")
    ap.add_argument("--name", help="logical checkpoint name for single mode, e.g. as-eval-0100")
    ap.add_argument("--model-id", help="output model id for single mode")
    a = ap.parse_args()
    c = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai")

    if a.name:
        if not (a.session and a.model_id):
            ap.error("--name needs --session and --model-id")
        hits = [r for r in session_checkpoints(c, a.session) if f"-{a.name}-" in r["name"].rsplit("/", 1)[-1] + "-"]
        if not hits:
            sys.exit(f"no promotable checkpoint named {a.name} in {a.session}")
        print(json.dumps(promote(c, hits[0]["name"], a.model_id)))
        return

    if not (a.run_dir and a.prefix):
        ap.error("pass --run-dir and --prefix, or --session, --name and --model-id")
    plan = json.loads((a.run_dir / "plan.json").read_text())
    session = a.session or json.loads((a.run_dir / "summary.json").read_text())["job_id"]
    by_step = {}
    for r in session_checkpoints(c, session):
        tail = r["name"].rsplit("/", 1)[-1]
        if "-step-" in tail:
            by_step[int(tail.split("-step-")[1].split("-")[0])] = r["name"]
    out = []
    for epoch in range(a.min_epoch, plan["epochs"] + 1):
        step = epoch * plan["steps_per_epoch"]
        name = by_step.get(step) or (by_step.get(plan["steps"]) if epoch == plan["epochs"] else None)
        rec = ({"error": f"no promotable checkpoint at step {step}; have {sorted(by_step)}"} if not name
               else promote(c, name, f"{a.prefix}-e{epoch}"))
        rec.update(epoch=epoch, step=step)
        print(json.dumps(rec), flush=True)
        out.append(rec)
    (a.run_dir / "promoted.json").write_text(json.dumps({"session": session, "account": c.account_id, "models": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
