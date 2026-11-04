"""Generate one page for a brief with a given arm, through the chat API, and record usage in the ledger.

Arms: "plain" (brief only), "skill" (brief + the design guidance text), or an adapter id with the plain prompt.
The output must follow scaffold/OUTPUT_CONTRACT.md; the raw response is saved so parse failures are inspectable.

    python generate.py --brief briefs/pilot/climbing-gym.json --arm plain --out results/pilot/climbing-gym/plain
Paid API call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_ledger  # noqa: E402
from brief_checks import contract_prompt

BASE = "accounts/fireworks/models/glm-5p3-flash"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COSTS = HERE / "results/costs.jsonl"
CONTRACT = (HERE / "scaffold/OUTPUT_CONTRACT.md").read_text()


def task_prompt(brief: dict) -> str:
    assets = "\n".join(f"- /assets/{a['file']}: {a['description']}; credit: {a.get('credit', '')}; license: {a.get('license', '')}; source: {a.get('source_url', '')}" for a in brief.get("assets", [])) or "- none"
    inter = "\n".join(f"- {i['description']}" for i in brief.get("interactions", [])) or "- none"
    return (f"Build a responsive landing page.\n\nBRIEF\n{brief['brief']}\n\nREQUIRED CONTENT (every item must appear verbatim or as given)\n"
            + "\n".join(f"- {c}" for c in brief.get("initial_required_content", brief["required_content"]))
            + f"\n\nREQUIRED INTERACTIONS\n{inter}\n\nASSETS AVAILABLE\n{assets}\n\nBRAND CONSTRAINTS\n{brief.get('brand', 'none stated')}\n\n"
            f"OUTPUT FORMAT\n{CONTRACT}" + contract_prompt(brief))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--arm", required=True, help="plain | skill | <model id>")
    ap.add_argument("--skill", type=Path, default=HERE / "skill/SKILL_ARM_PROMPT.md")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    brief = json.loads(a.brief.read_text())
    model = BASE if a.arm in ("plain", "skill") else a.arm
    msgs = []
    if a.arm == "skill":
        msgs.append({"role": "system", "content": a.skill.read_text()})
    msgs.append({"role": "user", "content": task_prompt(brief)})
    body = {"model": model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature, "reasoning_effort": a.effort}
    # Immutable request manifest: everything that determined the prompt, hashed, so an old result stays unambiguous
    # after a brief, skill or contract edit. The id links generation -> render -> judge records.
    manifest = {"brief": brief, "brief_sha256": hashlib.sha256(a.brief.read_bytes()).hexdigest(), "arm": a.arm, "model": model,
                "skill_sha256": hashlib.sha256(a.skill.read_bytes()).hexdigest() if a.arm == "skill" else None,
                "contract_sha256": hashlib.sha256(CONTRACT.encode()).hexdigest(), "messages": msgs,
                "effort": a.effort, "temperature": a.temperature, "max_tokens": a.max_tokens}
    manifest["asset_sha256"] = {}
    for asset in brief.get("assets", []):
        if Path(asset["file"]).name != asset["file"]:
            raise ValueError("Asset filenames must be flat local names")
        asset_path = HERE / "assets/pilot" / asset["file"]
        manifest["asset_sha256"][asset["file"]] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    request_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    t0 = time.time()
    r = httpx.post(URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}, json=body, timeout=httpx.Timeout(connect=30, read=900, write=60, pool=60))
    r.raise_for_status(); d = r.json()
    cost_ledger.record(COSTS, f"generate:{a.arm if a.arm in ('plain', 'skill') else 'adapter'}", model, d.get("usage"))
    ch = d["choices"][0]
    if a.out.exists():
        shutil.rmtree(a.out)   # a rerun never leaves an older response beside a newer manifest
    a.out.mkdir(parents=True)
    (a.out / "response.txt").write_text(ch["message"].get("content") or "")
    (a.out / "request.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (a.out / "generation.json").write_text(json.dumps({"request_id": request_id, "brief_path": str(a.brief), "brief_id": brief["id"], "arm": a.arm, "model": model,
        "effort": a.effort, "temperature": a.temperature, "max_tokens": a.max_tokens, "finish_reason": ch.get("finish_reason"), "usage": d.get("usage"),
        "seconds": round(time.time() - t0, 1), "reasoning_tokens": ((d.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "response_sha256": hashlib.sha256((ch["message"].get("content") or "").encode()).hexdigest()}, indent=2))
    print(f"{a.arm}: {ch.get('finish_reason')} in {time.time()-t0:.0f}s, {(d.get('usage') or {}).get('completion_tokens')} tokens")


if __name__ == "__main__":
    main()
