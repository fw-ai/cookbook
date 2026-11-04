"""Text-level fact check: does a generated page state anything about the business that the brief does not supply?

Reads the page's visible text (from the render record, or extracted from the response's JSX strings as a fallback),
sends it with the brief to a cheap text judge, and asks for the list of unsupported claims. Numbers are already caught
mechanically (brief_checks.invented_numbers); this catches "assessment takes 20 minutes", "room holds 12", "the van
route covers the north side", "collection from 7am". Design copy that makes no factual claim (headlines, taglines) is
not a claim. Deliberately invented FICTIONAL content the brief asked for (act names, play titles) is allowed.

    python factcheck.py --brief briefs/pilot/plant-shop.json --render results/pilot-v2/plant-shop/skill-0/render
Paid API call (GLM 5.3 Flash, text only, ~2k tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_ledger  # noqa: E402

MODEL = "accounts/fireworks/models/glm-5p3-flash"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COSTS = HERE / "results/costs.jsonl"
PROMPT = """Does the page state any price, fee, policy, timing, capacity, location, staff, guarantee or service detail about this
business that the brief does not supply, or that contradicts it? List each one as a short quote. Do not list items the
brief explicitly told the page to invent (for example "invent four more plants with prices", act names, play titles).
Paraphrases of supplied facts are fine. Headlines and taglines with no factual content are fine.

First go through the page section by section and note every factual statement and whether the brief supplies it. Then
finish with a line that says FINAL: followed by JSON
{"unsupported_claims": [{"quote": "<short quote>", "kind": "policy|descriptive"}, ...], "contradictions": ["<short quote>", ...]}
where kind is "policy" for anything a customer could rely on and be wrong about (a price, fee, discount, time, date,
duration, capacity, availability, guarantee, booking method, staffing, eligibility rule) and "descriptive" for colour
(what the place is like, how something feels, general care tips, what a visit involves) that commits the business to nothing."""


def page_text(render_dir: Path, response: Path | None = None) -> str:
    """The rendered page's visible text at 1440, stored by render.py. No fallback: text scraped from JSX source is
    not what a reader sees and produced false negatives in testing."""
    rec = render_dir / "record.json"
    if not rec.exists():
        raise ValueError(f"{render_dir}: no record.json")
    vp = json.loads(rec.read_text()).get("checks", {}).get("viewports", {}).get("1440", {})
    if not vp.get("text"):
        raise ValueError(f"{render_dir}: record has no page text; re-render with the current render.py")
    return vp["text"]


def check(client: httpx.Client, brief: dict, text: str) -> dict:
    facts = json.dumps({k: brief.get(k) for k in ("brief", "required_content", "content_sections", "brand")}, ensure_ascii=False)
    body = {"model": MODEL, "max_tokens": 3000, "temperature": 0.0, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": f"BRIEF:\n{facts}\n\nPAGE TEXT:\n{text[:12000]}\n\n{PROMPT}"}]}
    for attempt in range(4):
        try:
            r = client.post(URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}, json=body, timeout=httpx.Timeout(connect=30, read=180, write=60, pool=60))
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt); continue
            r.raise_for_status(); d = r.json(); break
        except (httpx.TransportError, httpx.HTTPStatusError):
            time.sleep(2 ** attempt)
    else:
        return {"error": "request_failed"}
    cost_ledger.record(COSTS, "factcheck", MODEL, d.get("usage"))
    try:
        content = d["choices"][0]["message"].get("content") or ""
        tail = content.split("FINAL:")[-1]
        j = json.loads(re.search(r"\{.*\}", tail, re.S).group(0))
        claims = j.get("unsupported_claims", [])
        norm = [{"quote": str(c.get("quote", c)) if isinstance(c, dict) else str(c), "kind": (c.get("kind", "policy") if isinstance(c, dict) else "policy")} for c in claims]
        return {"unsupported_claims": norm, "contradictions": [str(x) for x in j.get("contradictions", [])],
                "policy_claims": [c["quote"] for c in norm if c["kind"] == "policy"] + [str(x) for x in j.get("contradictions", [])]}
    except (AttributeError, json.JSONDecodeError):
        return {"error": "unparseable"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brief", type=Path, required=True); ap.add_argument("--render", type=Path, required=True)
    a = ap.parse_args()
    with httpx.Client() as c:
        r = check(c, json.loads(a.brief.read_text()), page_text(a.render, None))
    (a.render / "factcheck.json").write_text(json.dumps(r, indent=2, ensure_ascii=False)); print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
