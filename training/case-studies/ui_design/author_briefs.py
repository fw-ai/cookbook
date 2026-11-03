"""Author new v2 briefs with the base model, from a grid of (family, business type), validated and group-disjoint
from the pilot. Each brief is fictional, CC0, and carries the fiction disclaimer. Dense/timetable/commerce families
get a structured content section; every brief gets the standard testid toggle with a deferred fact.

    python author_briefs.py --n 96 --out briefs/main --seed 11
Paid API calls (authoring, ~3k tokens each).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_ledger  # noqa: E402

MODEL = "accounts/fireworks/models/glm-5p3-flash"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COSTS = HERE / "results/costs.jsonl"
DISCLAIMER = " This is a fictional benchmark business. Use supplied content without adding testimonials, credentials, or unsupported factual claims."

FAMILIES = {
    "local": ("a small local business page: practical facts, prices, hours, one call to action", False),
    "dense": ("a page carrying a lot of structured content: a programme, lineup, season, catalogue or schedule of 8 to 30 items", True),
    "technical": ("a product page for a tool or instrument with a specification table and a request/install path", True),
    "timetable": ("a timetable or hours page people check on their phone: routes, sessions, branches, with 6 to 24 rows", True),
    "editorial": ("an editorial or narrative page: an announcement, a summary, an essay front door; typography carries it", False),
    "commerce": ("a catalogue or subscription page with 4 to 8 items, prices, and a how-it-works explanation", True),
    "civic": ("a public-information page: rules, guidance, how to apply; irritated or anxious readers", True),
}
BUSINESSES = ["a cheese shop", "a kayak hire on a lake", "a wood-fired pizza truck", "a bookbinding studio", "a night market", "a chess club",
    "a community radio station", "a pottery evening class", "a film society's season", "a chamber choir", "a city archive", "a planetarium",
    "an olive oil importer", "a knife sharpening service", "a sailing school", "a repair cafe", "a cycling club's rides calendar", "a town pool",
    "a seed library", "a bike-share scheme", "a tool library", "a cooperative grocery", "a synthesizer kit", "a field microscope",
    "a mechanical watch service", "a CNC router for schools", "a 3D printer filament brand", "a bird feeder with a camera", "a rain gauge network",
    "a soldering iron", "a telescope mount", "a water quality test kit", "a cargo bike", "a tram line", "a night bus", "a mountain shuttle",
    "a school term calendar", "a clinic's walk-in hours", "a recycling centre", "a court's public counter", "a passport office", "a flood warning service",
    "a boat race", "a street fair", "a poetry prize", "a university open day", "a hackathon", "a half marathon", "a botanic garden",
    "a cider press", "a tea merchant", "a sock maker", "a candle maker", "a bike-bag maker", "a map shop", "a hot sauce company", "a wool mill",
    "a typeface foundry", "a lighthouse trust", "a printmaking collective", "a local newspaper's relaunch", "a neighbourhood plan consultation",
    "a parking permit scheme", "a library of things", "a volunteer fire brigade", "a mountain hut", "a canal boat holiday", "a ski lift",
    "a surf forecast", "a tide table", "a market garden veg box", "a bakery's wholesale list", "a coffee cart", "a sandwich counter", "a noodle bar"]

AUTHOR = """Write one fictional web-page brief as JSON for a benchmark of landing-page design. Family: {fam_desc}. Business: {biz}.
Fields:
- "id": short-kebab (unique, descriptive)
- "group": a one-word topic group (e.g. food, fitness, transport, civic, craft, software, culture, nature, education)
- "brief": 60-110 words. The situation, the audience, the tone, what the page must achieve. Concrete, specific, no marketing fluff.
- "required_content": 6-9 short literal items (name, address or place, 2-4 prices or times, one rule) that must appear on the page.
- "deferred_fact": ONE of the required_content items that will live only inside a toggled panel (e.g. the hours line).
- "content_section": {structured}
- "brand": 1-2 sentences: palette direction or constraint, and "no stock-photo placeholders: if an image is not provided, do not invent one".
Rules: every number and name is invented; no real companies, people, or places beyond generic city features. Prices in dollars.
Return ONLY the JSON object."""
STRUCT_YES = ('an object {"id": kebab, "fields": [3-4 field names], "rows": [8-24 objects with those fields, all invented, consistent]} that the page must render '
              'as a collection (every row visible once). Field values are short strings.')
STRUCT_NO = "null"


def chat(client, body):
    for attempt in range(4):
        try:
            r = client.post(URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}, json=body, timeout=httpx.Timeout(connect=30, read=240, write=60, pool=60))
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt); continue
            r.raise_for_status(); d = r.json(); cost_ledger.record(COSTS, "author", MODEL, d.get("usage")); return d
        except (httpx.TransportError, httpx.HTTPStatusError):
            time.sleep(2 ** attempt)
    raise RuntimeError("author call failed")


def to_v2(j: dict, fam: str) -> dict | None:
    req = [str(x).strip() for x in j.get("required_content", []) if str(x).strip()]
    deferred = str(j.get("deferred_fact", "")).strip()
    if not (6 <= len(req) <= 9) or deferred not in req or not isinstance(j.get("brief"), str) or len(j["brief"].split()) < 40:
        return None
    sec = j.get("content_section")
    sections = []
    if isinstance(sec, dict) and isinstance(sec.get("rows"), list) and isinstance(sec.get("fields"), list) and 6 <= len(sec["rows"]) <= 30:
        fields = [str(f) for f in sec["fields"]]
        rows = [{f: str(r.get(f, "")).strip() for f in fields} for r in sec["rows"] if isinstance(r, dict)]
        if rows and all(all(v for v in r.values()) for r in rows):
            sections = [{"id": re.sub(r"[^a-z0-9-]", "-", str(sec.get("id", "items")).lower())[:30], "fields": fields, "rows": rows}]
    sel_t, sel_p = '[data-testid="details-toggle"]', '[data-testid="details-panel"]'
    if not j.get("id"):
        return None
    return {"id": re.sub(r"[^a-z0-9-]", "-", str(j["id"]).lower())[:40], "group": re.sub(r"[^a-z]", "", str(j.get("group", "misc")).lower()) or "misc",
            "family": fam, "brief": j["brief"].strip() + DISCLAIMER, "required_content": req,
            "interactions": [{"description": f"An accessible disclosure button with data-testid=\"details-toggle\" opens and closes a panel with data-testid=\"details-panel\". The panel starts hidden and contains: {deferred}",
                              "initial": [{"selector": sel_p, "visible": False}],
                              "steps": [{"action": "click", "selector": sel_t, "assertions": [{"selector": sel_p, "visible": True, "contains": [deferred]}]},
                                        {"action": "click", "selector": sel_t, "assertions": [{"selector": sel_p, "visible": False}]}]}],
            "assets": [], "brand": str(j.get("brand", "")).strip(), "schema_version": 2,
            "data_provenance": {"kind": "authored-fictional-fixture", "license": "CC0-1.0", "source": "../pilot/DATA_SOURCES.md"},
            "initial_required_content": [r for r in req if r != deferred], "content_sections": sections}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--exclude-groups", type=Path, default=HERE / "briefs/pilot", help="brief dir whose groups and ids must not be reused")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    used_ids = {p.stem for p in a.exclude_groups.glob("*.json")}
    fams = list(FAMILIES); biz = BUSINESSES[:]; rng.shuffle(biz)
    jobs = [(fams[i % len(fams)], biz[i % len(biz)], rng.randint(1, 10**6)) for i in range(a.n)]
    a.out.mkdir(parents=True, exist_ok=True)

    def one(job):
        fam, b, seed = job
        desc, structured = FAMILIES[fam]
        prompt = AUTHOR.format(fam_desc=desc, biz=b, structured=STRUCT_YES if structured else STRUCT_NO)
        with httpx.Client() as c:
            d = chat(c, {"model": MODEL, "max_tokens": 4000, "temperature": 1.0, "reasoning_effort": "low", "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": prompt + f"\nSeed {seed}."}]})
        try:
            j = json.loads(re.search(r"\{.*\}", d["choices"][0]["message"].get("content") or "", re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            return fam, b, None
        return fam, b, to_v2(j, fam)

    written, rejected = [], 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        for fut in as_completed([ex.submit(one, j) for j in jobs]):
            fam, b, v = fut.result()
            if v is None or v["id"] in used_ids:
                rejected += 1; continue
            used_ids.add(v["id"]); (a.out / f"{v['id']}.json").write_text(json.dumps(v, indent=2, ensure_ascii=False) + "\n"); written.append(v)
    import collections
    print(f"wrote {len(written)} briefs ({rejected} rejected) -> {a.out}; families {dict(collections.Counter(v['family'] for v in written))}; groups {len({v['group'] for v in written})}; with sections {sum(bool(v['content_sections']) for v in written)}")
    md = ["# Main briefs", "", f"{len(written)} authored briefs, fictional, CC0, validated against the v2 schema. Families and groups below; read a sample before generation.", "",
          "| id | family | group | required | section rows |", "|---|---|---|---:|---:|"]
    for v in sorted(written, key=lambda v: (v["family"], v["id"])):
        md.append(f"| {v['id']} | {v['family']} | {v['group']} | {len(v['required_content'])} | {sum(len(s['rows']) for s in v['content_sections'])} |")
    (a.out / "README.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
