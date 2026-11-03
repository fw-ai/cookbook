"""One static HTML page per pilot folder: every brief, every draw, desktop and mobile shots, functional verdict,
and the judge's stated reasons. For reading pages before trusting numbers. No API calls.

    python build_gallery.py --results results/pilot-v2
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def img(p: Path, w: int) -> str:
    if not p.exists():
        return f'<div class="missing" style="width:{w}px">no screenshot</div>'
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="width:{w}px">'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--briefs", type=Path, default=HERE / "briefs/pilot")
    ap.add_argument("--max-height", type=int, default=900)
    a = ap.parse_args()
    rows = json.load(open(a.results / "pilot_rows.json")) if (a.results / "pilot_rows.json").exists() else []
    byrow = {(r["brief"], r["arm"], r["draw"]): r for r in rows}
    judg = collections.defaultdict(list)
    jp = a.results / "judgments.jsonl"
    if jp.exists():
        for l in jp.read_text().splitlines():
            r = json.loads(l)
            if not r.get("error"):
                judg[r["brief"]].append(r)
    parts = ["""<!doctype html><meta charset="utf-8"><title>Pilot gallery</title><style>
    body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;background:#f4f4f2;color:#222}
    .brief{margin:40px 0;padding:16px;background:#fff;border:1px solid #ddd}.cands{display:flex;gap:16px;overflow-x:auto}
    .cand{flex:0 0 auto;border:1px solid #ccc;padding:8px;background:#fafafa}.cand h4{margin:0 0 6px;font-size:13px}
    .shots{display:flex;gap:8px;align-items:flex-start}.shots>div{overflow:hidden;max-height:%dpx;border:1px solid #ddd;background:#fff}
    .ok{color:#1a7f37}.bad{color:#b42318}.missing{height:200px;display:flex;align-items:center;justify-content:center;background:#eee;color:#888}
    .reasons{font-size:12px;color:#444;margin-top:8px}.reasons li{margin:3px 0}pre{white-space:pre-wrap;font-size:12px;color:#555}
    </style><h1>Pilot gallery</h1><p>Every draw, desktop (1440) and mobile (390), cropped to %dpx tall. Read before trusting numbers.</p>""" % (a.max_height, a.max_height)]
    for bp in sorted(a.briefs.glob("*.json")):
        b = json.loads(bp.read_text()); bd = a.results / bp.stem
        if not bd.exists():
            continue
        parts.append(f'<div class="brief"><h2>{html.escape(bp.stem)} <small>({b.get("family")}, {len(b.get("assets", []))} assets)</small></h2><pre>{html.escape(b["brief"][:600])}</pre><div class="cands">')
        for cd in sorted(bd.iterdir()):
            if not cd.is_dir() or "-" not in cd.name:
                continue
            arm, d = cd.name.rsplit("-", 1)
            r = byrow.get((bp.stem, arm, int(d)), {})
            f = r.get("functional"); cls = "ok" if f else "bad"
            verdict = "functional" if f else ("not rendered" if not r.get("rendered") else "FAILS: " + ", ".join(k for k in ("missing", "section_failures") if r.get(k)) + (" interactions " + str(r.get("interactions")) if r.get("interactions") and r["interactions"].split("/")[0] != r["interactions"].split("/")[1] else ""))
            parts.append(f'<div class="cand"><h4>{html.escape(cd.name)} <span class="{cls}">{html.escape(verdict)}</span> <small>{r.get("tokens")} tok</small></h4><div class="shots"><div>{img(cd / "render/shots/1440.png", 560)}</div><div>{img(cd / "render/shots/390.png", 180)}</div></div></div>')
        parts.append("</div>")
        if judg[bp.stem]:
            parts.append('<div class="reasons"><b>Judge reasons</b><ul>')
            for r in judg[bp.stem][:12]:
                parts.append(f'<li>[{html.escape(r["judge_model"].split("/")[-1])}] {html.escape(r["x"])} vs {html.escape(r["y"])}: overall <b>{r["overall"]}</b> ({"a" if not r["swapped"] else "b"} shown first) — {html.escape(r.get("reason", ""))}</li>')
            parts.append("</ul></div>")
        parts.append("</div>")
    out = a.results / "gallery.html"
    out.write_text("\n".join(parts)); print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
