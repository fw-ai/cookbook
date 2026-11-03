"""Blind gallery for a compare_arms results dir: per brief, every arm's draws in a seeded random order labelled A, B, C...
with the arm names hidden. Writes gallery_blind.html (reveal toggle, scored by you in the page: pick the best per brief,
export your picks) and gallery_key.json (label -> arm per brief). No API calls.

    python build_blind_gallery.py --results results/compare-rl --arms skill,v2b,rl --seed 7
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def img(p: Path, w: int) -> str:
    if not p.exists():
        return f'<div class="missing" style="width:{w}px">no screenshot</div>'
    return f'<img loading="lazy" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="width:{w}px">'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--brief-dir", type=Path, default=HERE / "briefs/main"); ap.add_argument("--briefs", type=Path, default=None, help="JSON list of brief ids; default: every brief dir in results")
    ap.add_argument("--draw", type=int, default=0, help="which draw index to show (one per arm keeps the page blind and short)")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--max-height", type=int, default=820)
    a = ap.parse_args()
    arms = a.arms.split(",")
    ids = json.loads(a.briefs.read_text()) if a.briefs else sorted(p.name for p in a.results.iterdir() if p.is_dir())
    rng = random.Random(a.seed); key = {}
    css = """body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;background:#f4f4f2;color:#222}
    .brief{margin:40px 0;padding:16px;background:#fff;border:1px solid #ddd}.cands{display:flex;gap:16px;overflow-x:auto}
    .cand{flex:0 0 auto;border:1px solid #ccc;padding:8px;background:#fafafa}.cand h4{margin:0 0 6px;font-size:14px}
    .shots{display:flex;gap:8px;align-items:flex-start}.shots>div{overflow:hidden;max-height:%dpx;border:1px solid #ddd;background:#fff}
    .ok{color:#1a7f37}.bad{color:#b42318}.missing{height:200px;display:flex;align-items:center;justify-content:center;background:#eee;color:#888}
    pre{white-space:pre-wrap;font-size:12px;color:#555}.arm{display:none;color:#06c;font-weight:600}body.reveal .arm{display:inline}
    .pick label{margin-right:12px;font-size:13px}#bar{position:sticky;top:0;background:#fff;padding:10px;border-bottom:1px solid #ddd;z-index:2}
    textarea{width:100%%;height:80px;font-size:12px}""" % a.max_height
    parts = [f'<!doctype html><meta charset="utf-8"><title>Blind gallery</title><style>{css}</style>',
             '<div id="bar"><b>Blind gallery</b> — one draw per arm, order shuffled per brief. Pick the best page per brief, then '
             '<button onclick="document.body.classList.toggle(\'reveal\')">reveal arms</button> '
             '<button onclick="exportPicks()">export picks</button><textarea id="out" placeholder="picks appear here as JSON"></textarea></div>']
    for b in ids:
        bp = a.brief_dir / f"{b}.json"
        if not bp.exists() or not (a.results / b).exists():
            continue
        brief = json.loads(bp.read_text()); order = arms[:]; rng.shuffle(order)
        labels = [chr(65 + i) for i in range(len(order))]; key[b] = dict(zip(labels, order))
        parts.append(f'<div class="brief" data-brief="{html.escape(b)}"><h2>{html.escape(b)} <small>({brief.get("family")})</small></h2><pre>{html.escape(brief["brief"][:500])}</pre><div class="cands">')
        for lab, arm in zip(labels, order):
            cd = a.results / b / f"{arm}-{a.draw}"; rec = json.loads((cd / "render/record.json").read_text()) if (cd / "render/record.json").exists() else {}
            fc = json.loads((cd / "render/factcheck.json").read_text()) if (cd / "render/factcheck.json").exists() else {}
            f = rec.get("passes_functional"); verdict = "functional" if f else ("not rendered" if not rec.get("rendered") else "fails functional")
            claims = len(fc.get("policy_claims", [])) if fc else "?"
            parts.append(f'<div class="cand"><h4>{lab} <span class="arm">[{html.escape(arm)}]</span> <span class="{"ok" if f else "bad"}">{verdict}</span> <small>{claims} unsupported claims</small></h4>'
                         f'<div class="shots"><div>{img(cd / "render/shots/1440.png", 560)}</div><div>{img(cd / "render/shots/390.png", 180)}</div></div></div>')
        parts.append('</div><div class="pick">Best: ' + "".join(f'<label><input type="radio" name="{html.escape(b)}" value="{l}"> {l}</label>' for l in labels) + '</div></div>')
    parts.append('<script>function exportPicks(){const o={};document.querySelectorAll(".brief").forEach(d=>{const c=d.querySelector("input:checked");if(c)o[d.dataset.brief]=c.value});document.getElementById("out").value=JSON.stringify(o)}</script>')
    out = a.results / "gallery_blind.html"; out.write_text("\n".join(parts))
    (a.results / "gallery_key.json").write_text(json.dumps(key, indent=1))
    print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB; key in gallery_key.json ({len(key)} briefs)")


if __name__ == "__main__":
    main()
