"""A 30-pair calibration sheet for a human: side-by-side desktop+mobile shots, blind labels, verdict hidden behind a
click, and a form that saves choices to localStorage and exports JSON. Pairs are chosen where the judges disagreed or
flipped with order (the hard cases), plus clear-cut ones as anchors. No API calls.

    python build_calibration.py --results results/pilot-v2 --n 30
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    J = [json.loads(l) for l in (a.results / "judgments.jsonl").read_text().splitlines()]
    J = [j for j in J if j.get("kind") == "cross" and not j.get("error")]
    by = collections.defaultdict(list)
    for j in J:
        by[(j["brief"], j["x"], j["y"])].append(j)
    scored = []
    for key, js in by.items():
        v = collections.Counter(j["overall"] for j in js)
        kimi = [j["overall"] for j in js if "kimi" in j["judge_model"]]; glm = [j["overall"] for j in js if "glm" in j["judge_model"]]
        agreement = (max(v.values()) / len(js)) if js else 0
        scored.append((agreement, key, {"kimi": collections.Counter(kimi), "glm": collections.Counter(glm), "reasons": [j["reason"] for j in js[:2]]}))
    rng = random.Random(a.seed); rng.shuffle(scored)
    hard = sorted(scored, key=lambda s: s[0])[: a.n * 2 // 3]
    easy = sorted(scored, key=lambda s: -s[0])[: a.n - len(hard)]
    picks = hard + easy; rng.shuffle(picks)
    briefs = {p.stem: json.loads(p.read_text()) for p in (HERE / "briefs/pilot").glob("*.json")}

    def img(p, w):
        return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="width:{w}px">' if p.exists() else "<i>missing</i>"
    parts = ["""<!doctype html><meta charset="utf-8"><title>Calibration</title><style>
    body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;background:#f4f4f2}.pair{background:#fff;border:1px solid #ddd;padding:16px;margin:32px 0}
    .cands{display:flex;gap:24px}.cand{flex:1}.shots{display:flex;gap:8px}.shots>div{max-height:820px;overflow:hidden;border:1px solid #ddd;background:#fff}
    .form label{margin-right:14px}.hidden{display:none}button{margin-top:8px}pre{white-space:pre-wrap;font-size:12px;color:#555}
    .verdict{font-size:12px;color:#444;background:#fff8e6;padding:8px;margin-top:8px}
    </style><h1>Calibration: 30 pairs</h1><p>For each pair pick the page a real client would ship, on the brief. Choose BEFORE revealing the judges. Choices save in this browser; use Export at the bottom.</p>"""]
    for i, (agreement, (brief, x, y), info) in enumerate(picks):
        b = briefs[brief]; swap = rng.random() < 0.5
        left, right = (y, x) if swap else (x, y)
        bd = a.results / brief
        parts.append(f'<div class="pair" id="p{i}"><h3>Pair {i + 1}: {html.escape(brief)}</h3><pre>{html.escape(b["brief"][:500])}\nRequired: {html.escape("; ".join(b["required_content"]))}</pre><div class="cands">')
        for label, cand in (("Left", left), ("Right", right)):
            parts.append(f'<div class="cand"><h4>{label}</h4><div class="shots"><div>{img(bd / cand / "render/shots/1440.png", 520)}</div><div>{img(bd / cand / "render/shots/390.png", 160)}</div></div></div>')
        parts.append(f'</div><div class="form">Your choice: <label><input type="radio" name="c{i}" value="left" onchange="save()">Left</label><label><input type="radio" name="c{i}" value="right" onchange="save()">Right</label><label><input type="radio" name="c{i}" value="tie" onchange="save()">Tie</label>'
                     f'<input type="text" placeholder="one reason" id="r{i}" size="70" onchange="save()"> <button onclick="reveal({i})">Reveal judges</button>'
                     f'<div class="verdict hidden" id="v{i}">Left = {html.escape(left)}, Right = {html.escape(right)}. Kimi: {dict(info["kimi"])}; GLM: {dict(info["glm"])} (a = {html.escape(x)}, b = {html.escape(y)}). '
                     f'Judge reasons: {html.escape(" / ".join(info["reasons"]))}</div></div></div>')
        parts.append(f'<script>window._map=window._map||{{}};window._map[{i}]={{left:"{left}",right:"{right}",brief:"{brief}"}};</script>')
    parts.append("""<button onclick="exportAll()">Export choices (JSON)</button><pre id="out"></pre><script>
    function save(){const o={};for(const k in window._map){const c=document.querySelector(`input[name=c${k}]:checked`);o[k]={...window._map[k],choice:c?c.value:null,reason:document.getElementById('r'+k).value};}localStorage.setItem('calib',JSON.stringify(o));}
    function reveal(i){document.getElementById('v'+i).classList.remove('hidden');}
    function exportAll(){save();document.getElementById('out').textContent=localStorage.getItem('calib');}
    (function(){const s=localStorage.getItem('calib');if(!s)return;const o=JSON.parse(s);for(const k in o){if(o[k].choice){const e=document.querySelector(`input[name=c${k}][value=${o[k].choice}]`);if(e)e.checked=true;}document.getElementById('r'+k).value=o[k].reason||'';}})();
    </script>""")
    out = a.results / "calibration.html"
    out.write_text("\n".join(parts)); print("wrote", out, f"{out.stat().st_size/1e6:.0f} MB; {len(hard)} hard + {len(easy)} anchor pairs")


if __name__ == "__main__":
    main()
