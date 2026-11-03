"""A compact showcase page: for N briefs, the best functional plain draw and the best functional skill draw side by side
(desktop + mobile, downscaled JPEG), with the brief and the human/judge notes. Small enough to share. No API calls.

    python build_showcase.py --results results/pilot-v2 --out results/showcase.html --n 8
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jpeg(p: Path, width: int, max_h: int) -> str:
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im = im.crop((0, 0, im.width, min(im.height, int(max_h * im.width / width))))
    im = im.resize((width, int(im.height * width / im.width)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--briefs", type=Path, default=HERE / "briefs/pilot"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--adapter-results", type=Path, default=None, help="evaluate_adapter output folder; adds an adapter column")
    ap.add_argument("--adapter-label", default="LoRA, plain prompt, no skill text")
    a = ap.parse_args()
    rows = json.load(open(a.results / "pilot_rows.json"))
    if a.adapter_results:
        for bd in sorted(p for p in a.adapter_results.iterdir() if p.is_dir()):
            for cd in sorted(p for p in bd.iterdir() if p.name.startswith("adapter-")):
                rec = json.loads((cd / "render/record.json").read_text()) if (cd / "render/record.json").exists() else {}
                g = json.loads((cd / "generation.json").read_text()) if (cd / "generation.json").exists() else {}
                rows.append({"brief": bd.name, "arm": "adapter", "draw": int(cd.name.split("-")[1]), "functional": rec.get("passes_functional"), "rendered": rec.get("rendered"),
                             "tokens": (g.get("usage") or {}).get("completion_tokens"), "_dir": str(cd / "render/shots")})
    ids = a.ids or sorted({r["brief"] for r in rows})[: a.n]
    parts = ["""<!doctype html><meta charset="utf-8"><title>Open Weight Room: UI design pilot</title><style>
    body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px auto;max-width:1700px;padding:0 16px;background:#f4f4f2;color:#222}
    .brief{background:#fff;border:1px solid #ddd;padding:16px;margin:28px 0}h2{margin:0 0 6px}.cols{display:flex;gap:16px}.col{flex:1}
    .col h4{margin:6px 0}.shots{display:flex;gap:8px}.shots img{border:1px solid #ddd;background:#fff}pre{white-space:pre-wrap;font-size:12px;color:#555}
    .note{font-size:12px;color:#444;margin-top:6px}</style>
    <h1>UI design LoRA: 24 briefs, three ways</h1>
    <p>Every brief in the held-out test set, rendered by GLM 5.3 Flash three ways: the plain prompt, the plain prompt plus the 20k-token Taste Skill, and
    the LoRA <code>glm53-uidesign-v1</code> with the plain prompt and no skill text. Same scaffold, effort low, no images unless the brief supplied one.
    First functional draw of each arm (or first rendered draw when none passed). Desktop 1440 and mobile 390, cropped to the top. Nothing was edited.</p>
    <p style="background:#fff8e6;padding:10px 14px;border-left:3px solid #e1a21a">What to look for: the LoRA was trained on 105 pages distilled from the skill. On blind human review of 20 other held-out briefs it tied the plain model for first choice and beat the skill prompt 18 of 20. Its habit is editorial and roomy; on dense briefs the plain page is sometimes the more compact one.</p>"""]
    for b in ids:
        brief = json.loads((a.briefs / f"{b}.json").read_text())
        best = {}
        arms = ["plain", "skill"] + (["adapter"] if a.adapter_results else [])
        for arm in arms:
            cands = [r for r in rows if r["brief"] == b and r["arm"] == arm and r["functional"]] or [r for r in rows if r["brief"] == b and r["arm"] == arm and r["rendered"]]
            if cands:
                best[arm] = sorted(cands, key=lambda r: (not r["functional"], r["draw"]))[0]
        if len(best) < 2:
            continue
        parts.append(f'<div class="brief"><h2>{b}</h2><pre>{brief["brief"][:420]}</pre><div class="cols">')
        for arm, label in [("plain", "Plain prompt"), ("skill", "Taste Skill prompt (20k tokens)")] + ([("adapter", a.adapter_label)] if "adapter" in best else []):
            d = Path(best[arm]["_dir"]) if "_dir" in best[arm] else a.results / b / f"{arm}-{best[arm]['draw']}/render/shots"
            parts.append(f'<div class="col"><h4>{label}</h4><div class="shots"><img src="{jpeg(d / "1440.png", 400, 1300)}" width="400"><img src="{jpeg(d / "390.png", 120, 1300)}" width="120"></div>'
                         f'<div class="note">functional checks: {"pass" if best[arm]["functional"] else "fail"} · {best[arm]["tokens"]} tokens</div></div>')
        parts.append("</div></div>")
    a.out.write_text("\n".join(parts)); print("wrote", a.out, f"{a.out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
