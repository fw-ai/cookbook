"""Live playground and recorded views for lipogram_demo.ipynb.

`ask_lipogram` calls the served adapter (and, optionally, the base model with and without the rule) and runs
rl_lipogram's strict checker on every answer. Live inference requires both LIPO_ALLOW_LIVE=1 and FIREWORKS_API_KEY. Otherwise it shows
recorded/playground.json. A configured key alone never enables live calls.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RL = HERE.parent / "rl_lipogram"
for p in (HERE.parent / "common", RL, HERE):
    sys.path.insert(0, str(p))
import figures as F  # noqa: E402
from audit_unicode import leet_tokens  # noqa: E402
from build_prompts import SYSTEM_PROMPT  # noqa: E402
from lipogram_rewards import _canonical, _E_LOOKALIKE, _ESCAPED_E, _INVISIBLE, count_e, e_free  # noqa: E402

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover
    HTML = display = None

RECORDED = HERE / "recorded"
PLAYGROUND = RECORDED / "playground.json"
CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
BASE = "accounts/fireworks/models/glm-5p3-flash"
ADAPTER = os.environ.get("LIPO_ADAPTER", "accounts/research-train/models/arxm-lipo-p2-rl-s100")
SFT = os.environ.get("LIPO_SFT", "accounts/research-train/models/arxm-lipo-p2-lr1e4-e1")

# label -> (model, system prompt or None)
ARMS = {
    "RL adapter s100, no instruction": (ADAPTER, None),
    "Base, no instruction": (BASE, None),
    "Base + lipogram system prompt": (BASE, SYSTEM_PROMPT),
    "SFT adapter (lr1e4-e1), no instruction": (SFT, None),
}

CSS = """<style>
.lp-card{border:1px solid #d0d3d9;border-radius:6px;margin:8px 0;font-family:-apple-system,Segoe UI,sans-serif;font-size:13.5px}
.lp-card .lp-head{padding:6px 12px;background:#e9ebef;color:#333;font-weight:600;border-bottom:1px solid #d0d3d9}
.lp-card.live .lp-head{background:#fdebd9;border-bottom-color:#e1812c;color:#7a3f0c}
.lp-card .lp-block{padding:8px 12px;border-bottom:1px solid #e3e5e9;white-space:pre-wrap;line-height:1.45}
.lp-card .lp-block:last-child{border-bottom:none}
.lp-card .lp-label{display:block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#666;margin-bottom:3px}
.lp-card mark{background:#f9c9c9;padding:0 1px;font-weight:600}
.lp-pass{color:#1a7f37;font-weight:700}.lp-fail{color:#c03d3e;font-weight:700}
</style>"""


def _load_key_from_file() -> None:
    """If FIREWORKS_API_KEY is unset (e.g. a notebook kernel without your shell's environment), read the key
    from the file named by FIREWORKS_API_KEY_FILE."""
    path = os.environ.get("FIREWORKS_API_KEY_FILE")
    if os.environ.get("FIREWORKS_API_KEY") or not path:
        return
    p = Path(path).expanduser()
    if p.is_file() and p.read_text().strip():
        os.environ["FIREWORKS_API_KEY"] = p.read_text().strip()


def live() -> bool:
    if os.environ.get("LIPO_ALLOW_LIVE") != "1":
        return False
    _load_key_from_file()
    return bool(os.environ.get("FIREWORKS_API_KEY")) and not os.environ.get("LIPO_OFFLINE")


def _offending(ch: str) -> bool:
    if ch in "eE":
        return True
    if ord(ch) < 128 or _INVISIBLE.fullmatch(ch):
        return False
    canon = _E_LOOKALIKE.sub("e", unicodedata.normalize("NFKD", ch))
    if "e" in canon or "E" in canon:
        return True
    return ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN")


def mark_e(text: str) -> str:
    """HTML with every e, accented or look-alike e, escaped e and non-Latin letter marked [x] (what the checker counts)."""
    out, i = [], 0
    for m in _ESCAPED_E.finditer(text):
        out.append(("plain", text[i:m.start()]))
        out.append(("hit", m.group(0)))
        i = m.end()
    out.append(("plain", text[i:]))
    parts = []
    for kind, chunk in out:
        if kind == "hit":
            parts.append(f"<mark>[{html.escape(chunk)}]</mark>")
            continue
        for ch in chunk:
            if _offending(ch):
                parts.append(f"<mark>[{html.escape(ch)}]</mark>")
            else:
                parts.append(html.escape(ch))
    return "".join(parts)


def check(text: str, prompt: str = "") -> dict:
    es, letters = count_e(text)
    return {"e_free": e_free(text), "e_count": es, "words": len(text.split()), "letters": letters,
            "leet": leet_tokens(text, prompt), "ascii_e": len(re.findall("[eE]", text)),
            "disguised": len(re.findall("[eE]", _canonical(text))) - len(re.findall("[eE]", text))}


def _chat(model: str, prompt: str, system: str | None, effort: str, temperature: float, max_tokens: int) -> str:
    import httpx
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "reasoning_effort": effort}
    headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
    for attempt in range(4):
        r = httpx.post(CHAT_URL, headers=headers, json=body, timeout=600)
        if r.status_code == 404:
            acct, mid = model.split("/")[1], model.split("/")[-1]
            raise LookupError(f"not loaded on serverless for this key (firectl -a {acct} model load-lora {mid} --serverless)")
        if r.status_code < 500 and r.status_code != 429:
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        time.sleep(2 ** attempt * 3)
    r.raise_for_status()
    return ""


def _load_recorded() -> list[dict]:
    return json.loads(PLAYGROUND.read_text()) if PLAYGROUND.exists() else []


def _save_recorded(rows: list[dict]) -> None:
    rec = {(r["prompt"], r["arm"], r["effort"], r["temperature"]): r for r in _load_recorded()}
    rec.update({(r["prompt"], r["arm"], r["effort"], r["temperature"]): r for r in rows})
    PLAYGROUND.write_text(json.dumps(list(rec.values()), indent=1, ensure_ascii=False) + "\n")


def _card(prompt: str, rows: list[dict], source: str) -> None:
    blocks = [f'<div class="lp-block"><span class="lp-label">Prompt</span>{html.escape(prompt)}</div>']
    for r in rows:
        if r.get("error"):
            blocks.append(f'<div class="lp-block"><span class="lp-label">{html.escape(r["arm"])}</span><i>{html.escape(r["error"])}</i></div>')
            continue
        c = r["check"]
        badge = '<span class="lp-pass">PASS</span>' if c["e_free"] else '<span class="lp-fail">FAIL</span>'
        extra = f" · leet? {', '.join(c['leet'])}" if c["leet"] else ""
        label = f"{html.escape(r['arm'])} · {badge} · {c['e_count']} e · {c['words']} words{html.escape(extra)}"
        blocks.append(f'<div class="lp-block"><span class="lp-label">{label}</span>{mark_e(r["response"])}</div>')
    kind = "live" if source.startswith("live") else "recorded"
    head = f"{source} · {rows[0]['model'].split('/')[-1] if rows else ''} · effort {rows[0]['effort'] if rows else ''}"
    if display:
        display(HTML(f'{CSS}<div class="lp-card {kind}"><div class="lp-head">{html.escape(head)}</div>{"".join(blocks)}</div>'))
    for r in rows:
        if not r.get("error"):
            c = r["check"]
            print(f"{'PASS' if c['e_free'] else 'FAIL'}  {r['arm']}: {c['e_count']} e (strict), {c['words']} words")


def ask_lipogram(prompt: str, effort: str = "low", temperature: float = 0.0, compare_all: bool = True,
                 max_tokens: int = 2048, arms: list[str] | None = None) -> list[dict]:
    """Ask the RL adapter (and, with compare_all, the comparison arms) and check every answer strictly.

    Live when FIREWORKS_API_KEY is set; otherwise shows the recorded answer for the same prompt and settings.
    An arm whose model is not serving for your key is reported and skipped.
    """
    names = arms or (list(ARMS) if compare_all else [next(iter(ARMS))])
    if not live():
        rec = {(r["prompt"], r["arm"], r["effort"], r["temperature"]): r for r in _load_recorded()}
        rows = [rec[k] for k in ((prompt, n, effort, temperature) for n in names) if k in rec]
        if not rows:
            print("No FIREWORKS_API_KEY and no recorded answer for this prompt. Set RUN_LIVE = True and supply a key to call the model live.")
            return []
        _card(prompt, rows, f"recorded {rows[0].get('recorded_at', '')[:10]} (set RUN_LIVE = True and supply FIREWORKS_API_KEY to run live)")
        return rows

    def one(name: str) -> dict:
        model, system = ARMS[name]
        row = {"prompt": prompt, "arm": name, "model": model, "effort": effort, "temperature": temperature,
               "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            row["response"] = _chat(model, prompt, system, effort, temperature, max_tokens)
            row["check"] = check(row["response"], prompt)
        except LookupError as e:
            row["error"] = f"skipped, {e}"
        return row

    with ThreadPoolExecutor(len(names)) as ex:
        rows = list(ex.map(one, names))
    if os.environ.get("LIPO_RECORD"):
        _save_recorded([r for r in rows if not r.get("error")])
    _card(prompt, rows, "live")
    return rows


def playground() -> None:
    """A text box and a button (needs ipywidgets and a live key)."""
    try:
        import ipywidgets as w
    except ImportError:
        print("pip install ipywidgets for the text box, or call ask_lipogram('...') directly.")
        return
    box = w.Textarea(placeholder="Ask anything...", layout=w.Layout(width="100%", height="70px"))
    compare = w.Checkbox(value=False, description="compare with base")
    go, out = w.Button(description="Ask", button_style="warning"), w.Output()

    def on_click(_):
        out.clear_output()
        with out:
            ask_lipogram(box.value, compare_all=compare.value)
    go.on_click(on_click)
    display(w.VBox([box, w.HBox([go, compare]), out]))


# ---- recorded views (no network) ----

def load() -> tuple[dict, dict, dict]:
    return (json.loads((RECORDED / "demo_cases.json").read_text()),
            json.loads((RECORDED / "all_checkpoints.json").read_text()),
            json.loads((RECORDED / "unicode_audit.json").read_text()))


def results_table(demo: dict) -> None:
    lines = ["| model (effort low, no instruction unless noted) | greedy success | sampled success | sampled e-free | completeness | retention |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in demo["metrics"]:
        lines.append(f"| {m['label']} | {m['greedy_success']:.1%} | {m['sampled_success']:.1%} | {m['sampled_e_free']:.1%} | "
                     f"{m['sampled_completeness']:.2f} | {m['retention']:.1%} |")
    _md("\n".join(lines) + "\n\nBar: sampled ≥ 90%, greedy ≥ 95%, retention loss ≤ 2 points. Bar met: **"
        + ("yes" if demo["bar_met"] else "no") + "**.")


def progression_table(demo: dict) -> None:
    lines = ["| RL step | greedy | sampled | e-free greedy / sampled | e per 100 letters | answered | completeness | median words | retention | in-session validation |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in demo["rl_progression"]:
        v = p.get("in_session_validation")
        val = f"{v['success']:.1%}" if v else ""
        lines.append(f"| {p['rl_step']} | {p['greedy_success']:.1%} | {p['sampled_success']:.1%} | {p['greedy_e_free']:.1%} / {p['sampled_e_free']:.1%} | "
                     f"{p['sampled_e_per_100_letters']:.3f} | {p['sampled_answered']:.1%} | {p['sampled_completeness']:.2f} | "
                     f"{p['median_words_sampled']:g} | {p['retention']:.1%} | {val} |")
    _md("\n".join(lines) + "\n\nStep 0 is the SFT start. Sampled metrics unless marked. In-session validation: 16 of the "
        "eval prompts × 8 samples, strict reward, logged by `lipogram_rl.py`.")


def progression_figure(ckpt: dict, path: Path):
    import matplotlib.pyplot as plt
    F.style()
    order = [("SFT\npilot", "p1-e2"), ("SFT\n10.5k", "p2-lr1e4-e1"), ("RL\ns20", "p2-rl-s20"), ("RL\ns40", "p2-rl-s40"),
             ("RL\ns60", "p2-rl-s60"), ("RL\ns80", "p2-rl-s80"), ("RL\ns100", "p2-rl-s100")]
    xs = range(len(order))
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label in [("greedy_success", "greedy joint success"), ("sampled_success", "sampled joint success"),
                       ("sampled_e_free", "sampled e-free")]:
        ys = [100 * ckpt[t][key] for _, t in order]
        ax.plot(xs, ys, marker="o", label=label, color=F.color_for(label, len(ax.lines)))
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.axhline(90, ls="--", color="#c03d3e", lw=1)
    ax.text(len(order) - 1, 91.5, "bar: sampled 90%", color="#c03d3e", ha="right", fontsize=8)
    ax.set_xticks(list(xs), [o[0] for o in order])
    ax.set_ylabel("% of held-out answers")
    ax.set_ylim(0, 100)
    ax.set_title("No-e answers at effort low, no instruction (60 held-out prompts)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    F.save(fig, path)
    return fig


NAMES = {"base": "Base, no instruction", "base_prompt": "Base + lipogram system prompt", "rob_sft": "Rob's sft-v2-300",
         "our_sft": "SFT (lr1e4-e1)", "our_rl_s60": "RL s60", "our_rl": "RL s100"}


def case_card(case: dict) -> None:
    blocks = [f'<div class="lp-block"><span class="lp-label">Prompt ({html.escape(case["family"])})</span>{html.escape(case["question"])}</div>']
    for k, name in NAMES.items():
        o = case["outputs"].get(k)
        if not o:
            continue
        badge = '<span class="lp-pass">PASS</span>' if o["success"] else '<span class="lp-fail">FAIL</span>'
        blocks.append(f'<div class="lp-block"><span class="lp-label">{html.escape(name)} · {badge} · {o["e_count"]} e · '
                      f'{o["words"]} words</span>{mark_e(o["response"][:900])}{"…" if len(o["response"]) > 900 else ""}</div>')
    if display:
        display(HTML(f'{CSS}<div class="lp-card"><div class="lp-head">{html.escape(case["why"])} · recorded, greedy, effort low</div>{"".join(blocks)}</div>'))


def audit_table(audit: dict) -> None:
    rows = audit["eval_samples_audit"]
    keep = ["p2-rl-s100 / adapter", "p2-rl-s80 / adapter", "p2-rl-s60 / adapter", "p2-lr1e4-e1 / adapter",
            "rob-sft-v2-300 / adapter", "GLM 5.3 Flash base, pooled over 12 runs / base+prompt",
            "GLM 5.3 Flash base, pooled over 12 runs / base+mask"]
    lines = ["| model / arm | answers | no ASCII e | e-free (strict checker) | accented e | look-alike e | non-ASCII letter | e escape | invisible | leet |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keep:
        m = rows[k]
        lines.append(f"| {k} | {m['responses']} | {m['ascii_e_free']} | {m['rob_checker_e_free']} | {m['with_nfkd_e']} | {m['with_homoglyph']} | "
                     f"{m['with_non_ascii_letter']} | {m['with_e_escape']} | {m['with_invisible']} | {m['with_leet_candidate']} |")
    lines += ["", "| s100 adversarial probe (no instruction) | answers | e-free (strict) | success | accented e | look-alike e | non-ASCII letter | leet |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, m in audit["adversarial_probe_s100"]["sets"].items():
        lines.append(f"| {k} | {m['responses']} | {m['rob_checker_e_free']} | {m['success']} | {m['with_nfkd_e']} | {m['with_homoglyph']} | "
                     f"{m['with_non_ascii_letter']} | {m['with_leet']} |")
    _md("\n".join(lines))


def _md(text: str) -> None:
    from IPython.display import Markdown
    display(Markdown(text)) if display else print(text)
