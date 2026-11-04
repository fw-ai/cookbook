"""Evidence cards for the Humanizer notebook.

A card shows a passage and one or more rewrites with the changed words highlighted.
Grey cards hold recorded evidence from the committed snapshot; orange cards (``kind="live"``)
describe the reader's own run. Outside Jupyter the same functions print plain text, so
the notebook's cells can be executed by tests without a display.
"""

from __future__ import annotations

import difflib
import html
import re

try:  # Inside a Jupyter kernel we render HTML; anywhere else we print.
    from IPython import get_ipython
    from IPython.display import HTML, display

    IN_NOTEBOOK = get_ipython() is not None
except ImportError:  # pragma: no cover - plain Python without IPython installed
    IN_NOTEBOOK = False

ARM_LABELS = {
    "base_plain": "Base, plain prompt",
    "base_short": "Base, editor instruction",
    "skill": "Prompted, full skill text",
    "adapter_s29": "Adapter, editor instruction",
}

CSS = """<style>
.hz-card{border:1px solid #d0d3d9;border-radius:6px;margin:8px 0;font-family:-apple-system,Segoe UI,sans-serif;font-size:13.5px}
.hz-card .hz-head{padding:6px 12px;background:#e9ebef;color:#333;font-weight:600;border-bottom:1px solid #d0d3d9}
.hz-card.live .hz-head{background:#fdebd9;border-bottom-color:#e1812c;color:#7a3f0c}
.hz-card .hz-note{padding:4px 12px;color:#555;font-size:12px;border-bottom:1px solid #e3e5e9}
.hz-card .hz-block{padding:8px 12px;border-bottom:1px solid #e3e5e9;white-space:pre-wrap;line-height:1.45}
.hz-card .hz-block:last-child{border-bottom:none}
.hz-card .hz-label{display:block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:#666;margin-bottom:3px}
.hz-card mark{background:#fde6b6;padding:0 1px}
</style>"""


def recorded_note(lesson: dict) -> str:
    """The provenance line printed under every recorded card."""
    return (
        f"Recorded: {lesson['base_model'].split('/')[-1]} base, {lesson['recorded_date']}, "
        f"seed {lesson['seed']}, {len(lesson['cases'])} held-out passages, one greedy draw each"
    )


def highlight(source: str, text: str) -> str:
    """Mark the words of ``text`` that are not in ``source``; whitespace tokens keep paragraph breaks."""
    a, b = re.split(r"(\s+)", source), re.split(r"(\s+)", text)
    out = []
    for op, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        seg = "".join(b[j1:j2])
        if op == "equal" or not seg.strip():
            out.append(html.escape(seg))
        else:
            out.append("<mark>" + html.escape(seg) + "</mark>")
    return "".join(out)


def card(title: str, blocks, note: str = "", kind: str = "recorded") -> None:
    """Render one card. ``blocks`` holds (label, text) or (label, text, source_to_diff_against)."""
    if not IN_NOTEBOOK:
        print(f"== {title}" + (f"  [{note}]" if note else ""))
        for label, text, *_ref in blocks:
            print(f"-- {label}\n{text}\n")
        return
    body = "".join(
        f'<div class="hz-block"><span class="hz-label">{html.escape(label)}</span>'
        f"{highlight(ref[0], text) if ref else html.escape(text)}</div>"
        for label, text, *ref in blocks
    )
    note_html = f'<div class="hz-note">{html.escape(note)}</div>' if note else ""
    display(
        HTML(
            f'{CSS}<div class="hz-card {kind}"><div class="hz-head">{html.escape(title)}</div>'
            f"{note_html}{body}</div>"
        )
    )


def compare_card(cases: dict, case_id: str, arms=("skill", "adapter_s29"), note: str = "") -> None:
    """Source passage plus each arm's rewrite, with the arm's verdict in the block label."""
    c = cases[case_id]
    blocks = [("Source", c["source"])]
    for arm in arms:
        r = c["arms"][arm]
        verdict = "passes" if r["pass"] else ("fails the judge" if r["preserve"] else "fails a mechanical check")
        if r["text"].strip() == c["source"].strip():
            verdict += ", left unchanged"
        blocks.append((f"{ARM_LABELS[arm]}: {verdict}", r["text"], c["source"]))
    card(f"Passage {case_id} ({c['cls']}, {c['genre']}, audience: {c['audience']})", blocks, note=note)
