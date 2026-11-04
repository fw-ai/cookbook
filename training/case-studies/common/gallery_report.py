"""Offline paired reports inspired by Sinan's skill_signal and effort_reducer.

See GALLERY_REPORTS.md for provenance, examples, and interpretation limits.
No network, model calls, or third-party dependencies.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
import random
from statistics import mean


def compare(
    rows, baseline, *, quality="pass", cost="words", margin=2.0, iterations=2000
):
    """Resample prompts together, averaging repeated draws within each prompt."""
    if not math.isfinite(margin) or not 0 <= margin <= 100 or iterations < 100:
        raise ValueError(
            "margin must be 0..100 points; iterations must be at least 100"
        )
    arms = defaultdict(dict)
    for row in rows:
        key = (str(row["id"]), row.get("draw", 0))
        arm = row["arm"]
        if key in arms[arm]:
            raise ValueError(f"duplicate id/draw in {arm}: {key}")
        if type(row.get(quality)) is not bool:
            raise ValueError(f"{arm} {key}: {quality} must be a scored boolean")
        if row.get("error") or ("judge" in row and row["judge"] is None):
            raise ValueError(
                f"{arm} {key}: unresolved scorer error; repair before comparing"
            )
        value = len(row["text"].split()) if cost == "words" else row.get(cost)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{arm} {key}: missing/invalid {cost}")
        arms[arm][key] = (float(row[quality]), float(value))
    if baseline not in arms or len(arms) < 2:
        raise ValueError("need a baseline and at least one comparison arm")
    keys = set(arms[baseline])
    ids = sorted({k[0] for k in keys})
    if len(ids) < 2:
        raise ValueError("need at least two paired prompts")
    for name, values in arms.items():
        if set(values) != keys:
            raise ValueError(
                f"{name}: id/draw coverage differs from baseline; no silent intersection"
            )
    grouped = {}
    for name, values in arms.items():
        by_id = defaultdict(list)
        for (pid, _), pair in values.items():
            by_id[pid].append(pair)
        grouped[name] = [
            (mean(x[0] for x in by_id[pid]), mean(x[1] for x in by_id[pid]))
            for pid in ids
        ]
    a = grouped[baseline]

    def interval(values):
        values = sorted(values)
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    comparisons = []
    for name in sorted(grouped):
        if name == baseline:
            continue
        b = grouped[name]
        rng = random.Random(0)
        deltas, savings = [], []
        cost_defined = all(x[1] > 0 for x in a)
        for _ in range(iterations):
            idx = rng.choices(range(len(ids)), k=len(ids))
            deltas.append(100 * mean(b[i][0] - a[i][0] for i in idx))
            if cost_defined:
                savings.append(
                    100 * (1 - sum(b[i][1] for i in idx) / sum(a[i][1] for i in idx))
                )
        ci = interval(deltas)
        verdict = (
            "preserved"
            if ci[0] >= -margin
            else ("regressed beyond margin" if ci[1] < -margin else "inconclusive")
        )
        comparisons.append(
            dict(
                arm=name,
                baseline_quality=mean(x[0] for x in a),
                quality=mean(x[0] for x in b),
                quality_delta_points=100 * mean(y[0] - x[0] for x, y in zip(a, b)),
                quality_ci_points=ci,
                preservation=verdict,
                baseline_cost=mean(x[1] for x in a),
                cost=mean(x[1] for x in b),
                savings_percent=(
                    100 * (1 - sum(x[1] for x in b) / sum(x[1] for x in a))
                    if cost_defined
                    else None
                ),
                savings_ci_percent=interval(savings) if savings else None,
            )
        )
    return dict(
        baseline=baseline,
        quality_metric=quality,
        cost_metric=cost,
        prompts=len(ids),
        draws_per_arm=len(keys),
        margin_points=margin,
        bootstrap_iterations=iterations,
        comparisons=comparisons,
    )


def render(report, rows, title):
    e = lambda x: escape(str(x), quote=True)

    def metric(value, ci):
        return (
            "undefined (zero baseline cost)"
            if value is None
            else f"{value:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]"
        )

    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{e(x)}</td>"
            for x in [
                r["arm"],
                metric(r["quality_delta_points"], r["quality_ci_points"]),
                metric(r["savings_percent"], r["savings_ci_percent"]),
                r["preservation"],
            ]
        )
        + "</tr>"
        for r in report["comparisons"]
    )
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(str(row["id"]), row.get("draw", 0))][row["arm"]] = row
    names = [report["baseline"]] + [r["arm"] for r in report["comparisons"]]
    examples = []
    for (pid, draw), arms in sorted(grouped.items()):
        cards = []
        for name in names:
            row = arms[name]
            checks = {
                k: row[k]
                for k in ("problems", "judge", "finish", "finish_reason")
                if k in row
            }
            cards.append(
                f'<article><h3>{e(name)}</h3><p>{e(report["quality_metric"])}: {e(row[report["quality_metric"]])}</p>'
                f'<pre>{e(row["text"])}</pre><details><summary>Recorded checks</summary><pre>{e(json.dumps(checks, indent=2))}</pre></details></article>'
            )
        examples.append(
            f'<details><summary>{e(pid)} · draw {e(draw)}</summary><div class="cards">{"".join(cards)}</div></details>'
        )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172538}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ccd}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}}article{{min-width:0;padding:16px;background:#f3f6f8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 monospace}}summary{{cursor:pointer;padding:12px}}details{{margin:12px 0}}</style>
<h1>{e(title)}</h1><p>{report['prompts']} paired prompts; {report['draws_per_arm']} draws per arm. Baseline: {e(report['baseline'])}.</p>
<p>95% paired prompt-bootstrap intervals; {report['bootstrap_iterations']} resamples. Quality: {e(report['quality_metric'])}; length: {e(report['cost_metric'])}.</p>
<table><tr><th>Arm</th><th>Quality change (points)</th><th>Length saved (%)</th><th>Preservation vs −{report['margin_points']:g} points</th></tr>{table}</table>
<p>The margin is an analysis choice, not proof it was selected before the experiment. Inconclusive does not establish preservation. Word savings are not measured token cost or latency. These intervals describe this fixture set and its recorded scorer, not new domains or training seeds.</p>
<p>Omitted arms: {e(', '.join(report.get('omitted_arms', [])) or 'none')}.</p><h2>Read every example</h2><p>All recorded outputs in ID order, including failures. These are evaluation examples, not fresh unseen showcase prompts. No new model calls.</p>{''.join(examples)}</html>"""


def load_rows(paths):
    return [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "samples",
        nargs="+",
        help="saved samples.jsonl; do not combine independent runs",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        help="explicit subset; omitted arms are disclosed in the report",
    )
    parser.add_argument("--quality", default="pass")
    parser.add_argument("--cost", default="words")
    parser.add_argument("--margin-points", type=float, default=2)
    parser.add_argument("--title", default="Recorded gallery comparison")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new directory; never overwrites a report",
    )
    args = parser.parse_args()
    rows = load_rows(args.samples)
    all_arms = sorted({r["arm"] for r in rows})
    if args.arm:
        if set(args.arm) - set(all_arms):
            raise ValueError("requested arm is absent")
        rows = [r for r in rows if r["arm"] in args.arm]
    result = compare(
        rows,
        args.baseline,
        quality=args.quality,
        cost=args.cost,
        margin=args.margin_points,
    )
    result["omitted_arms"] = sorted(set(all_arms) - {r["arm"] for r in rows})
    html = render(result, rows, args.title)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "index.html").write_text(html)
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
