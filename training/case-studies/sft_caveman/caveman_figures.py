"""Loaders, interval helpers and plot boilerplate for ``caveman_sft.ipynb``.

Everything here reads committed files under ``results/`` and draws with ``../common/figures.py``.
Nothing calls a model or the network. The notebook keeps the one-line calls; this module keeps the
plumbing so the first code cell stays short.

Terms (also in the notebook glossary): an *arm* is one way of answering the same fixtures (base
model, base plus the skill prompt, or the adapter); a *fixture* is one held-out question with its
facts; ``pass`` is the 0/1 verdict of the shipped checker per fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path.cwd()
if not (HERE / "manifest.json").exists():
    raise SystemExit("Open the notebook from the sft_caveman folder (manifest.json was not found in the cwd).")
sys.path.insert(0, str(HERE.parent / "common"))

from figures import (  # noqa: E402  (path set above)
    COLORS,
    arm_bars,
    bootstrap_ci,
    color_for,
    delta_bars,
    paired_delta_ci,
    save,
    style,
    two_panel,
    wilson_ci,
)

__all__ = [  # re-exported from ../common/figures.py for the notebook
    "COLORS", "arm_bars", "bootstrap_ci", "color_for", "delta_bars", "paired_delta_ci", "save", "style",
    "two_panel", "wilson_ci",
]

style()
COLORS["concise"] = "#7b5ea7"  # purple: the one-line "Answer concisely." control, used only in section 8

ARM_LABELS = ["Base", "Prompted", "Adapter"]


def recorded(what: str) -> None:
    """Every output that comes from results/ announces itself, so it is never mistaken for a live run."""
    print(f"RECORDED RESULT  {what}")


def load_arms(run: str) -> dict[str, dict[str, dict]]:
    """``<run>/samples.jsonl`` -> ``{arm: {fixture id: row}}``, first draw only."""
    rows = [json.loads(line) for line in (HERE / run / "samples.jsonl").read_text().splitlines()]
    return {
        arm: {r["id"]: r for r in rows if r["arm"] == arm and r.get("draw", 0) == 0}
        for arm in dict.fromkeys(r["arm"] for r in rows)
    }


def adapter_of(arms: dict[str, dict]) -> dict:
    """The adapter arm of a run, whatever seed suffix it carries."""
    return next(v for k, v in arms.items() if k.startswith("adapter"))


def pass_ci(arm: dict[str, dict]) -> tuple[float, float, float]:
    """Pass rate in percent with a 95% Wilson interval (one 0/1 outcome per fixture)."""
    p, lo, hi = wilson_ci(sum(r["pass"] for r in arm.values()), len(arm))
    return 100 * p, 100 * lo, 100 * hi


def words_ci(arm: dict[str, dict]) -> tuple[float, float, float]:
    """Mean words per answer with a 95% bootstrap interval over fixtures."""
    return bootstrap_ci([r["words"] for r in arm.values()])


def delta(adapter: dict[str, dict], other: dict[str, dict]) -> tuple[float, float, float]:
    """Adapter pass minus the other arm's pass on the same fixtures, in points, with a paired bootstrap interval."""
    ids = sorted(other)
    m, lo, hi = paired_delta_ci((other[i]["pass"], adapter[i]["pass"]) for i in ids)
    return 100 * m, 100 * lo, 100 * hi


def uneven_bars(ax, groups, *, fmt, ylabel, title, ylim):
    """House-style bars where groups may hold different arms: ``groups = [(label, [(arm, mean, lo, hi), ...])]``."""
    width, x, seen = 0.26, 0.0, set()
    ticks = []
    for glabel, arms in groups:
        start = x
        for arm, m, lo, hi in arms:
            bar = ax.bar(x, m, width * 0.95, yerr=[[m - lo], [hi - m]], capsize=4, color=color_for(arm),
                         label=arm if arm not in seen else None, error_kw={"linewidth": 1.2})
            seen.add(arm)
            ax.bar_label(bar, labels=[fmt.format(m)], padding=12, fontsize=9)
            x += width
        ticks.append((start + x - width) / 2)
        x += 0.35
    ax.set_xticks(ticks, [g for g, _ in groups])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    return ax


def headline_figure(groups, path: str | Path):
    """Section 1 figure: pass rate (left) and mean words (right) for ``groups = [(run label, [(arm label, arm)])]``."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.8))
    uneven_bars(ax_l, [(g, [(a, *pass_ci(arm)) for a, arm in arms]) for g, arms in groups],
                fmt="{:.0f}%", ylabel="% of answers passing (95% Wilson CI)", title="Pass rate", ylim=(0, 112))
    uneven_bars(ax_r, [(g, [(a, *words_ci(arm)) for a, arm in arms]) for g, arms in groups],
                fmt="{:.1f}", ylabel="mean words per answer (95% bootstrap CI)", title="Mean words per answer",
                ylim=(0, 46))
    handles, labels = ax_l.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ARM_LABELS.index(labels[i]))
    ax_l.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.suptitle("Does the adapter answer as accurately as the prompt, in fewer words, with no prompt?")
    fig.tight_layout()
    save(fig, HERE / path)
    return fig


def print_arms(groups) -> None:
    """One line per arm: pass rate and mean words, each with its interval."""
    for g, arms in groups:
        print(g.replace("\n", " "))
        for a, arm in arms:
            p, plo, phi = pass_ci(arm)
            w, wlo, whi = words_ci(arm)
            print(f"  {a:<9} pass {p:5.1f}% [{plo:.0f}, {phi:.0f}]   mean words {w:5.1f} [{wlo:.0f}, {whi:.0f}]")


def delta_figure(comparisons, path: str | Path):
    """Section 1 paired chart: ``comparisons = [(label, (delta, lo, hi))]``."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    delta_bars(ax, [c[0] for c in comparisons], [c[1][0] for c in comparisons],
               [c[1][1] for c in comparisons], [c[1][2] for c in comparisons], fmt="{:+.0f}",
               title="Is the adapter better than the base, or than the prompt?")
    ax.set_ylim(-25, 45)
    fig.tight_layout()
    save(fig, HERE / path)
    return fig


def brevity_scatter(runs: dict[str, dict], path: str | Path):
    """Section 1 scatter: one point per arm per run, mean words on x, pass rate on y, whiskers for both intervals."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    seen = set()
    offsets = [(8, 10), (8, 2), (8, -7), (8, -16)]  # one label row per run, so the stacked base points stay legible
    for i, (name, arms) in enumerate(runs.items()):
        for label, arm in [("Base", arms["base"]), ("Prompted", arms.get("skill")), ("Adapter", adapter_of(arms))]:
            if arm is None:
                continue
            w, wlo, whi = words_ci(arm)
            p, plo, phi = pass_ci(arm)
            ax.errorbar(w, p, xerr=[[w - wlo], [whi - w]], yerr=[[p - plo], [phi - p]], fmt="o",
                        color=color_for(label), ecolor=color_for(label), elinewidth=0.9, capsize=2.5,
                        alpha=0.45, markersize=0)
            ax.plot(w, p, "o", color=color_for(label), markersize=7, label=label if label not in seen else None)
            seen.add(label)
            tag = name.replace("\n", " ") + (" (prompted)" if label == "Prompted" else "")
            dx, dy = offsets[i] if label != "Prompted" else (-8, 10)
            ax.annotate(tag, (w, p), textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                        ha="right" if dx < 0 else "left")
    ax.set_xlim(0, 40)
    ax.set_ylim(40, 105)
    ax.set_xlabel("mean words per answer (shorter to the left)")
    ax.set_ylabel("% of answers passing")
    ax.set_title("Is shorter just worse?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, HERE / path)
    return fig


def family_table(run: str) -> None:
    """Pass rate per fixture family and arm, read from ``<run>/summary.json``."""
    summary = json.loads((HERE / run / "summary.json").read_text())["summary"]
    arms = list(summary)
    families = list(next(iter(summary.values()))["by_family"])
    print(f"{'family':<12}{'n':>4}  " + "".join(f"{a:>13}" for a in arms))
    for fam in families:
        n = summary[arms[0]]["by_family"][fam]["n"]
        cells = "".join(f"{summary[a]['by_family'][fam]['pass_rate'] * n:>9.0f} / {n:<2}" for a in arms)
        print(f"{fam:<12}{n:>4}  {cells}")


def token_bars(responses, groups: dict[str, str], arms, *, ylim, title, path: str | Path):
    """Section 8 bars of mean visible tokens per arm and probe group, 95% bootstrap interval over probes."""
    series = []
    for label, arm in arms:
        stats = [bootstrap_ci([r["visible_tokens"] for r in responses if r["arm"] == arm and r["split"] == g])
                 for g in groups]
        series.append((label, [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats]))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    arm_bars(ax, list(groups.values()), series, fmt="{:.0f}", ylabel="mean visible tokens", ylim=ylim, title=title)
    fig.tight_layout()
    save(fig, HERE / path)
    return fig


def summary_token_bars(pilot: dict, groups: dict[str, str], arms, *, ylim, title, path: str | Path):
    """Section 8 bars from packaged summary means only (no per-row data, so no interval)."""
    series = [(label, [pilot["groups"][g][arm]["mean_visible_tokens"] for g in groups]) for label, arm in arms]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    arm_bars(ax, list(groups.values()), series, fmt="{:.0f}", ylabel="mean visible tokens", ylim=ylim, title=title)
    fig.tight_layout()
    save(fig, HERE / path)
    return fig
