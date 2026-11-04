"""House style for case-study figures: one look across every notebook, offline, matplotlib only.

Interval policy for every notebook: a per-row 0/1 rate gets `wilson_ci`; samples that share a
task get `cluster_bootstrap_ci` over tasks; "is A better than B" on the same cases is answered by
`paired_delta_ci`, never by whether two per-arm intervals overlap; a 100% observed rate still gets
an interval.

Every helper takes plain numbers that a notebook has already read from its committed results, draws
one clear comparison with its confidence interval, and returns the figure. Nothing here calls a model
or reads a file. The look follows the effort_reducer study: grouped bars, black whiskers for the
95% interval, the value printed above each bar, a light grid, a one-line title that says what is
being compared and on what data.

    from figures import style, arm_bars, two_panel, save
    style()
    fig = two_panel(
        groups=["28 held-out prompts"],
        series=[("Base", [71]), ("Prompted", [82]), ("Adapter", [75])],
        left=dict(title="Pass rate", ylabel="% of answers passing", fmt="{:.0f}%"),
        right=dict(title="Length", ylabel="words per answer", fmt="{:.0f}"),
        right_series=[("Base", [26]), ("Prompted", [12]), ("Adapter", [8])],
        suptitle="Caveman: adapter vs prompt vs base (95% bootstrap CI)",
    )
    save(fig, "figures/caveman_result.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

# Arm colours. Same arm, same colour, in every notebook.
COLORS = {
    "base": "#8a8f98",      # grey: the untouched model
    "prompted": "#3274a1",  # blue: the skill as a system prompt
    "adapter": "#e1812c",   # orange: the LoRA
    "accent": "#c03d3e",    # red: thresholds and margins
    "teacher": "#5b8c5a",   # green: a stronger reference model
}
_FALLBACK = ["#3274a1", "#e1812c", "#8a8f98", "#5b8c5a", "#7b5ea7", "#c03d3e"]


def style() -> None:
    """Apply the house rcParams once per notebook."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9.5,
    })


def color_for(name: str, index: int = 0) -> str:
    key = name.strip().lower()
    for k, c in COLORS.items():
        if key.startswith(k):
            return c
    return _FALLBACK[index % len(_FALLBACK)]


def _err(values: Sequence[float], lo: Sequence[float] | None, hi: Sequence[float] | None):
    if lo is None or hi is None:
        return None
    return [[v - low for v, low in zip(values, lo)], [h - v for v, h in zip(values, hi)]]


def arm_bars(
    ax,
    groups: Sequence[str],
    series: Sequence[tuple],
    *,
    title: str = "",
    ylabel: str = "",
    fmt: str = "{:.1f}",
    baseline: float | None = None,
    baseline_label: str = "",
    ylim: tuple | None = None,
    legend: bool = True,
):
    """Grouped bars with 95% whiskers.

    series: (label, values) or (label, values, lo, hi); one value per group.
    """
    width = 0.8 / max(len(series), 1)
    for i, s in enumerate(series):
        label, values = s[0], list(s[1])
        lo = list(s[2]) if len(s) > 2 and s[2] is not None else None
        hi = list(s[3]) if len(s) > 3 and s[3] is not None else None
        x = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(x, values, width * 0.95, yerr=_err(values, lo, hi), capsize=4,
                      color=color_for(label, i), label=label, error_kw={"linewidth": 1.2})
        pad = 12 if lo is not None else 3
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=pad, fontsize=9)
    ax.set_xticks(range(len(groups)), groups)
    ax.axhline(0, color="black", linewidth=1)
    if baseline is not None:
        ax.axhline(baseline, color=COLORS["accent"], linestyle="--", linewidth=1.2)
        if baseline_label:
            ax.text(len(groups) - 0.55, baseline, baseline_label, color=COLORS["accent"],
                    ha="right", va="bottom", fontsize=9)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if legend and len(series) > 1:
        ax.legend(loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, -0.14), frameon=False)
    return ax


def two_panel(
    *,
    groups: Sequence[str],
    series: Sequence[tuple],
    left: dict,
    right: dict,
    right_series: Sequence[tuple] | None = None,
    suptitle: str = "",
    figsize=(12, 4.6),
):
    """The standard result figure: quality on the left, cost (length or tokens) on the right."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    arm_bars(ax_l, groups, series, **left)
    arm_bars(ax_r, groups, right_series or series, legend=False, **right)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def delta_bars(
    ax,
    labels: Sequence[str],
    deltas: Sequence[float],
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    *,
    title: str = "",
    ylabel: str = "percentage points",
    fmt: str = "{:+.1f}",
    margin: float | None = None,
    margin_label: str = "",
):
    """Signed differences about zero, one bar per comparison, whiskers for the interval."""
    colors = [COLORS["adapter"] if d >= 0 else COLORS["base"] for d in deltas]
    bars = ax.bar(range(len(labels)), deltas, 0.6, yerr=_err(deltas, lo, hi), capsize=4,
                  color=colors, error_kw={"linewidth": 1.2})
    ax.bar_label(bars, labels=[fmt.format(d) for d in deltas], padding=12 if lo else 3, fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.axhline(0, color="black", linewidth=1)
    if margin is not None:
        ax.axhline(margin, color=COLORS["accent"], linestyle="--", linewidth=1.2)
        if margin_label:
            ax.text(len(labels) - 0.5, margin, margin_label, color=COLORS["accent"],
                    ha="right", va="top", fontsize=9)
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def curve(
    ax,
    x: Sequence[float],
    series: Sequence[tuple],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    marker: str = "o",
):
    """Lines over steps or seeds: (label, ys) or (label, ys, lo, hi) draws a shaded band."""
    for i, s in enumerate(series):
        label, ys = s[0], list(s[1])
        c = color_for(label, i)
        ax.plot(x, ys, marker=marker, color=c, label=label, linewidth=1.8, markersize=4)
        if len(s) > 3 and s[2] is not None and s[3] is not None:
            ax.fill_between(x, list(s[2]), list(s[3]), color=c, alpha=0.15, linewidth=0)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="best")
    return ax


def save(fig, path: str | Path) -> Path:
    """Write the PNG next to the notebook so the README can embed it, then return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap interval, for notebooks that hold per-row scores."""
    import random

    vals = list(values)
    if not vals:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_boot))
    mean = sum(vals) / n
    return mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with a Wilson score interval, which stays open at 0% and 100%.

    Returns (rate, lo, hi) in 0..1. Use this for pass rates and other per-row 0/1 outcomes
    instead of a bootstrap, which collapses to zero width when every row agrees.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap_ci(
    groups: Iterable[Iterable[float]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with a 95% interval that resamples whole groups (tasks, prompts), not rows.

    Use when several samples share a task: resampling rows would treat four draws of one
    problem as four independent facts and give an interval that is too narrow.
    """
    import random

    gs = [list(g) for g in groups if list(g)]
    if not gs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    flat = [v for g in gs for v in g]
    mean = sum(flat) / len(flat)
    means = []
    for _ in range(n_boot):
        pick = [gs[rng.randrange(len(gs))] for _ in range(len(gs))]
        vals = [v for g in pick for v in g]
        means.append(sum(vals) / len(vals))
    means.sort()
    return mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def paired_delta_ci(
    pairs: Iterable[tuple[float, float]], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference (b - a) over paired rows with a 95% bootstrap interval.

    The question "is the adapter better than the prompt?" is answered by this, not by whether
    two per-arm intervals overlap.
    """
    import random

    ds = [b - a for a, b in pairs]
    if not ds:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(ds)
    mean = sum(ds) / n
    means = sorted(sum(rng.choice(ds) for _ in range(n)) / n for _ in range(n_boot))
    return mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
