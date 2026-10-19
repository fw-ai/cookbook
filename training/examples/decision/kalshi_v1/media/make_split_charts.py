#!/usr/bin/env python3
"""Render dataset-breakdown and category-mix charts for the kalshi_v1 README.

Matches the clean-chart visual language used by the study figures
(kalshi-clean-*.png): white ground, bold title, framed legend, dpi=150.

    python make_split_charts.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#202733"})

OUT_DIR = Path(__file__).resolve().parent

SPLIT_COLORS = {
    "Training": "#1f4fd1",
    "Calibration (dev)": "#5b6676",
    "Test": "#2e9e4f",
}

CATEGORY_COLORS = {
    "Sports": "#1f4fd1",
    "Politics": "#e8730c",
    "Entertainment": "#2e9e4f",
    "Economics": "#7c3aed",
    "Science and Technology": "#0891b2",
    "Elections": "#db2777",
    "Mentions": "#ca8a04",
    "Climate and Weather": "#0d9488",
    "Other": "#9aa4b2",
}

SPLIT_SIZES = {
    "Training": {"events": 6461, "questions": 20051},
    "Calibration (dev)": {"events": 2000, "questions": 6232},
    "Test": {"events": 8765, "questions": 33712},
}

CATEGORY_COUNTS: dict[str, dict[str, int]] = {
    "Training": {
        "Sports": 1202,
        "Politics": 1503,
        "Entertainment": 1495,
        "Economics": 546,
        "Science and Technology": 100,
        "Elections": 419,
        "Mentions": 371,
        "Climate and Weather": 145,
        "Other": 680,
    },
    "Calibration (dev)": {
        "Sports": 1803,
        "Politics": 10,
        "Entertainment": 38,
        "Economics": 6,
        "Science and Technology": 4,
        "Elections": 16,
        "Mentions": 118,
        "Climate and Weather": 5,
    },
    "Test": {
        "Sports": 4814,
        "Politics": 540,
        "Entertainment": 865,
        "Economics": 632,
        "Science and Technology": 400,
        "Elections": 525,
        "Mentions": 859,
        "Climate and Weather": 114,
        "Other": 16,
    },
}


def _style_axes(ax) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _donut(ax, labels: list[str], sizes: list[int], colors: list[str], title: str, subtitle: str) -> None:
    total = sum(sizes)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.40, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
    ax.text(0, 0.05, f"{total:,}", ha="center", va="center", fontsize=17, fontweight="bold", color="#202733")
    ax.text(0, -0.16, subtitle, ha="center", va="center", fontsize=10, color="#596474")
    # Pull the ring to the panel edges (default pie leaves a large empty margin).
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    _style_axes(ax)

    for wedge, size in zip(wedges, sizes):
        share = size / total
        if share < 0.08:
            continue
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        rad = math.radians(angle)
        ax.text(
            0.78 * math.cos(rad),
            0.78 * math.sin(rad),
            f"{100 * share:.0f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#222",
        )


def _framed_legend(ax, handles, labels, *, fontsize: float = 9.0) -> None:
    ax.axis("off")
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0,
        fontsize=fontsize,
        title="Legend",
        title_fontsize=fontsize + 0.5,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#c8cdd5",
        facecolor="#f7f8fa",
        borderpad=0.7,
        labelspacing=0.55,
        handlelength=1.1,
        handletextpad=0.55,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color("#222")


def _content_margins(path: Path) -> dict[str, float]:
    arr = np.asarray(Image.open(path).convert("RGB"))
    mask = arr.min(axis=2) < 248
    ys, xs = np.where(mask)
    h, w = arr.shape[:2]
    return {
        "top": float(ys.min() / h),
        "bottom": float((h - 1 - ys.max()) / h),
        "left": float(xs.min() / w),
        "right": float((w - 1 - xs.max()) / w),
        "width": w,
        "height": h,
    }


def _assert_tight(path: Path, *, max_side: float = 0.06, max_bottom: float = 0.08) -> None:
    m = _content_margins(path)
    print(f"QA {path.name}: {m['width']}x{m['height']} margins={m}")
    assert m["top"] <= max_side, m
    assert m["left"] <= max_side, m
    assert m["right"] <= max_side, m
    assert m["bottom"] <= max_bottom, m


def render_dataset_breakdown(path: Path) -> None:
    # Compact row: pie | pie | legend. Footnote rides under with almost no gap.
    fig = plt.figure(figsize=(12.8, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.01)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])

    labels = list(SPLIT_SIZES)
    colors = [SPLIT_COLORS[name] for name in labels]
    _donut(ax0, labels, [SPLIT_SIZES[n]["events"] for n in labels], colors, "Events by split", "events")
    _donut(ax1, labels, [SPLIT_SIZES[n]["questions"] for n in labels], colors, "Questions by split", "questions")

    _framed_legend(
        ax_leg,
        handles=[Patch(facecolor=c, edgecolor="none") for c in colors],
        labels=[
            f"{name}\n{SPLIT_SIZES[name]['events']:,} events · {SPLIT_SIZES[name]['questions']:,} q" for name in labels
        ],
        fontsize=10,
    )

    fig.suptitle("Dataset breakdown", x=0.02, ha="left", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.83, bottom=0.14)
    fig.text(
        0.02,
        0.025,
        "One published test set: 8,765 events / 33,712 questions. "
        "Unseen topics (765 events) is a subset of test, not another split.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#596474",
    )
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def render_category_mix(path: Path) -> None:
    order = list(CATEGORY_COLORS)
    fig = plt.figure(figsize=(12.8, 4.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.70], wspace=0.02)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_leg = fig.add_subplot(gs[0, 3])

    panel_titles = {
        "Training": "Training (balanced)",
        "Calibration (dev)": "Calibration / fit-T",
        "Test": "Consolidated test",
    }
    for ax, split in zip(axes, CATEGORY_COUNTS):
        counts = CATEGORY_COUNTS[split]
        labels = [c for c in order if c in counts and counts[c] > 0]
        sizes = [counts[c] for c in labels]
        colors = [CATEGORY_COLORS[c] for c in labels]
        _donut(ax, labels, sizes, colors, panel_titles[split], "events")

    legend_cats = [c for c in order if any(c in CATEGORY_COUNTS[s] for s in CATEGORY_COUNTS)]
    _framed_legend(
        ax_leg,
        handles=[Patch(facecolor=CATEGORY_COLORS[c], edgecolor="none") for c in legend_cats],
        labels=legend_cats,
        fontsize=10,
    )

    fig.suptitle("Category mix across the three splits", x=0.02, ha="left", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.84, bottom=0.14)
    fig.text(
        0.02,
        0.025,
        "Sports capped at 18.6% in training. The consolidated test includes the unseen-topic subset.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#596474",
    )
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    breakdown = OUT_DIR / "kalshi-clean-dataset-breakdown.png"
    mix = OUT_DIR / "kalshi-clean-category-mix.png"
    render_dataset_breakdown(breakdown)
    render_category_mix(mix)
    print(f"wrote {breakdown}")
    print(f"wrote {mix}")
    # Fail the script if we reintroduce large empty margins.
    _assert_tight(breakdown, max_side=0.06, max_bottom=0.06)
    _assert_tight(mix, max_side=0.06, max_bottom=0.06)


if __name__ == "__main__":
    main()
