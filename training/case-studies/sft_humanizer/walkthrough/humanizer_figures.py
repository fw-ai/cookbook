"""Figure helpers for the Humanizer notebook.

Loads the shared gallery plotting module ``../common/figures.py`` by file path (so the
``figures/`` folder of PNGs beside the notebook can never shadow it) and adds the
Humanizer-specific pieces: the four metrics, their denominators, Wilson intervals from a
rate, and the per-arm series the shared ``two_panel``/``arm_bars`` helpers draw.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import matplotlib

from .cards import ARM_LABELS, IN_NOTEBOOK

if not IN_NOTEBOOK:  # figures are saved, never shown in a window
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import matplotlib.pyplot as plt  # noqa: E402, F401  (re-exported; backend must be chosen first)
from matplotlib.container import BarContainer  # noqa: E402

STUDY_DIR = Path(__file__).resolve().parent.parent
COMMON_FIGURES = STUDY_DIR.parent / "common" / "figures.py"


def _load_common():
    spec = importlib.util.spec_from_file_location("gallery_figures", COMMON_FIGURES)
    if spec is None or spec.loader is None:
        raise ImportError(f"Shared plotting module not found at {COMMON_FIGURES}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_common = _load_common()
style = _common.style
arm_bars = _common.arm_bars
two_panel = _common.two_panel
delta_bars = _common.delta_bars
save = _common.save
wilson_ci = _common.wilson_ci
paired_delta_ci = _common.paired_delta_ci

# The four Humanizer metrics: (key in summary.json, bar label, denominator per arm).
# Two point up (higher is better), two point down (lower is better).
UP = [("preserve_rate", "Preserve", 99), ("pass_rate", "Pass", 99)]
DOWN = [
    ("over_edit_good", "Over-edits good prose\n(n = 11 good)", 11),
    ("idem_changed", "2nd pass changed >10%\n(n = 99)", 99),
]
METRICS = UP + DOWN
HEADLINE_ARMS = ["base_short", "skill", "adapter_s29"]
LEFT = dict(title="Kept the meaning and passed", ylabel="% of 99 passages (higher is better)", fmt="{:.0f}%", ylim=(0, 118))
RIGHT = dict(
    title="Stopped editing",
    ylabel="% of passages (lower is better)\nn = 11 good passages, n = 99",
    fmt="{:.0f}%",
    ylim=(0, 118),
)
PLAIN_GREY = "#c9ccd3"  # the plain-prompt base, when drawn, is a lighter grey than the editor-instruction base


def rate_ci(rate: float, n: int) -> tuple[float, float, float]:
    """Percent and 95% Wilson interval for a per-passage yes/no rate reported over ``n`` passages."""
    p, lo, hi = wilson_ci(round(rate * n), n)
    return 100 * p, 100 * lo, 100 * hi


def metric_series(summary: dict, arms, metrics):
    """One (label, values, lows, highs) tuple per arm, in the order ``metrics`` lists the keys."""
    series = []
    for arm in arms:
        vals, los, his = zip(*(rate_ci(summary[arm][key], n) for key, _, n in metrics))
        series.append((ARM_LABELS.get(arm, arm), vals, los, his))
    return series


def lighten_plain(fig, labels) -> None:
    """Both base arms share the house grey; draw the plain-prompt base lighter and move the legend below."""
    for ax in fig.axes:
        bars = [c for c in ax.containers if isinstance(c, BarContainer)]  # bar() with yerr also registers errorbars
        for container, label in zip(bars, labels):
            if label == ARM_LABELS["base_plain"]:
                for bar in container:
                    bar.set_color(PLAIN_GREY)
        if ax.get_legend() is not None:
            ax.legend(loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, -0.14), frameon=False)


def headline_figure(summary: dict, out_path, suptitle: str, arms=HEADLINE_ARMS):
    """The standard four-metric figure from a ``summary.json`` ``summary`` dict; saves the PNG and returns the figure."""
    fig = two_panel(
        groups=[label for _, label, _ in UP],
        series=metric_series(summary, arms, UP),
        left=LEFT,
        right=RIGHT,
        right_series=metric_series(summary, arms, DOWN),
        suptitle=suptitle,
    )
    fig.axes[1].set_xticks(range(len(DOWN)), [label for _, label, _ in DOWN])
    save(fig, out_path)
    return fig
