"""Figure helpers for the Anti-slop DPO notebook (``dpo_glm53_flash.ipynb``).

Two plots live here so the notebook cells that call them stay short:

* :func:`headline_figure` rebuilds the recorded result from the committed
  ``results/<date>/<run>/samples.jsonl`` files: detector P(human) per arm
  with a 95% bootstrap interval, and the number of distinct openers. No key,
  no network.
* :func:`training_curves` draws the four live training panels (loss, train
  margin and accuracy, implicit rewards, held-out change from step 0) from
  the ``history`` and ``eval_history`` lists the training loop fills.

Both import the gallery's shared ``figures`` module from ``../common``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_COMMON = Path(__file__).resolve().parent.parent / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from figures import COLORS, bootstrap_ci, save, style, two_panel  # noqa: E402

# (label, run, arm). The RL + diversity arm was evaluated in a second pass on
# the same 20 held-out briefs; every other arm comes from the first pass.
HEADLINE_ARMS = [
    ("Base", "2026-10-08/antislop-eval-rl", "base"),
    ("Prompted ban", "2026-10-08/antislop-eval-rl", "base_ban"),
    ("DPO v1", "2026-10-08/antislop-eval-rl", "dpo_v1"),
    ("RL step 20", "2026-10-08/antislop-eval-rl", "rl_s20"),
    ("RL + diversity", "2026-10-08/antislop-eval-rl2", "rl_div"),
]


def load_arm(results_dir: Path, run: str, arm: str) -> list[dict]:
    """Rows of one arm from ``results_dir/run/samples.jsonl`` that carry a detector score."""
    path = Path(results_dir) / run / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["arm"] == arm and r.get("detector") is not None]


def opener(text: str, n: int = 3) -> str:
    """First ``n`` whitespace-separated words, lower-cased: the "distinct openers" unit."""
    return " ".join(text.lower().split()[:n])


def arm_color(label: str) -> str:
    """Base grey, prompted blue, every trained adapter orange."""
    if label == "Base":
        return COLORS["base"]
    if label.startswith("Prompted"):
        return COLORS["prompted"]
    return COLORS["adapter"]


def headline_figure(results_dir: Path, save_to: Path | None = None, arms=HEADLINE_ARMS):
    """Detector P(human) (left, 95% bootstrap whiskers) and distinct openers (right) per arm.

    Prints one line per arm and returns ``(fig, stats)`` where ``stats`` is a
    list of dicts with ``label, n, detector, lo, hi, distinct_openers``.
    """
    style()
    stats = []
    for label, run, arm in arms:
        rows = load_arm(results_dir, run, arm)
        mean, lo, hi = bootstrap_ci([r["detector"] for r in rows])
        distinct = len({opener(r["text"]) for r in rows})
        stats.append(dict(label=label, n=len(rows), detector=mean, lo=lo, hi=hi, distinct_openers=distinct))
        print(f"{label:15s} n={len(rows)}  detector {mean:.2f} [{lo:.2f}, {hi:.2f}]  distinct openers {distinct}")

    labels = [s["label"] for s in stats]
    n_posts = stats[0]["n"]
    fig = two_panel(
        groups=labels,
        series=[("adapter", [s["detector"] for s in stats], [s["lo"] for s in stats], [s["hi"] for s in stats])],
        left=dict(title="Did the reward go up? Detector P(human)", ylabel=f"mean over {n_posts} posts",
                  fmt="{:.2f}", ylim=(0, 1.05)),
        right=dict(title="What did the writing lose? Distinct openers", ylabel=f"of {n_posts} posts",
                   fmt="{:.0f}", ylim=(0, 44)),
        right_series=[("adapter", [s["distinct_openers"] for s in stats])],
        suptitle="Recorded 2026-10-08: did the detector score rise, and what did the writing lose? "
                 "(20 held-out briefs x 2 tweets per arm)",
    )
    for ax in fig.axes:
        for patch, label in zip(ax.patches, labels):
            patch.set_color(arm_color(label))
    if save_to is not None:
        print("saved", save(fig, save_to))
    return fig, stats


def _series(records, key):
    """Steps and values for one metric, skipping entries that reported None."""
    pts = [(r["step"], r[key]) for r in records if r.get(key) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def training_curves(history: list[dict], eval_history: list[dict]):
    """Four panels from the live run: loss, train margin + accuracy, implicit rewards, held-out change.

    ``history`` holds one dict per optimizer step (``step, dpo_loss, margin, accuracy,
    chosen_reward, rejected_reward``); ``eval_history`` one per held-out check
    (``step, chosen, rejected, margin``), its first entry being the step-0 baseline.
    The last panel is the one to trust: it is the same fixed pairs every time.
    """
    fig, ax = plt.subplots(1, 4, figsize=(19.5, 3.8))

    # 1. loss
    ax[0].plot(*_series(history, "dpo_loss"), marker="o", ms=4, color="#c2410c", zorder=3)
    ax[0].axhline(math.log(2), ls="--", c="gray", lw=1, zorder=1)
    ax[0].annotate("ln 2 = no preference", xy=(0.97, math.log(2)),
                   xycoords=("axes fraction", "data"), ha="right", va="bottom",
                   fontsize=8, color="gray",
                   bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5), zorder=4)
    ax[0].set_title("train: dpo_loss (down is good)")
    ax[0].set_ylabel("loss")

    # 2. margin + accuracy (train)
    axa = ax[1].twinx()
    axa.plot(*_series(history, "accuracy"), marker="s", ms=3, lw=1, alpha=.45,
             color="#15803d", label="accuracy", zorder=1)
    axa.axhline(0.5, ls=":", c="#15803d", lw=1, alpha=.6, zorder=1)
    axa.set_ylim(0, 1.05)
    axa.set_ylabel("accuracy", color="#15803d")
    axa.tick_params(axis="y", colors="#15803d")
    ax[1].plot(*_series(history, "margin"), marker="o", ms=4, color="#1d4ed8",
               label="margin (unscaled)", zorder=3)
    ax[1].axhline(0, ls="--", c="gray", lw=1, zorder=2)
    ax[1].set_ylabel("margin (log units)", color="#1d4ed8")
    ax[1].set_zorder(axa.get_zorder() + 1)
    ax[1].patch.set_visible(False)
    ax[1].set_title("train: margin + accuracy")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = axa.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=.9)

    # 3. implicit rewards: catches likelihood displacement
    ax[2].plot(*_series(history, "chosen_reward"), marker="o", ms=4, color="#15803d", label="chosen")
    ax[2].plot(*_series(history, "rejected_reward"), marker="s", ms=4, color="#b91c1c", label="rejected")
    ax[2].axhline(0, ls="--", c="gray", lw=1)
    ax[2].legend(fontsize=8, loc="lower left")
    ax[2].set_ylabel("implicit reward")
    ax[2].set_title("train: implicit rewards\n(watch for BOTH falling)")

    # 4. the headline: held-out, same pairs every time, plotted as change from step 0
    base = eval_history[0]
    for key, color, mk, lbl in (("margin", "#1d4ed8", "o", "margin"),
                                ("chosen", "#15803d", "o", "chosen"),
                                ("rejected", "#b91c1c", "s", "rejected")):
        xs, ys = _series(eval_history, key)
        dy = [y - base[key] for y in ys]
        ax[3].plot(xs, dy, marker=mk, ms=5, lw=2 if key == "margin" else 1.5,
                   color=color, label=lbl, alpha=1.0 if key == "margin" else .8)
        if dy:
            # Label the endpoint: `chosen` is often small next to `rejected` on a
            # shared axis, and its sign is the whole question.
            ax[3].annotate(f"{dy[-1]:+.2f}", xy=(xs[-1], dy[-1]), xytext=(4, 0),
                           textcoords="offset points", va="center", fontsize=8,
                           color=color, fontweight="bold")
    ax[3].axhline(0, ls="--", c="gray", lw=1)
    ax[3].legend(fontsize=8, loc="upper left")
    ax[3].set_ylabel("change from step 0 (nats)")
    ax[3].set_title("HELD OUT: change vs baseline\n(margin up + chosen up = real)")

    for a in ax:
        a.set_xlabel("optimizer step")
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.show()
    return fig
