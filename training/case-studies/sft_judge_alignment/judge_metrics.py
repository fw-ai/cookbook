"""Agreement, calibration, latency and cost for a rubric judge.

Three axes, and the calibration axis is the one that usually gets skipped.

Agreement (`kappa_report`) uses quadratic-weighted Cohen's kappa, the standard
choice for ordinal 1-5 labels, with a bootstrap CI so a 50-row holdout does not
produce a confident ranking it cannot support.

Calibration is measured four ways because a single ECE number hides too much:

  `ece`                    scalar + the per-bin table the diagram is drawn from,
                           plus a *signed* gap so over- and under-confidence are
                           distinguishable, plus MCE for the worst bin.
  `reliability_curve`      the diagram. Several judges on one set of axes, with
                           bin counts underneath so a 3-sample bin at the far
                           right is not read as a finding.
  `brier_decomposition`    reliability / resolution / uncertainty. Separates
                           "miscalibrated" from "not discriminative", which ECE
                           conflates, and degrades more gracefully on small n.
  `panel_tv`               distance from the judge's distribution to the panel's
                           own spread. The sharpest test here: on traces the
                           annotators split on, does the judge hedge too?

`fit_temperature` is the remedy rather than just the warning -- one scalar, fit
on a calibration split, that usually recovers most of the ECE a fine-tune costs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

SCALE = [1, 2, 3, 4, 5]
N_CLASSES = len(SCALE)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def _kappa(x: Sequence[int], y: Sequence[int]) -> float:
    from sklearn.metrics import cohen_kappa_score

    if len(x) < 2:
        return float("nan")
    # cohen_kappa is undefined when both raters are constant and identical;
    # perfect agreement on a degenerate column is a 1.0, not a nan.
    if len(set(x)) == 1 and len(set(y)) == 1:
        return 1.0 if x[0] == y[0] else 0.0
    return float(cohen_kappa_score(x, y, weights="quadratic", labels=SCALE))


def kappa_report(
    pred: Sequence[dict[str, int]],
    gold: Sequence[dict[str, int]],
    attributes: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-attribute quadratic-weighted kappa + exact match + bootstrap CI on the mean."""
    assert len(pred) == len(gold)
    per_attr, exact = {}, {}
    for a in attributes:
        x = [p[a] for p in pred]
        y = [g[a] for g in gold]
        per_attr[a] = _kappa(x, y)
        exact[a] = sum(int(i == j) for i, j in zip(x, y)) / len(x) if x else float("nan")
    mean = float(np.mean([per_attr[a] for a in attributes]))

    rng = np.random.default_rng(seed)
    n = len(pred)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = [_kappa([pred[i][a] for i in idx], [gold[i][a] for i in idx]) for a in attributes]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            boots.append(float(np.mean(vals)))
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"),) * 2
    return {"per_attribute": per_attr, "exact": exact, "mean": mean, "ci95": (lo, hi), "n": n}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class ECEResult:
    ece: float
    signed_gap: float          # mean(confidence - accuracy); >0 means overconfident
    mce: float                 # worst populated bin
    accuracy: float
    mean_confidence: float
    n: int
    bins: list[dict[str, float]] = field(default_factory=list)


def ece(confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10) -> ECEResult:
    """Equal-width binned ECE. Returns the scalar *and* the table the plot uses.

    Both come from one code path on purpose: a reliability diagram that disagrees
    with the reported number is worse than no diagram.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    n = len(conf)
    if n == 0:
        return ECEResult(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, [])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by confidence; the first bin owns its left edge so p=0 is not dropped.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    rows, total_gap, worst = [], 0.0, 0.0
    signed = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "count": 0,
                         "mean_confidence": float("nan"), "accuracy": float("nan"), "gap": float("nan")})
            continue
        mc = float(conf[mask].mean())
        acc = float(corr[mask].mean())
        gap = mc - acc
        rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "count": count,
                     "mean_confidence": mc, "accuracy": acc, "gap": gap})
        total_gap += (count / n) * abs(gap)
        signed += (count / n) * gap
        worst = max(worst, abs(gap))

    return ECEResult(
        ece=float(total_gap), signed_gap=float(signed), mce=float(worst),
        accuracy=float(corr.mean()), mean_confidence=float(conf.mean()), n=n, bins=rows,
    )


def flatten(
    dists: Sequence[dict[str, list[float]]],
    gold: Sequence[dict[str, int]],
    attributes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-attribute predictions into (probs [N,5], gold_index [N]) for pooled metrics."""
    P, Y = [], []
    for d, g in zip(dists, gold):
        for a in attributes:
            P.append(d[a])
            Y.append(g[a] - 1)
    return np.asarray(P, dtype=float), np.asarray(Y, dtype=int)


def confidence_correct(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-class probability and whether that class is the gold label."""
    pred = probs.argmax(axis=1)
    return probs.max(axis=1), pred == y


def ece_from(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> ECEResult:
    conf, corr = confidence_correct(probs, y)
    return ece(conf, corr, n_bins=n_bins)


def suggest_bins(n: int) -> int:
    """Fewer bins on small holdouts.

    With 50 rows and 10 bins most bins hold 0-3 samples and ECE becomes a
    measurement of the binning, not the judge.
    """
    return 5 if n < 100 else 10


def brier_decomposition(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> dict[str, float]:
    """Multiclass Brier score split into reliability, resolution and uncertainty.

    Computed one-vs-rest per class and summed (Murphy's decomposition), so
    `brier ~= reliability - resolution + uncertainty`. Lower reliability is
    better; higher resolution is better.
    """
    n = len(y)
    onehot = np.zeros((n, N_CLASSES))
    onehot[np.arange(n), y] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    rel = res = unc = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for k in range(N_CLASSES):
        p = probs[:, k]
        o = onehot[:, k]
        base = o.mean()
        unc += base * (1 - base)
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
        for b in range(n_bins):
            mask = idx == b
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            rel += cnt / n * (p[mask].mean() - o[mask].mean()) ** 2
            res += cnt / n * (o[mask].mean() - base) ** 2
    return {"brier": brier, "reliability": float(rel), "resolution": float(res), "uncertainty": float(unc)}


def panel_tv(
    dists: Sequence[dict[str, list[float]]],
    panel_dists: Sequence[dict[str, list[float]]],
    attributes: Sequence[str],
) -> dict[str, float]:
    """Mean total-variation distance between judge and annotator-panel distributions.

    Reported both overall and restricted to the traces the panel actually split
    on -- the average is dominated by unanimous easy traces, and the split ones
    are where a judge's uncertainty is supposed to show up.
    """
    all_tv, split_tv = [], []
    for d, pd in zip(dists, panel_dists):
        for a in attributes:
            tv = 0.5 * float(np.abs(np.asarray(d[a]) - np.asarray(pd[a])).sum())
            all_tv.append(tv)
            if max(pd[a]) < 0.99:  # the annotators did not agree unanimously
                split_tv.append(tv)
    return {
        "tv_all": float(np.mean(all_tv)) if all_tv else float("nan"),
        "tv_contested": float(np.mean(split_tv)) if split_tv else float("nan"),
        "n_contested": len(split_tv),
    }


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """p ** (1/T), renormalized -- temperature scaling on the implied logits."""
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0)) / max(T, 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray, *, bounds: tuple[float, float] = (0.05, 10.0)) -> float:
    """Fit the single scalar T minimizing NLL. T > 1 softens an overconfident judge."""
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        p = apply_temperature(probs, T)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def fit_temperature_guarded(
    probs: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> dict[str, Any]:
    """Fit T, then only recommend it if it actually improves ECE on the fitting split.

    Temperature scaling minimizes NLL, which is not the metric anyone reports. On an
    already well-calibrated judge -- and on a small calibration split -- the NLL optimum
    routinely overshoots and turns mild overconfidence into worse underconfidence. A
    remedy that is applied unconditionally is not a remedy.

    Returns the fitted `T`, whether it helped (`adopt`), and the before/after ECE on the
    calibration data so the caller can report the decision rather than assume it.
    """
    T = fit_temperature(probs, y, bounds=bounds)
    before = ece_from(probs, y, n_bins=n_bins)
    after = ece_from(apply_temperature(probs, T), y, n_bins=n_bins)
    return {
        "T": T,
        "adopt": after.ece < before.ece,
        "ece_before": before.ece,
        "ece_after": after.ece,
        "gap_before": before.signed_gap,
        "gap_after": after.signed_gap,
        "n": len(y),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def reliability_curve(
    series: dict[str, ECEResult],
    *,
    title: str = "Reliability",
    ax=None,
    show_counts: bool = True,
):
    """Overlay reliability diagrams for several judges, with bin counts underneath.

    `series` maps a label ("prompted baseline", "fine-tuned", ...) to its ECEResult.
    A curve below the diagonal is overconfident: it claimed more certainty than
    its accuracy earned.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        if show_counts:
            fig, axes = plt.subplots(
                2, 1, figsize=(6, 6.4), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
            ax, ax_hist = axes
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax_hist = None
    else:
        ax_hist = None

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration", zorder=1)
    for label, r in series.items():
        xs = [b["mean_confidence"] for b in r.bins if b["count"] > 0]
        ys = [b["accuracy"] for b in r.bins if b["count"] > 0]
        ax.plot(xs, ys, "o-", lw=1.8, ms=5, zorder=3,
                label=f"{label} (ECE={r.ece:.3f}, gap={r.signed_gap:+.3f})")
    ax.set_ylabel("accuracy in bin")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    if ax_hist is not None:
        width = 0.9 / max(len(series), 1)
        for i, (label, r) in enumerate(series.items()):
            centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in r.bins]
            counts = [b["count"] for b in r.bins]
            span = (r.bins[0]["bin_hi"] - r.bins[0]["bin_lo"]) if r.bins else 0.1
            offs = (i - (len(series) - 1) / 2) * width * span
            ax_hist.bar([c + offs for c in centers], counts, width=width * span, label=label, alpha=0.85)
        ax_hist.set_xlabel("predicted confidence")
        ax_hist.set_ylabel("count")
        ax_hist.grid(alpha=0.25, axis="y")
    else:
        ax.set_xlabel("predicted confidence")
    return ax


# ---------------------------------------------------------------------------
# Latency and cost
# ---------------------------------------------------------------------------

def cost_report(
    latencies: Sequence[float],
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    *,
    price_in_per_m: float,
    price_out_per_m: float,
) -> dict[str, float]:
    """p50/p95 latency and dollars per 1000 judgments.

    Prices are per million tokens and must be supplied by the caller -- hard-coding
    a price list into a notebook guarantees it is wrong within a quarter.
    """
    lat = sorted(latencies)
    n = len(lat) or 1
    pin = float(np.mean(prompt_tokens)) if len(prompt_tokens) else 0.0
    pout = float(np.mean(completion_tokens)) if len(completion_tokens) else 0.0
    per_call = pin / 1e6 * price_in_per_m + pout / 1e6 * price_out_per_m
    return {
        "p50_latency_s": float(np.percentile(lat, 50)) if lat else float("nan"),
        "p95_latency_s": float(np.percentile(lat, 95)) if lat else float("nan"),
        "mean_prompt_tokens": pin,
        "mean_completion_tokens": pout,
        "usd_per_1k": per_call * 1000,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def format_row(name: str, kap: dict[str, Any], cal: ECEResult, cost: dict[str, float] | None = None) -> str:
    ci = kap.get("ci95", (float("nan"), float("nan")))
    base = (f"{name:<26s} {kap['mean']:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
            f"  ECE={cal.ece:.3f} ({cal.signed_gap:+.3f})")
    if cost:
        base += f"  p50={cost['p50_latency_s']:.2f}s  ${cost['usd_per_1k']:.2f}/1k"
    return base


HEADER = f"{'judge':<26s} {'mean kappa [95% CI]':<22s} {'ECE (signed)':<20s} {'latency / cost'}"
