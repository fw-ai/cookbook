"""Statistics and figure helpers for ``ask_before_acting_walkthrough.ipynb``.

Everything here reads committed files under ``results/`` and draws with the shared
``../common/figures.py``; no key, no network. The notebook keeps the environment code,
the scripted episodes and the live-run cells visible; the plotting plumbing lives here.

Per-task bookkeeping (``by_task``) mirrors the inline summary in ``evaluate_served.py``;
that script keeps its own copy because it imports ``httpx`` and is not reused here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from figures import (  # noqa: E402
    arm_bars,
    cluster_bootstrap_ci,
    curve,
    delta_bars,
    paired_delta_ci,
    save,
    style,
    two_panel,
    wilson_ci,
)

IN_SESSION = "results/2026-10-09/full"
SERVERLESS = "results/2026-10-12/serverless-eval"
GROUPS = {
    "serverless": "Served: serverless chat API\nbase vs adapter, 2026-10-12",
    "in_session": "In-session: training sampler\nstep 0 vs step 20, 2026-10-09",
}
ARM_NAMES = ("Base", "Adapter")


def recorded(what: str) -> None:
    """Every output that comes from results/ announces itself, so it is never mistaken for a live run."""
    print(f"RECORDED RESULT  {what}")


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in (HERE / path).read_text().splitlines() if line.strip()]


def load_arms() -> dict[str, dict[str, list[dict]]]:
    """{evaluation: {arm: episodes}} for the two committed evaluations.

    Served: base and adapter arms of ``episodes.jsonl``. In-session: the step-0 and
    step-20 eval transcripts (base model before training, adapter after).
    """
    served = load_jsonl(f"{SERVERLESS}/episodes.jsonl")
    return {
        "serverless": {"Base": [e for e in served if e["arm"] == "base"],
                       "Adapter": [e for e in served if e["arm"] == "adapter"]},
        "in_session": {"Base": load_jsonl(f"{IN_SESSION}/eval_transcripts/step-0000.jsonl"),
                       "Adapter": load_jsonl(f"{IN_SESSION}/eval_transcripts/step-0020.jsonl")},
    }


# --- statistics -------------------------------------------------------------------------


def by_task(episodes: list[dict]) -> dict[str, list[bool]]:
    """{request: [correct per sampled episode]}; the request text identifies the held-out task."""
    out: dict[str, list[bool]] = {}
    for e in episodes:
        out.setdefault(e["task"]["request"], []).append(bool(e["score"]["right_args"]))
    return out


def pct(stat) -> tuple[float, ...]:
    return tuple(100.0 * x for x in stat)


def _grouped(episodes: list[dict], flag) -> list[list[float]]:
    groups: dict[str, list[float]] = {}
    for e in episodes:
        groups.setdefault(e["task"]["request"], []).append(float(flag(e)))
    return list(groups.values())


def episode_rate(episodes: list[dict]) -> tuple[float, ...]:
    """Attempt accuracy in %; four attempts share a task, so the bootstrap resamples tasks."""
    return pct(cluster_bootstrap_ci(_grouped(episodes, lambda e: e["score"]["right_args"])))


def pass_k(episodes: list[dict]) -> tuple[float, ...]:
    """pass^4 in %: Wilson interval on the per-task all-correct flags."""
    flags = [all(v) for v in by_task(episodes).values()]
    return pct(wilson_ci(sum(flags), len(flags)))


def prose_rate(episodes: list[dict]) -> tuple[float, ...]:
    """Share of attempts (in %) that ended with a prose reply and no tool call, task bootstrap."""
    return pct(cluster_bootstrap_ci(_grouped(episodes, lambda e: e["ended"] == "no_tool_call")))


def paired_delta(adapter: list[dict], base: list[dict]) -> tuple[float, ...]:
    """Adapter minus base pass^4, in points, bootstrapped over the task pairs."""
    a, b = by_task(adapter), by_task(base)
    return pct(paired_delta_ci((float(all(b[k])), float(all(a[k]))) for k in sorted(b)))


def series(arms: dict, stat) -> list[tuple]:
    """One (arm, values, lows, highs) tuple per arm, across the evaluations in GROUPS."""
    out = []
    for arm in ARM_NAMES:
        stats = [stat(arms[g][arm]) for g in GROUPS]
        out.append((arm, [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats]))
    return out


# --- figures ----------------------------------------------------------------------------


def plot_result(arms: dict) -> None:
    """The headline two-panel figure: pass^4 on the left, attempt accuracy on the right."""
    style()
    recorded(f"GLM 5.3 Flash, 16 held-out tasks x 4 episodes per arm, {IN_SESSION} and {SERVERLESS}")
    fig = two_panel(
        groups=list(GROUPS.values()),
        series=series(arms, pass_k),
        left=dict(title="Tasks correct in all 4 attempts (pass^4)", ylabel="% of 16 tasks", fmt="{:.1f}%", ylim=(0, 125)),
        right=dict(title="Attempts correct", ylabel="% of 64 attempts", fmt="{:.1f}%", ylim=(0, 125)),
        right_series=series(arms, episode_rate),
        suptitle="Does the adapter act correctly more often than the base model?",
        figsize=(13, 4.8),
    )
    save(fig, HERE / "figures/aba_result.png")
    plt.show()


def print_result(arms: dict) -> None:
    for g in GROUPS:
        for arm in ARM_NAMES:
            p, plo, phi = pass_k(arms[g][arm])
            c, clo, chi = episode_rate(arms[g][arm])
            n_tasks, n_eps = len(by_task(arms[g][arm])), len(arms[g][arm])
            print(f"{g:<11} {arm:<8} pass^4 {p:5.1f}% [{plo:.0f}, {phi:.0f}] Wilson over {n_tasks} tasks   "
                  f"attempts {c:5.1f}% [{clo:.0f}, {chi:.0f}] task bootstrap over {n_eps} attempts")


def plot_paired_delta(arms: dict) -> None:
    """One bar per evaluation: adapter pass^4 minus base pass^4 on the same tasks."""
    labels, deltas, lo, hi = [], [], [], []
    for g, label in GROUPS.items():
        d, low, high = paired_delta(arms[g]["Adapter"], arms[g]["Base"])
        labels.append(label)
        deltas.append(d)
        lo.append(low)
        hi.append(high)
        print(f"{g:<11} adapter minus base, pass^4: {d:+.1f} points [{low:+.0f}, {high:+.0f}] over 16 tasks")
    recorded("paired by task, same 16 held-out requests in both arms of each evaluation")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    delta_bars(ax, labels, deltas, lo, hi, fmt="{:+.1f}", title="How much better is the adapter on the same 16 tasks?")
    ax.set_ylim(-30, 60)
    fig.tight_layout()
    save(fig, HERE / "figures/aba_paired_delta.png")
    plt.show()


def plot_prose_rate(arms: dict) -> None:
    """Share of attempts per arm that answered in prose instead of calling a tool."""
    recorded("episodes that ended with a prose reply and no tool call, same files as above")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    arm_bars(ax, list(GROUPS.values()), series(arms, prose_rate), fmt="{:.1f}%", ylabel="% of 64 attempts", ylim=(0, 14),
             title="How often did each arm answer in prose instead of acting?")
    fig.tight_layout()
    save(fig, HERE / "figures/aba_prose_rate.png")
    plt.show()
    for g in GROUPS:
        for arm in ARM_NAMES:
            n = sum(e["ended"] == "no_tool_call" for e in arms[g][arm])
            print(f"{g:<11} {arm:<8} {n} of {len(arms[g][arm])} attempts ended in prose")


def plot_training_curve() -> list[dict]:
    """Held-out pass^4 and attempt accuracy at every in-session evaluation (steps 0, 5, 10, 15, 20)."""
    evals = load_jsonl(f"{IN_SESSION}/eval_metrics.jsonl")
    steps = [r["completed_steps"] for r in evals]
    recorded(f"{IN_SESSION}/eval_metrics.jsonl, 16 tasks x 4 episodes at every evaluation")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    curve(ax, steps, [("pass^4 (tasks correct in all 4 samples)", [100 * r["eval/task_all_correct_rate"] for r in evals]),
                      ("episodes correct", [100 * r["eval/correct_rate"] for r in evals])],
          title="When did the held-out numbers move during training?", xlabel="optimizer steps completed", ylabel="%")
    ax.set_ylim(80, 102)
    ax.set_xticks(steps)
    fig.tight_layout()
    save(fig, HERE / "figures/aba_training_curve.png")
    plt.show()
    for r in evals:
        print(f"step {r['completed_steps']:>2}  episodes {100 * r['eval/correct_rate']:5.1f}%  "
              f"pass^4 {100 * r['eval/task_all_correct_rate']:5.1f}%  "
              f"questions {r['eval/mean_questions']:.2f}  unneeded {r['eval/mean_unnecessary_questions']:.2f}")
    return evals


def plot_kept_groups() -> list[dict]:
    """Stacked bars per optimizer step: groups with reward spread (trained on) and groups dropped."""
    train = load_jsonl(f"{IN_SESSION}/metrics.jsonl")
    recorded(f"{IN_SESSION}/metrics.jsonl, 8 tasks x 8 episodes per step")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = [r["step"] for r in train]
    kept = [r["train/kept_groups"] for r in train]
    ax.bar(x, kept, color="#e1812c", label="groups with reward spread (trained on)")
    ax.bar(x, [r["train/groups"] - k for r, k in zip(train, kept)], bottom=kept,
           color="#8a8f98", alpha=0.45, label="groups dropped (all episodes scored alike)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("groups of 8 episodes")
    ax.set_ylim(0, 9.5)
    ax.set_xticks(x)
    ax.set_title("How many groups per step carried training signal?")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    save(fig, HERE / "figures/aba_kept_groups.png")
    plt.show()
    print(f"kept groups per step: min {min(kept)}, max {max(kept)}, mean {sum(kept) / len(kept):.1f} of 8; "
          f"datums per step: {min(r['train/datums'] for r in train)} to {max(r['train/datums'] for r in train)}")
    return train
