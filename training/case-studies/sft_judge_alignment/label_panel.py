"""A three-annotator panel that stands in for your human labelers.

WHAT IS REAL AND WHAT IS NOT
----------------------------
The traces are real: an actual LangGraph agent, actual tool calls, actual
failures. The *labels* here are not human. They come from one frontier model
prompted as three reviewers who genuinely weight the rubric differently.

That substitution buys the two things a single synthetic label cannot:

  1. a **human-human agreement ceiling** -- pairwise kappa between the three
     annotators, the number every judge result has to be read against;
  2. a **soft label** per trace -- when the panel splits 4/4/2, that spread is
     the uncertainty a well-calibrated judge should reproduce.

It also has a real limitation, and the notebook says so plainly: a judge
fine-tuned on an LLM panel distills that panel, blind spots included. Swap
`label_traces` for a loader over your own annotations and everything downstream
works unchanged -- the only contract is the `panel` / `gold` / `dist` shape
documented in `aggregate`.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from judge_client import SCALE, judge_once

ATTRIBUTES = ["correctness", "efficiency", "tool_use"]

_RUBRIC = """You are reviewing a trace from an AI data-analyst agent that answers questions
about an e-commerce SQL database. Score three attributes on an integer 1-5 scale.

- correctness: is the final answer factually right, given what the tools returned?
- efficiency: did it reach the answer without redundant or wasteful steps?
- tool_use: did it use the right tool with well-formed, targeted queries?

You are also told the true answer. Use it to judge correctness; judge the other two
attributes from the trace itself.

Use the whole scale. Two cases in particular are mid-scale, not 1 and not 5:
- A partially correct answer. If the question asks for two things and the agent gets one
  right, that is a 3 on correctness, not a 1 and not a 5.
- Right method, wrong result. An agent can run a flawless query against data that turns out
  to be wrong or incomplete. Score correctness on the answer and tool_use on the method --
  do not let one drag the other down."""

# Three reviewers who agree on the rubric and disagree on emphasis -- which is
# exactly how real annotation teams disagree.
PERSONAS: dict[str, str] = {
    "strict": _RUBRIC + """

Your review style: strict and correctness-first. A wrong or unstated final answer
caps correctness at 2 no matter how good the process looked. You do not give 5s
unless the work is flawless. Recovering from a self-inflicted error still cost the
user time, so it should not score above 3 on efficiency.""",
    "pragmatic": _RUBRIC + """

Your review style: pragmatic and outcome-focused. What matters is whether the user
got the right answer; a messy path to a correct result is acceptable. You reserve
low scores for agents that mislead the user. An agent that honestly says it cannot
answer is far better than one that guesses, and you score it accordingly.""",
    "process": _RUBRIC + """

Your review style: process-focused. You care most about method: verifying assumptions
before trusting a query, handling errors properly, and filtering correctly (for
example, excluding cancelled orders when the question is about sales). An agent that
gets the right number by luck or by an unverified query does not deserve a high
tool_use score.""",
}

_INSTRUCTION = '\n\nRespond with ONLY a JSON object: {"correctness": <1-5>, "efficiency": <1-5>, "tool_use": <1-5>}'


def build_user_prompt(record: dict[str, Any]) -> str:
    """The trace as shown to an annotator: task, true answer, transcript."""
    from agent_traces import render_trace

    return (
        f"Task given to the agent:\n{record['task']}\n\n"
        f"True answer: {record['gold_answer']}\n\n"
        f"Agent trace:\n{render_trace(record['trace'])}"
    )


def label_traces(
    records: Sequence[dict[str, Any]],
    client: Any,
    model: str,
    *,
    temperature: float = 0.8,
    concurrency: int = 8,
    attributes: Sequence[str] = tuple(ATTRIBUTES),
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run every persona over every trace. Returns records with a `panel` field.

    Temperature is deliberately high: annotators are not deterministic, and a
    panel that always agrees would report a ceiling of 1.0 and teach the reader
    nothing.
    """
    jobs = [(i, p) for i in range(len(records)) for p in PERSONAS]
    done = {"n": 0}

    def work(job):
        i, persona = job
        user = build_user_prompt(records[i])
        out = judge_once(
            client, model, PERSONAS[persona] + _INSTRUCTION, user, attributes,
            temperature=temperature, want_logprobs=False,
        )
        done["n"] += 1
        if progress and done["n"] % 25 == 0:
            print(f"  labels {done['n']}/{len(jobs)}", flush=True)
        return i, persona, out

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(work, jobs))

    labeled = [dict(r, panel={}) for r in records]
    for i, persona, out in results:
        if out.ok:
            labeled[i]["panel"][persona] = out.scores
    return [r for r in labeled if len(r["panel"]) == len(PERSONAS)]


def aggregate(records: Sequence[dict[str, Any]], attributes: Sequence[str] = tuple(ATTRIBUTES)) -> list[dict[str, Any]]:
    """Add `gold` (per-attribute median) and `dist` (empirical panel distribution).

    Median rather than mean: the scale is ordinal, and the median is the label a
    real team would settle on in adjudication. `dist` is what ECE is measured
    against -- it is the panel's own uncertainty, so a judge that emits 0.99 on a
    trace the panel split on is miscalibrated even when its argmax is right.
    """
    out = []
    for r in records:
        gold, dist = {}, {}
        for a in attributes:
            votes = [r["panel"][p][a] for p in r["panel"]]
            gold[a] = int(statistics.median_low(votes))
            dist[a] = [votes.count(s) / len(votes) for s in SCALE]
        out.append(dict(r, gold=gold, dist=dist))
    return out


def ceiling(records: Sequence[dict[str, Any]], attributes: Sequence[str] = tuple(ATTRIBUTES)) -> dict[str, Any]:
    """Two agreement ceilings, because the obvious one is the wrong comparator.

    `pairwise` is annotator-vs-annotator quadratic-weighted kappa. It is the number
    everyone quotes as "human agreement", and it is **not** what a judge should be
    measured against here. The judge is scored against the panel *median*, which is a
    denoised target: averaging three annotators cancels some of each one's individual
    noise. Pairwise kappa carries that noise on both sides of the comparison, so it is
    systematically pessimistic. A judge can legitimately beat it without having learned
    anything idiosyncratic.

    `consensus` is the apples-to-apples version: hold out one annotator, build the
    median from the rest, and score the held-out annotator against it -- exactly the
    comparison the judge faces. This is the ceiling to read judge results against.

    Expect `consensus` > `pairwise`. If a judge clears `consensus` too, *then* you are
    looking at something fitting the panel's idiosyncrasies.
    """
    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score

    personas = sorted(records[0]["panel"].keys()) if records else []
    pairwise_attr: dict[str, float] = {}
    consensus_attr: dict[str, float] = {}
    pairs: dict[str, dict[str, float]] = {}

    for a in attributes:
        ks = {}
        for p, q in combinations(personas, 2):
            x = [r["panel"][p][a] for r in records]
            y = [r["panel"][q][a] for r in records]
            ks[f"{p}/{q}"] = float(cohen_kappa_score(x, y, weights="quadratic", labels=SCALE))
        pairs[a] = ks
        pairwise_attr[a] = sum(ks.values()) / len(ks) if ks else float("nan")

        loo = []
        for p in personas:
            held = [r["panel"][p][a] for r in records]
            rest = [int(statistics.median_low([r["panel"][q][a] for q in personas if q != p]))
                    for r in records]
            loo.append(float(cohen_kappa_score(held, rest, weights="quadratic", labels=SCALE)))
        consensus_attr[a] = sum(loo) / len(loo) if loo else float("nan")

    def _mean(d):
        return sum(d.values()) / len(d) if d else float("nan")

    return {
        # `per_attribute` / `mean` stay on the consensus ceiling: it is the one the
        # notebook compares judges against, so the default should not be the trap.
        "per_attribute": consensus_attr,
        "mean": _mean(consensus_attr),
        "consensus": consensus_attr,
        "consensus_mean": _mean(consensus_attr),
        "pairwise": pairwise_attr,
        "pairwise_mean": _mean(pairwise_attr),
        "pairs": pairs,
        "n": len(records),
    }


def save(records: Sequence[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
