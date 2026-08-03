# Fine-tuning a judge to match human labels

Your regression gates, eval dashboards, and release decisions inherit the quality of one LLM judge.
If that judge disagrees with the people who own quality, every number downstream is confidently
wrong. This case study takes a judge from *prompted frontier model* to *fine-tuned small model* and
measures whether that trade is worth making.

- `[judge_alignment_dedicated.ipynb](judge_alignment_dedicated.ipynb)` — the end-to-end notebook:
  generate traces, build a gold set, iterate a prompted judge, LoRA SFT, deploy, compare, tear down.

**Is this you?** You already run an LLM judge in CI and suspect it does not match your rubric. Or
you want a judge that is cheap enough to run on every trace instead of a sample. Or you were told
"just fine-tune the judge" and want to know what that actually costs.

## What makes this different from a normal SFT case study

Most fine-tuning notebooks report one number going up. A judge has three things that matter and
they trade against each other:

| axis | metric | why it is here |
| --- | --- | --- |
| agreement | quadratic-weighted Cohen's kappa, with bootstrap CI | ordinal 1–5 labels; near-misses should not be punished like large misses |
| calibration | ECE with signed gap, reliability diagrams, Brier decomposition | matching the label is not matching the *uncertainty* |
| cost | p50/p95 latency, $ per 1k judgments | the actual reason to replace a frontier judge |

And one number that bounds all of them: the **human–human agreement ceiling**. A judge scoring
above it has not beaten your annotators, it has learned their noise.

## The data

**Traces are real.** A LangGraph `create_react_agent` running on `ChatFireworks` answers analytics
questions against a small SQLite store, with `run_sql` / `describe_schema` / `calculator` tools.
Four conditions spread the quality:

| tier | share | what is different | what it produces |
| --- | --- | --- | --- |
| `strong` | 17% | capable model, clean prompt, all tools | mostly correct, sometimes over-verifies |
| `stale_schema` | 25% | prompt describes a **pre-migration schema** | real SQL errors, recovery, subtly wrong filters |
| `stale_data` | 33% | queries a **replica missing recent rows** | flawless method, wrong number |
| `no_tools` | 8% | tools removed — the retrieval layer is "down" | refusals, or confident invented numbers |
| `flaky` | 17% | the SQL tool fails on first call | retries, or giving up and guessing |

The two `stale_*` tiers exist to solve a specific measurement problem. A gold set built only from
clean successes and total failures is **all 1s and 5s**, and every judge scores brilliantly on it,
because distinguishing a disaster from a clean run is not the job you are trying to measure. The
notebook flags this in section 3 if it happens.

`stale_data` is the sharper of the two: the method is perfect and the answer is wrong, which forces
`correctness` and `tool_use` apart instead of letting them move together. Several questions also ask
for two things ("which category, and how many units"), so an agent can get the label right and the
quantity wrong — a 3, not a 1 and not a 5.

The shares are deliberately non-uniform for the same reason. `strong` is nearly always a 5 and
`no_tools` nearly always a 1, so an equal split spends the budget on cases nobody disagrees about.
This is stratifying toward the decision boundary, not mirroring production traffic — though some easy
cases stay in, because without them the scale has no anchors. `_stratify` also pairs `stale_data`
only with the ~10 questions whose answer actually changes on the stale replica; random pairing wastes
most of the tier on questions it gets right anyway.

**Labels are simulated, and the notebook says so loudly.** Three annotator personas — strict,
pragmatic, process-focused — grade every trace independently at `temperature=0.8`. The median is
the gold label; pairwise agreement between personas is the ceiling; the spread is the soft label
that calibration is measured against.

This is the honest limitation: **a judge fine-tuned on an LLM panel distills that panel, blind
spots included.** Point `label_panel.load()` at your own 100–200 hand-labeled traces and everything
downstream runs unchanged. `judge_traces.sample.jsonl` shows the required shape.

## Files

| file | what it does |
| --- | --- |
| `agent_traces.py` | LangGraph agent, SQLite store, four quality tiers, trace serialization |
| `label_panel.py` | annotator personas, gold aggregation, agreement ceiling |
| `judge_client.py` | one judge call → scores **plus** a 1–5 probability distribution per attribute |
| `judge_metrics.py` | kappa, ECE, reliability curves, Brier decomposition, temperature scaling, cost |
| `judge_traces.sample.jsonl` | 19 labeled traces showing the input shape |

### How the confidence distributions work

Calibration needs a distribution, not an argmax. Fireworks returns `top_logprobs` even when
`response_format` constrains generation, so `judge_client.extract_dists` reconstructs the text from
the token stream, keeps a char-offset → token-index map, finds `"attr": <digit>` in the final JSON
object, and renormalizes that token's alternatives over the digits 1–5.

The offset map is the fiddly part and it is not optional: reasoning models emit hundreds of tokens
before the JSON, and attribute names split across token boundaries (`'correct'` + `'ness'`). If a
deployment ever stops returning usable logprobs, `JudgeOutput.dist_is_fallback` goes `True` and the
section 6 smoke test fails loudly rather than silently reporting one-hot calibration.

## Running it

Needs `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` in a repo-root `.env`.

```bash
pip install fireworks-ai langchain-fireworks langgraph openai scikit-learn scipy matplotlib python-dotenv
```

Validate the pipeline for a few cents first — 24 traces, writes to `data_smoke/`:

```bash
export JUDGE_ALIGN_SMOKE=1
```

Then unset it for the real run (200 traces). Sections 1–5 are inference only. Training and
deployment are gated behind a confirmation cell that prints the fully resolved plan — account,
dataset, model ids, hyperparameters, cost, teardown — and will not proceed until you set
`CONFIRMED = True` (or `export JUDGE_ALIGN_CONFIRM=1`). The final cell deletes the deployment, or
scales it to zero if Fireworks blocks the delete during the post-traffic cooldown.

Pin `JUDGE_ALIGN_RUN` to resume a run rather than creating a second dataset, job, and deployment
for the same work.

> **Cost.** The gold set and the prompted rounds are serverless inference. The fine-tune is a small
> LoRA job. The deployment bills per GPU-second for as long as it is up — that is the expensive part,
> so do not skip the teardown cell.

## Reading the results

Three things to check, in this order.

**Is the gold set degenerate?** Section 3 flags it if fewer than 25% of labels land in the 2–4 band.
An all-1s-and-5s gold set makes every judge look excellent, because telling a disaster from a clean
run is not the job you are trying to measure.

**Is the baseline already at the ceiling?** Section 5 says so explicitly if it is. When it is,
fine-tuning cannot buy agreement — judge it on latency, cost, and getting a long prompt out of every
call instead.

**Did rubric iteration actually help?** Do not assume it did. On our 194-trace run kappa fell
monotonically across the three rounds (+0.885 → +0.845 → +0.708) while ECE rose and cost tripled.
The cause is worth internalizing: round 1's scale definitions are written in the notebook, and the
annotators never saw them, so each round teaches the judge a rubric that is more precise but less
like the one the labels came from. **Agreement is about match, not quality** — if you are aligning to
a labeled set, lift the judge's scale definitions from the guidelines your annotators actually used.
Sampling few-shot examples from the largest disagreements compounds it, since those are the cases
where the panel itself is least reliable; a falling Brier *resolution* is the tell.

The notebook picks the baseline by kappa rather than assuming the last round wins, so a fine-tune is
never flattered by being compared against a judge you would not ship.

**Which way did calibration move?** Do not assume. The folk wisdom is that fine-tuning on hard argmax
targets sharpens a judge past what the data supports. On our run the opposite happened: ECE fell
0.287 → 0.078 and Brier reliability improved nearly nine-fold, because the model was trained on
medians from a panel that genuinely disagreed and learned to hedge where the panel hedged — while the
prompted frontier judge sat at ~0.95 confidence on nearly everything. Measure it; the reliability
diagram is the artifact that tells you.

Section 7 fits a temperature scalar on a calibration split carved out of train, but **only adopts it
if it improves ECE on that split**. Temperature scaling minimizes NLL, which is not the metric you
report, so on an already-calibrated judge the optimum overshoots and trades mild overconfidence for
worse underconfidence — ours went 0.078 → 0.141 with the signed gap flipping +0.062 → −0.037. A
signed gap changing sign is overcorrection, not a fix.

## Caveats worth repeating

- **Use the right ceiling.** Pairwise annotator-vs-annotator kappa is the number everyone quotes and
  it is the wrong comparator: the judge is scored against the panel *median*, which is denoised,
  while pairwise carries each annotator's noise on both sides. `ceiling()` reports both, and the
  notebook compares judges against the leave-one-out `consensus` figure. Beating pairwise proves
  nothing; beating consensus means fitting the panel's idiosyncrasies.
- **Rubric drift is costlier after fine-tuning.** Updating a prompted judge is a text edit; updating
  a fine-tuned one means relabeling, retraining, redeploying, and re-verifying calibration. Fine-tune
  once the rubric has stopped moving.
- **Report per-attribute ceilings separately.** Annotators see the true answer, so `correctness` is
  close to objective and has a high ceiling, while `efficiency` and `tool_use` stay genuinely
  subjective. Averaging them hides which half of your rubric is contested.
- **Keep the gold set.** It outlives every judge built on it and it is the only artifact here you
  cannot regenerate.
