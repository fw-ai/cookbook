# Configure response template

Customers should know they are in **configure**, which **path** was chosen, and
that spend waits on approval. Keep every turn short and lead with the result or
the next action.

## Skill banner (required, first line)

```text
**Configure**: plan, run, and monitor Fireworks training.
```

## Before the final plan (one decision still open)

```text
**Configure**: plan, run, and monitor Fireworks training.

<one line: what you will do by default>

<AskQuestion — the single open decision from path-intake.md>
```

Ask only when the answer changes the plan. Otherwise go straight to the final
plan with sensible defaults.

## Final plan (compact form)

```text
**Configure**: plan, run, and monitor Fireworks training.

Ready to launch. Nothing has been created yet.

| | |
|---|---|
| Path | <Managed / Serverless / Dedicated>, <surface>, because <reason> |
| Model | <resolved model and tuning mode> |
| Data | <resolved source, format, rows, and split> |
| Params | <user choices + resolved defaults; label backend-resolved unknowns> |
| IDs | <stable resource IDs> |
| Cost | <sourced estimate, assumptions, and excluded work> |
| Eval | <baseline, holdout, metric, and success threshold> |
| Monitor / teardown | <progress + timeout>; <resources removed or retained> |

Approve and I'll launch and watch it.
```

For the temporary Muse starter values, read
`../../research/references/starter-jobs.md`; do not copy dated model, parameter,
or promotion values into this generic template.

## Final plan (full cost block when cost is non-trivial)

```text
**Path** — Workflow (Managed / serverless / dedicated), surface (SDK / Training
API recipe / firectl run by the user), method, and why in one sentence.

**Cost** (from `cost-estimation.md`)

| | |
|---|---|
| Recommended path | Managed / Serverless / Dedicated (estimator page) / RL handoff |
| Estimate type | Promotional / Planning range / Unpadded baseline / Not calculated |
| Cost result | $0 / $low–$high / Baseline $value / <$0.01 / **Not calculated** |
| Rate certainty | product promotion / published / unavailable |
| Usage certainty | dataset-derived / observed / inferred / unknown |

| Line | Estimate | Assumptions / source |
|---|---|---|
| Training | … | Managed or Serverless meters only |
| Reference sampler | … | Serverless DPO one-time all-uncached prefill |
| Eval / inference | … | if in plan |
| Deployment | … | post-training uptime only; not Dedicated training cost |
| **Total** | … | dominant line called out |

Dedicated SFT/DPO: **Not calculated** — use
<https://docs.fireworks.ai/fine-tuning/cost-estimator>. RL outside the
promotion: contact the Training team.

Pricing: <live URL> (fetched <UTC>). Never call an estimate a quote.

**Plan** — account, model, dataset, eval, IDs, monitoring, teardown.

Approve and I'll launch.
```

## While running

```text
**Configure**: plan, run, and monitor Fireworks training.

<job ID>: RUNNING, 40% (epoch 1/1), last change 2 minutes ago. Next check in
1 minute.
```

## Completion

```text
**Configure**: plan, run, and monitor Fireworks training.

Done: <job ID> COMPLETED in <minutes>. Output model `<name>`.
Train loss <first> → <last>; eval loss <value>.
Cost: training $0 (promotion). Nothing billable is running.
Next: <one concrete option, with its cost>.
```

## What configure must never do

- Route agent mutations through `firectl` or hand the user commands the agent
  can run through the SDK.
- Ask about each parameter separately; show defaults in the plan instead.
- Launch before the user approves the exact plan.
