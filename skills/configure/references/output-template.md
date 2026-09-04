# Configure response template

Customers should know they are in **configure**, which **path** was chosen, and
that spend waits on approval.

## Skill banner (required, first line)

```text
**Configure**: plan, run, and monitor Fireworks training.
```

## Before the final plan (intake incomplete)

```text
**Configure**: plan, run, and monitor Fireworks training.

<AskQuestion — path or method from path-intake.md>

**Next:** I'll draft the full plan after this answer. No upload or jobs yet.
```

## Final plan (after path-intake completion gate)

```text
**Configure**: plan, run, and monitor Fireworks training.

**Path**
| | |
|---|---|
| Workflow | Managed / serverless / dedicated |
| Surface | firectl / SDK / Training API recipe |
| Method | SFT / DPO / … |
| Why | <one sentence> |

**Cost** (from `cost-estimation.md`)

| | |
|---|---|
| Recommended path | Managed / Serverless / Dedicated (estimator page) / RL handoff |
| Cost range | $low–$high or **Not calculated** |
| Rate certainty | published / unavailable |
| Usage certainty | dataset-derived / observed / inferred / unknown |

| Line | Estimate | Assumptions / source |
|---|---|---|
| Training | … | Managed or Serverless meters only |
| Reference sampler | … | Serverless DPO only, when applicable |
| Eval / inference | … | if in plan |
| Pair generation | … | if in plan |
| Deployment | … | post-training uptime only; not Dedicated training cost |
| **Total** | … | dominant line called out |

**Supplied inputs** … · **Inferred inputs** (value + source) … · **Assumptions** …

Dedicated SFT/DPO: **Not calculated** — use
<https://docs.fireworks.ai/fine-tuning/cost-estimator>. RL: contact Training team.

Pricing: <live URL> (fetched <UTC>). Never call an estimate a quote.

**Plan** … (account, model, dataset, eval, teardown)

**Waiting for your approval** before any upload or job create.
```

## What configure must never do

- Default to `firectl` without confirming workflow path.
- Skip path intake because the request mentions SFT or a model name.
- Start Training API provisioning without confirming preview access when unknown.
