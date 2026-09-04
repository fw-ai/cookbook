# Training cost estimation

Use this workflow for cost, budget, comparison, or spend-ceiling questions.
Estimation is read-only and never authorizes a billable action.

Run **after** path/method preflight resolves model, dataset size, and workflow
path, and **before** the mandatory final-plan approval gate. Research does
**not** estimate cost; defer all spend numbers to this step.

Live sources:

- [Models and method support](https://docs.fireworks.ai/fine-tuning/models.md)
- [Serverless Training](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md)
- [Pricing](https://fireworks.ai/pricing)
- [Training cost estimator](https://docs.fireworks.ai/fine-tuning/cost-estimator.md) for Dedicated only. Do not copy rates out of it.

Shape and catalog context (not rates): [`models-shapes-and-cost.md`](models-shapes-and-cost.md).

## Route

| Workload | Skill action |
|---|---|
| Managed SFT or DPO | Resolve rendered tokens, tuning mode, and live rate; calculate |
| Serverless LoRA SFT or DPO | Resolve metered trainer and sampler operations; calculate a dataset-derived or rough range |
| Dedicated SFT or DPO | Do not calculate. Point to the public cost estimator |
| Training API RL | Prepare a workload summary and contact the Training team |
| Managed RFT | Excluded from this estimator workflow |

Do not expose, infer, reverse engineer, or reproduce private throughput, MFU,
GPU economics, benchmark coefficients, or Dedicated $/M rates. Dedicated
numbers stay on the public estimator page only.

## Progressive disclosure

Escalate one level at a time. Do not paste rate tables into chat — link
<https://fireworks.ai/pricing> and record `pricing_source` in the manifest.

| Level | When | Show |
|---|---|---|
| **L0 — Route** | Path intake incomplete | Route table above; billing mode drivers only |
| **L1 — Formula** | Preflight done, rates not fetched | Formulas below with unknown rate lines |
| **L2 — Estimate** | Rates resolved from live sources | Full output contract with USD ranges |
| **L3 — Sweep** | Hyperparameter grid | Per-cell cost × cell count |

Dedicated SFT or DPO stays **Not calculated** at every level. Link
<https://docs.fireworks.ai/fine-tuning/cost-estimator> instead.

## Inputs

Collect supplied values first. Infer only from a local dataset, manifest, or
existing job:

- method and training surface
- base model and tuning mode
- example or preference-pair count
- epochs or optimizer passes
- rendered sequence lengths, including masking and unrolling behavior
- number of candidate runs (`sweep_cells`)

For eval, pair generation, or post-training deployment uptime, also collect
drivers from preflight (see manifest). If a driver is unknown, list it in
`cost_estimate.unknowns[]` — do not guess.

For an RL handoff, also collect:

- rollout model, turns, prompt and completion lengths, and concurrency
- reward, verifier, or judge design
- evaluation volume
- target schedule and budget

Use aggregate counts and length statistics. Do not transmit dataset contents
when aggregates are sufficient.

## Resolve rates (live only)

Priority order (same as `configure/SKILL.md` source precedence):

1. Live [pricing page](https://fireworks.ai/pricing) (record URL + UTC fetch time).
2. Live docs `.md` pages cited in `models-shapes-and-cost.md`.
3. Read-only `firectl` catalog / user-pasted `--dry-run -o json` output.
4. If unreachable: **L1 formula only**; ask user for current per-unit rates.

Never copy rates into `SKILL.md`, this file, or the run manifest as durable
truth — only snapshot what was used for **this** estimate.

## Managed estimates

Read the current rate at estimation time. Never rely on a remembered rate.

```text
billable_tokens_per_run = rendered_dataset_tokens × epochs
cost_per_run = billable_tokens_per_run / 1M × current rate
total_cost = cost_per_run × candidate_runs
```

For SFT, use the final rendered dataset and account for masking, multi-turn
unrolling, and reasoning traces. For DPO, include both chosen and rejected
sequences according to current billing documentation. Use the published rate
for the resolved LoRA or full-parameter mode.

For `N` preference pairs, `E` epochs, average prompt tokens `P`, chosen tokens
`C`, and rejected tokens `R`:

```text
pair_tokens = 2P + C + R
policy_tokens = N × E × pair_tokens
reference_tokens = N × pair_tokens
```

The managed DPO rate includes the managed method premium. Do not add a separate
reference line to the published managed rate.

## Serverless estimates

Serverless trainer work is token-priced and has no idle GPU charge.

```text
trainer_tokens_per_run = tokens sent to forward or forward_backward × optimizer passes
trainer_cost_per_run = trainer_tokens_per_run / 1M × train rate
total_trainer_cost = trainer_cost_per_run × candidate_runs
```

For SFT without sampling, the train meter may be the only line. For DPO,
inspect the actual recipe. The current serverless DPO example trains the policy
through `forward_backward_custom` and scores the frozen reference through a
sampling client. Add reference input at prefill and cached-prefill rates, plus
any optional evaluation or generation sampling. Reference scoring is normally
performed once per unique pair for each candidate run, not once per epoch when
the recipe cache is reused.

```text
reference_cost_per_run =
  uncached_reference_tokens / 1M × prefill rate
  + cached_reference_tokens / 1M × cached-prefill rate

total_serverless_cost =
  (trainer_cost_per_run + reference_cost_per_run + optional_sampling_cost)
  × candidate_runs
```

If cache behavior or optional operations are unknown, return a rough range and
list the omitted meters. Do not silently quote train-meter cost as total DPO
cost.

Use these two reference bounds when only aggregate pair lengths are available:

```text
cache_effective_reference =
  N × (P + C + R) / 1M × prefill rate
  + N × P / 1M × cached-prefill rate

all_uncached_reference =
  N × (2P + C + R) / 1M × prefill rate
```

## Dedicated estimates

Do not calculate Dedicated SFT or DPO. Do not quote a Dedicated dollar range,
$/M rate, GPU-hour training cost, or Dedicated vs Tinker comparison.

Those numbers depend on private throughput. The skill must not reconstruct
them from GPU count, GPU price, utilization, or a copied catalog.

Return **Cost range:** `Not calculated`. Send the user to
<https://docs.fireworks.ai/fine-tuning/cost-estimator> for Dedicated planning,
or to <https://fireworks.ai/contact-training> if the page cannot cover the
workload.

## Ancillary lines (Managed / Serverless only)

When the approved plan includes paid eval inference, pair generation, or a
post-training deployment, add separate line items with live inference rates.
Call out when deployment uptime likely dominates (common for small LoRA SFT).
See `preference-data-and-evaluators.md` and `deploy-and-troubleshoot.md`.
Do not use these lines to reconstruct Dedicated training cost.

## Output contract

Return in the final plan **Cost** block (see `output-template.md`):

- **Recommended path**
- **Cost range:** whole dollars when supported; otherwise "Not calculated"
- **Rate certainty:** published or unavailable
- **Usage certainty:** dataset-derived, observed, inferred, or unknown
- **Supplied inputs**
- **Inferred inputs:** value and source
- **Assumptions**
- **Next action**

Never present a point estimate when usage is inferred. Never call an estimate a
quote. Dedicated SFT and DPO are `Not calculated` in this skill. All RL
estimates route to <https://fireworks.ai/contact-training>.

## Manifest block

Append under `## Approved plan` in `run.md` (see `run-state-and-reporting.md`):

```yaml
cost_estimate:
  level: L0 | L1 | L2 | L3
  route: managed-sft | managed-dpo | serverless-sft | serverless-dpo | dedicated-sft | dedicated-dpo | training-api-rl | managed-rft
  pricing_source: https://fireworks.ai/pricing
  pricing_fetched_at_utc:
  recommended_path:
  cost_range: Not calculated | $low–$high
  rate_certainty: published | unavailable
  usage_certainty: dataset-derived | observed | inferred | unknown
  supplied_inputs: {}
  inferred_inputs: []
  assumptions: []
  lines: []           # training, reference_sampler, eval_inference, pair_generation, deployment
  unknowns: []
  total_low_usd:
  total_high_usd:
  dominant_line:
  dedicated_estimator_url: https://docs.fireworks.ai/fine-tuning/cost-estimator  # when route is dedicated-*
  notes:
```

Recompute and bump `level` when method, model, dataset size, sweep breadth, or
deploy plan changes — triggers renewed user confirmation per `configure/SKILL.md`.

## Safety

Cost planning does not satisfy the mandatory final-plan confirmation. Dataset
upload, paid evaluation, job creation, trainer or deployment provisioning,
promotion, and deployment still require the complete resolved plan and the
user's explicit confirmation from `SKILL.md`.

## Cross-references (do not duplicate)

| Topic | Reference |
|---|---|
| Model/shape selection | `models-shapes-and-cost.md` |
| Pair-gen inference cost | `preference-data-and-evaluators.md` |
| RFT multi-turn rollout cost | docs [multi-turn cost comparison](https://docs.fireworks.ai/fine-tuning/multi-turn-cost-comparison.md) |
| Final plan gate | `configure/SKILL.md` § Mandatory final-plan confirmation |
| Report actuals vs estimate | `run-state-and-reporting.md` § Cost |
