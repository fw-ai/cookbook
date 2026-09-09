# Training cost estimation

Use this workflow for cost, budget, comparison, or spend-ceiling questions.
Estimation is read-only and never authorizes a billable action.

Run **after** path/method preflight resolves model, dataset size, and workflow
path, and **before** the mandatory final-plan approval gate. Research does
**not** estimate cost; defer all spend numbers to this step.

Live sources:

- [Models and method support](https://docs.fireworks.ai/fine-tuning/models)
- [Serverless Training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
- [Pricing](https://fireworks.ai/pricing)
- [Training cost estimator](https://docs.fireworks.ai/fine-tuning/cost-estimator) for Dedicated only. Do not copy rates out of it.

Shape and catalog context (not rates):
[`models-shapes-and-cost.md`](../../fireworks-training/references/models-shapes-and-cost.md).

## Route

| Workload | Skill action |
|---|---|
| Managed SFT or DPO | Resolve rendered tokens, tuning mode, and live rate; calculate |
| Managed ORPO | Do not calculate until its billing contract is documented |
| Serverless LoRA SFT | Resolve trainer tokens and live rate; calculate a planning range |
| Serverless LoRA DPO | Calculate an unpadded baseline with policy train and one-time reference prefill |
| Dedicated SFT or DPO | Do not calculate. Point to the public cost estimator |
| Vision SFT or DPO | Do not calculate. Model-specific visual token accounting is unavailable |
| Training API RL | Prepare a workload summary and contact the Training team |
| Embedding, IGPO, or distillation | Do not calculate. Prepare the method-specific workload and contact the Training team |
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

Embedding, IGPO, and distillation also stay **Not calculated**. Their training,
rollout, search, teacher-inference, or deployment mix is method-specific; do not
reuse SFT or DPO formulas. Prepare those drivers for the Training team.

## Inputs

Collect supplied values first. Infer only from a local dataset, manifest, or
existing job:

- method and training surface
- base model and tuning mode
- dataset modality
- example or preference-pair count
- epochs or optimizer passes
- rendered sequence lengths, including masking and unrolling behavior
- number of candidate runs (`sweep_cells`)

Before calculating, resolve the exact compatible method, tuning mode, and
training shape or serverless pool. Verify that the longest rendered sequence
does not exceed that selected path's context limit. If availability, shape, or
context is unknown, do not estimate from memory; resolve it from the live model
catalog first.

If the dataset contains images, return **Estimate type:** `Not calculated`.
Model-specific visual token accounting is unavailable. Do not substitute text
token counts or infer image tokens from dimensions.

For eval, pair generation, or post-training deployment uptime, also collect
drivers from preflight (see manifest). If a driver is unknown, list it in
`estimated_cost.unknowns[]` — do not guess.

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
2. Live docs cited in
   `../../fireworks-training/references/models-shapes-and-cost.md`.
3. Read-only `firectl` catalog / user-pasted `--dry-run -o json` output.
4. If unreachable: **L1 formula only**; ask user for current per-unit rates.

Never copy rates into `SKILL.md`, this file, or the run manifest as durable
truth — only snapshot what was used for **this** estimate.

## Managed estimates

Read the current rate at estimation time. Never rely on a remembered rate.
Unless the live pricing source states otherwise, training rates are USD per
1 million billed tokens and all cost results are USD.

```text
billable_tokens_per_run = rendered_dataset_tokens × epochs
cost_per_run = billable_tokens_per_run / 1M × current rate
total_cost = cost_per_run × candidate_runs
planning_range = total_cost × [0.7, 1.3]
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
serverless_sft_planning_range = total_trainer_cost × [0.7, 1.3]
```

For SFT without sampling, the train meter may be the only line. For DPO,
inspect the actual recipe. The current serverless DPO example trains the policy
through `forward_backward_custom` and scores the frozen reference through a
sampling client. Reference scoring is normally performed once per unique pair
for each candidate run, not once per epoch when the recipe cache is reused.

```text
policy_pair_occurrences = number of pairs actually sent across optimizer steps
unique_reference_pairs = number of distinct pairs scored by the reference

unpadded_policy_tokens =
  sum of (2P + C + R) across policy_pair_occurrences
all_uncached_reference_tokens =
  sum of (2P + C + R) across unique_reference_pairs

unpadded_baseline_per_run =
  unpadded_policy_tokens / 1M × train rate
  + all_uncached_reference_tokens / 1M × prefill rate

total_unpadded_baseline = unpadded_baseline_per_run × candidate_runs
```

Return this as **Estimate type:** `Unpadded baseline`, never as a range or upper
bound. Batch padding, evaluation, and generation sampling are excluded. The
current reference loop has not demonstrated prompt caching, so do not discount
reference tokens with the cached-prefill rate. For a full `N`-pair dataset run
for `E` epochs with a persistent reference cache, policy pair occurrences are
`N × E` and unique reference pairs are `N`. For partial, repeated, or
step-limited runs, count actual occurrences and distinct referenced rows
instead of assuming complete epochs.

Serverless bills padded policy sequences. The inflation depends on the dataset,
batch size, and ordering, so do not invent a universal padding coefficient from
one run. The Dedicated `dpo_loop` recipe exposes `group_by_length=True`, but it
must not be recommended as a Serverless replacement because it provisions a
different billing surface. The current `training/examples/serverless_dpo`
runner does not expose length grouping; bucket pairs by rendered length before
batching and state that actual cost can exceed the unpadded baseline.

Add optional evaluation or generation sampling as separate lines only when
their token volumes are known. Do not silently quote train-meter cost as total
DPO cost.

## Dedicated estimates

Do not calculate Dedicated SFT or DPO. Do not quote a Dedicated dollar range,
$/M rate, GPU-hour training cost, or Dedicated vs Tinker comparison.

Those numbers depend on private throughput. The skill must not reconstruct
them from GPU count, GPU price, utilization, or a copied catalog.

Return **Cost result:** `Not calculated`. Send the user to
<https://docs.fireworks.ai/fine-tuning/cost-estimator> for Dedicated planning,
or to <https://fireworks.ai/contact-training> if the estimator URL is not live
yet or the page cannot cover the workload.

## Ancillary lines (Managed / Serverless only)

When the approved plan includes paid eval inference, pair generation, or a
post-training deployment, add separate line items with live inference rates.
Call out when deployment uptime likely dominates (common for small LoRA SFT).
See `../../fireworks-training/references/preference-data-and-evaluators.md` and
`../../fireworks-training/references/deploy-and-troubleshoot.md`.
Do not use these lines to reconstruct Dedicated training cost.

## Output contract

Return in the final plan **Cost** block (see `output-template.md`):

- **Recommended path**
- **Estimate type:** Planning range, Unpadded baseline, or Not calculated
- **Cost result:** two decimals below $100, whole dollars at $100 or more,
  `<$0.01` for a positive sub-cent result, or "Not calculated"
- **Rate certainty:** published or unavailable
- **Usage certainty:** dataset-derived, observed, inferred, or unknown
- **Supplied inputs**
- **Inferred inputs:** value and source
- **Assumptions**
- **Next action**

Never present an inferred value as a bounded quote. A Serverless DPO point value
must be labeled `Unpadded baseline` and list excluded work. Dedicated SFT and
DPO are `Not calculated` in this skill. All RL estimates route to
<https://fireworks.ai/contact-training>.

## Manifest block

Append under `## Approved plan` in `run.md` (see
`../../fireworks-training/references/run-state-and-reporting.md`):

```yaml
estimated_cost:
  level: L0 | L1 | L2 | L3
  route: managed-sft | managed-dpo | managed-orpo | serverless-sft | serverless-dpo | dedicated-sft | dedicated-dpo | vision-sft | vision-dpo | training-api-rl | embedding | igpo | distillation | managed-rft
  pricing_source: https://fireworks.ai/pricing
  pricing_fetched_at_utc:
  recommended_path:
  estimate_type: planning_range | unpadded_baseline | not_calculated
  cost_result: Not calculated | $low–$high | Baseline $value | <$0.01
  rate_certainty: published | unavailable
  usage_certainty: dataset-derived | observed | inferred | unknown
  supplied_inputs: {}
  inferred_inputs: []
  assumptions: []
  lines: []           # training, reference_sampler, eval_inference, pair_generation, deployment
  unknowns: []
  total_low_usd:
  total_high_usd:
  unpadded_baseline_usd:
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
| Model/shape selection | `../../fireworks-training/references/models-shapes-and-cost.md` |
| Pair-gen inference cost | `../../fireworks-training/references/preference-data-and-evaluators.md` |
| RFT multi-turn rollout cost | docs [multi-turn cost comparison](https://docs.fireworks.ai/fine-tuning/multi-turn-cost-comparison) |
| Final plan gate | `configure/SKILL.md` § Mandatory final-plan confirmation |
| Report actuals vs estimate | `../../fireworks-training/references/run-state-and-reporting.md` § Cost |
