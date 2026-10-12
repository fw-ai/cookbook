# Starter jobs

Offer these when the user is greenfield, asks what they can try, or wants a
first successful training run. Prefer a starter over a long interview when the
user has not named their own dataset.

Read this file after `cookbook-catalog.md`. Selecting a starter sets
`completion_gate_bypassed: true` and fills the Research handoff for Configure.
Configure still confirms path, live model availability, and cost before spend.

## Onboarding promotion

Fireworks Serverless Training API LoRA SFT on
`accounts/fireworks/models/muse-glimmer-30b` has a promotional `$0` SFT
training rate for jobs started before `2026-11-01T00:00:00Z`.

This is a dated Fireworks product term last confirmed `2026-10-12`; there is
not yet a public promotion URL. Verify the account's current entitlement or
serverless pool terms before asserting that the payable training amount is
`$0`. If that cannot be verified, describe the expected promotional rate as
pending account confirmation.

The promotion covers Muse Glimmer Serverless SFT training only. Do not extend
it to managed training, DPO, RL, paid sampling or inference, dedicated
resources, deployment, or usage after the deadline. Before calling the starter
free:

1. Confirm the current UTC time is before the deadline.
2. Confirm the Muse Glimmer serverless pool is available to the account.
3. Keep non-SFT work out of the plan or disclose it as a separate paid line.
4. After the deadline, ignore this section and resolve the live rate from
   <https://fireworks.ai/pricing>.

## Preloaded datasets

The following dataset IDs are account-preloaded only in eligible onboarding
accounts:

| Dataset ID | Task | Rows |
|---|---|---:|
| `onboarding-prompt-router` | Text prompts → routing JSON | 600 |
| `onboarding-cord-receipts` | Receipt images → structured JSON | 300 |

Never hard-code the account that authored or staged these datasets. Resolve
them as `accounts/<current-account>/datasets/<dataset-id>` and verify `READY`
with a read-only SDK/API lookup or `firectl dataset get <dataset-id>` before
planning a run. For a Serverless Training API recipe, download the dataset to
the private run directory through the SDK/API when supported, otherwise use
`firectl dataset download <dataset-id> --output-dir <run-data-dir>`, then pass
the resulting JSONL path to the recipe.

When authenticated, check dataset and pool eligibility before offering a Muse
starter. Before authentication, label both Muse options **conditional on
eligible onboarding-account access**. If either resource is absent, offer
`rl_deepmath` or `explore_custom` instead.

## Built-in starters

| Option ID | Goal | Preloaded data | Exact training entry | Eval status |
|---|---|---|---|---|
| `muse_prompt_router` | First text SFT: teach routing JSON | `onboarding-prompt-router` (600 rows) | `training/case-studies/sft_prompt_router/prompt_router_serverless.ipynb` | Inline exact-match and field metrics; adapt its loader and model config |
| `muse_cord_receipts` | Vision SFT: receipt image → JSON | `onboarding-cord-receipts` (300 rows) | `training/examples/sft/train_sft_serverless.py` | Gap: adapt JSON F1 from `training/case-studies/sft_cord_receipts/cord_receipt_sft_sdk.ipynb` without assuming its paid deployment path |
| `rl_deepmath` | Verifiable math RL smoke | `training/examples/rl/deepmath/dataset.jsonl` (200 rows) | `training/examples/rl/deepmath/train_deepmath_serverless.py` | Inline verifier score |
| `explore_custom` | I have my own task or data | — | continue interview | — |

## AskQuestion — What should we try first?

Telemetry: `question_id: research-starter`, field `intake_starter_job`

Title: `Research`

Prompt: `What should we try first?`

Use the four option IDs above. STOP and wait.

## After a starter is selected

1. Skip Q1 / Q2 / Q-eval unless the user chose `explore_custom`.
2. Set:

```yaml
completion_gate_bypassed: true
starter_job_id: <option id>
cookbook_entry_tier: example
cookbook_entry_path: <path from table>
implied_method: sft | rft
suggested_path: serverless
supervision_signal: gold input-output pairs | verifiable answers
dataset_plan:
  source: account_preloaded | cookbook_bundled
  candidates:
    - <preloaded data path>
eval_plan:
  metric: exact_match | json_f1 | verifier_score
  baseline_required: true
  cookbook_eval_hook: recipe_inline | gap
  cookbook_eval_entry: <path when adaptation exists>
match_confidence: high | medium
```

3. Present a short readiness package naming the dataset, method, suggested
   path, held-out baseline requirement, and any adaptation or eval gap. Use
   `medium` confidence for both Muse starters until their account-preloaded
   data renders with the Muse configuration. Do not imply that the CORD
   notebook's managed/deployment evaluator is already a free serverless path.
4. Fire the normal **Handoff** AskQuestion. Prefer `plan_configure`.

## Rules

- Apply the Muse Glimmer promotion only inside the exact model, method,
  surface, and deadline above. Otherwise resolve live rates in Configure.
  Label a positive rate as published price, credit-covered, or estimated
  maximum — never as free unless payable amount is verified as `$0`.
- Prefer serverless or managed LoRA when the live catalog supports them.
  Do not default to dedicated trainers or deployments for a starter.
- Keep the first run tiny: one epoch or a few RL steps, no deployment unless
  the user separately approves serving.
- For Muse Glimmer use base model
  `accounts/fireworks/models/muse-glimmer-30b`, tokenizer
  `meta-models/Muse-Glimmer-30B`, and renderer `muse_glimmer`. Start with a
  bounded subset or one epoch. Render and inspect examples before launch.
- Treat receipt extraction as a vision run. Validate image chunks and visual
  token counts before final approval; do not estimate post-promotion vision SFT
  cost without model-specific visual token accounting.
- If the selected starter's path is unavailable for the chosen model, say so
  and offer the next supported starter or `explore_custom`.
