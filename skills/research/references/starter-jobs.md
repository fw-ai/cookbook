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

The following dataset IDs are preloaded into eligible onboarding accounts:

| Dataset ID | Task | Rows |
|---|---|---:|
| `onboarding-prompt-router` | Text prompts → routing JSON | 600 |
| `onboarding-cord-receipts` | Receipt images → structured JSON | 300 |

Never hard-code the account that authored or staged these datasets. Resolve
them as `accounts/<current-account>/datasets/<dataset-id>` and verify `READY`
with `firectl dataset get <dataset-id>` before planning a run. For a Serverless
Training API recipe, download the dataset to the private run directory with
`firectl dataset download <dataset-id> --output-dir <run-data-dir>` and pass
the resulting JSONL path to the recipe.

## Built-in starters

| Option ID | Goal | Method | Preloaded data | Cookbook entry | Suggested path |
|---|---|---|---|---|---|
| `muse_prompt_router` | First text SFT: teach routing JSON | SFT | `onboarding-prompt-router` (600 rows) | `training/examples/sft/train_sft_serverless.py`; `training/case-studies/sft_prompt_router/` for before/after eval | serverless |
| `muse_cord_receipts` | Vision SFT: receipt image → JSON | SFT | `onboarding-cord-receipts` (300 rows) | `training/examples/sft/train_sft_serverless.py`; `training/case-studies/sft_cord_receipts/` for eval | serverless |
| `rl_deepmath` | Verifiable math RL smoke | RFT | `training/examples/rl/deepmath/dataset.jsonl` (200 rows) | `training/examples/rl/deepmath/train_deepmath_serverless.py` | serverless |
| `explore_custom` | I have my own task or data | — | — | continue interview | — |

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
suggested_path: managed | serverless
dataset_plan:
  source: account_preloaded | cookbook_bundled
  candidates:
    - <preloaded data path>
eval_plan:
  metric: exact_match | verifier_score
  baseline_required: true
  cookbook_eval_hook: recipe_inline
```

3. Present a short readiness package naming the dataset, method, suggested
   path, and held-out baseline requirement.
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
