# Starter jobs

Offer these when the user is greenfield, asks what they can try, or wants a
first successful training run. Prefer a starter over a long interview when the
user has not named their own dataset.

Read this file after `cookbook-catalog.md`. Selecting a starter sets
`completion_gate_bypassed: true` and fills the Research handoff for Configure.
Configure still confirms path, live model availability, and cost before spend.

## Built-in starters

| Option ID | Goal | Method | Preloaded data | Cookbook entry | Suggested path |
|---|---|---|---|---|---|
| `sft_support_triage` | First SFT: structured support triage JSON | SFT | `training/examples/sft/data/support_triage_train.jsonl` (72 rows) | `training/examples/sft/train_sft.py` or `train_sft_serverless.py` | managed or serverless |
| `sft_food_reasoning` | Short reasoning SFT smoke | SFT | `training/examples/sft/food_reasoning.jsonl` (50 rows) | `training/examples/sft/train_sft.py` or `train_sft_serverless.py` | managed or serverless |
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
  source: cookbook_bundled
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

- Do not invent a free tier. Resolve live rates and account coverage in
  Configure. Label a positive rate as published price, credit-covered, or
  estimated maximum — never as free unless payable amount is verified as `$0`.
- Prefer serverless or managed LoRA when the live catalog supports them.
  Do not default to dedicated trainers or deployments for a starter.
- Keep the first run tiny: one epoch or a few RL steps, no deployment unless
  the user separately approves serving.
- If the selected starter's path is unavailable for the chosen model, say so
  and offer the next supported starter or `explore_custom`.
