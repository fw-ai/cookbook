# Starter jobs

Offer these when the user is greenfield, asks what they can try, or wants a
first successful training run. Prefer a starter over a long interview when the
user has not named their own dataset.

Selecting a starter sets `completion_gate_bypassed: true` and goes straight to
Configure's fast path: read-only preflight plus the complete final plan in the
same turn. Do not add a separate readiness-package or handoff question; the
user's next approval launches the job.

## Onboarding promotion

Source: <https://docs.fireworks.ai/fine-tuning/models>. Muse Glimmer 30B
(`accounts/fireworks/models/muse-glimmer-30b`) is free for jobs started before
`2026-12-01T00:00:00-08:00` (Nov 30) on:

- **Managed LoRA SFT** — the UI, or the same managed job created through the SDK/API.
- **Serverless Training API LoRA SFT.**

Not covered: full-parameter SFT, RL (on any surface), DPO, Managed RFT, Dedicated trainers,
deployments, inference outside the training run, and other models. Say this in
one line; do not list every exclusion unless the user asks.

Before calling a starter free:

1. Confirm the current UTC time is before the deadline. State it unambiguously
   as `2026-12-01T00:00:00-08:00` or "through Nov 30 Pacific".
2. Confirm the chosen surface is enabled for the account (read-only check).
3. Keep paid work (deployment, inference evaluation) out of the plan or show it
   as a separate line that needs its own approval. Say `separately priced`
   unless its live rate was fetched in this transcript.
4. After the deadline, ignore this section. In the same reply, open
   <https://fireworks.ai/pricing> (or the catalog), read the live rate for
   the recommended starter, and give a rough estimate (dataset tokens ×
   epochs × rate, labeled estimated). Do not defer the number to a later plan.

## Preloaded datasets

Eligible onboarding accounts have these account-preloaded datasets:

| Dataset ID | Task | Rows |
|---|---|---:|
| `onboarding-prompt-router` | Text prompts → routing JSON | 600 |
| `onboarding-cord-receipts` | Receipt images → structured JSON | 300 |

Never hard-code the account that authored or staged these datasets. Resolve
them as `accounts/<current-account>/datasets/<dataset-id>` and verify `READY`
with a read-only SDK lookup (`client.datasets.get(<dataset-id>)`).

Managed SFT trains on the dataset in place; do not download or re-upload it.
Account datasets cannot be downloaded with an API key (downloads need a
per-account flag that is off by default, plus `firectl signin` or the console),
so never plan around downloading an account dataset.
To inspect data that already lives in the account:

1. **Before launch:** `client.datasets.get(<id>)` gives state, format, and row
   count.
2. **Right after a Managed SFT job is created:** read its render samples, the
   rows exactly as the platform rendered them (see
   `../../fireworks-training/references/execution-notes.md`).

If a dataset is absent, say so and offer `explore_custom` (bring a small
JSONL).

## Built-in starters

The human-readable walkthrough for every starter below is
`training/examples/muse_glimmer/README.md`; link it when the user wants to run
the steps themselves.

The commands call `training/examples/sft/muse_glimmer_starter.py` and run from
`$COOKBOOK_DIR/training` after the one-time install in
`../../fireworks-training/references/cookbook-code.md`; the installed skill
folder does not contain the scripts.

| Option ID | Goal | Data | Easiest entry | Evaluation |
|---|---|---|---|---|
| `muse_prompt_router` | First text SFT: routing JSON | `onboarding-prompt-router` (600) | Managed SFT: `python -m examples.sft.muse_glimmer_starter launch --surface managed` | Success metric: route accuracy on a holdout, base vs tuned, via the serverless surface. Managed SFT itself reports only eval loss from `eval_auto_carveout` |
| `muse_serverless_own_data` | Code-first SFT on the Serverless Training API with a base-vs-tuned comparison | The user's local chat JSONL | Before approval run `python -m examples.sft.muse_glimmer_starter prepare --file <data.jsonl>` (local, free): it validates the file, writes `train.jsonl` and `holdout.jsonl`, and prints one rendered example; show it. Then run `launch --surface serverless --file <data.jsonl>` | Exact match (per-key accuracy for JSON answers), base vs tuned on the holdout, in-session sampling |
| `muse_cord_receipts` | Vision SFT: receipt image → JSON | `onboarding-cord-receipts` (300) | Managed SFT through the SDK, same pattern as the router starter, with `--max-context-length 8192` for image-token headroom. After launch, confirm `image_token_count` in the job's render samples | JSON-field F1 on a held-out split (needs a paid deployment, priced separately); Managed SFT itself reports eval loss (adapt `training/case-studies/sft_cord_receipts/cord_receipt_sft_sdk.ipynb`) |
| `explore_custom` | I have my own task or data | — | continue with a direct recommendation | — |

RL is not part of the promotion. For a user who wants RL, route to the
Training API RL path (for example `training/examples/rl/deepmath/`) and price it
in Configure like any other paid run.

Default recommendation for "whatever's easiest": `muse_prompt_router` on
Managed SFT. Offer `muse_serverless_own_data` when the user has their own data
and wants to write code, see the training loop, or get a behavioral comparison
without paying for a deployment.

Muse Glimmer public serverless inference was deprecated on 2026-09-25. A tuned
Muse LoRA can be sampled in-session on the Serverless Training API, or served
from a paid on-demand deployment that needs separate approval.

## First turn shape

Answer in under 150 words: one sentence on what is free and until when, the
starter table trimmed to Goal and Data, a one-line recommendation, and one
question. Show the privacy notice after that, once.

## AskQuestion — What should we try first?

Telemetry: `question_id: research-starter`, field `intake_starter_job`

Title: `Research`

Prompt: `What should we try first?`

Use the option IDs above, recommended option first. STOP and wait.

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
supervision_signal: gold input-output pairs | verifiable answers
dataset_plan:
  source: account_preloaded | user_file
  candidates:
    - <preloaded data path>
eval_plan:
  metric: eval_loss | exact_match | json_f1 | verifier_score
  baseline_required: true
  cookbook_eval_hook: recipe_inline | gap
  cookbook_eval_entry: <path when adaptation exists>
match_confidence: high | medium
```

3. In the same turn, run Configure's read-only preflight and show the complete
   final plan (`../../configure/SKILL.md`, Mandatory final-plan gate). Ask for
   approval of that plan. On approval, launch, monitor, and report.

## Rules

- Apply the Muse Glimmer promotion only inside the exact model, methods,
  surfaces, and deadline above. Otherwise resolve live rates in Configure.
  Label a positive rate as published price, credit-covered, or estimated
  maximum — never as free.
- Prefer Managed SFT, then the Serverless Training API. Do not default to
  Dedicated trainers or deployments for a starter.
- Keep the first run tiny: one epoch or a few RL steps, no deployment unless
  the user separately approves serving.
- For Muse Glimmer use base model
  `accounts/fireworks/models/muse-glimmer-30b`, tokenizer
  `meta-models/Muse-Glimmer-30B`, and renderer `muse_glimmer`. On the
  Serverless Training API, set an explicit max sequence length and render and
  inspect one example before launch.
- Treat receipt extraction as a vision run. Validate image chunks and visual
  token counts before final approval; do not estimate post-promotion vision SFT
  cost without model-specific visual token accounting.
- Managed RFT is paused. Route RL starters to the Serverless Training API.
- If the selected starter's path is unavailable for the account, say so and
  offer the next supported starter or `explore_custom`.
