---
name: fireworks-auto-tune
description: Use public Fireworks firectl for account inspection and customer-controlled LoRA SFT tuning workflows.
---

# Fireworks Auto Tune

Use this skill when a customer asks for Fireworks account/resource help or a
supervised fine-tuning workflow that should run through public Fireworks tools.

## Local Machine Security

Assume you are running on the customer's personal machine. Treat local files,
shell history, environment variables, Fireworks config, datasets, and command
output as customer-private unless the customer explicitly asks to share them.

- Keep API keys, tokens, raw environment dumps, shell history, credential files,
  and secret-bearing config out of chat, logs, and final answers.
- Refer to credentials by variable name, not value. Use
  `"$FIREWORKS_API_KEY"` in commands as a literal environment reference.
- If `FIREWORKS_API_KEY` is missing, ask the customer to configure Fireworks
  credentials securely before continuing. Use secure setup instructions instead
  of asking the customer to paste a key into chat.
- If `FIREWORKS_ACCOUNT` is missing, ask which account id to target or ask the
  customer to set it before continuing.
- Pass `-a "$FIREWORKS_ACCOUNT"` on account-scoped commands instead of relying
  on interactive default-account prompts inside an agent run.
- Keep downloaded datasets, generated slices, eval outputs, and command
  evidence in the current workspace unless the customer asks for another path.

## Getting firectl

If `firectl` is not installed, point the customer to the current Fireworks CLI
docs: `https://docs.fireworks.ai/tools-sdks/firectl/firectl`. Use the official
instructions for the customer's platform, then verify with:

```bash
firectl version
```

For authentication, prefer a customer-managed setup: `firectl signin`, a
securely stored API key, or `FIREWORKS_API_KEY` exported by the customer's shell
or secret manager.

## Operating Contract

- Use public `firectl` and public Fireworks APIs.
- Treat the installed CLI as the command contract. When flags, filters,
  pagination, or output shape matter, read the relevant `firectl ... --help`
  output before choosing commands.
- Pass auth and account explicitly on account-scoped commands:
  `firectl --api-key "$FIREWORKS_API_KEY" -a "$FIREWORKS_ACCOUNT" ...`.
- Preserve exact resource names from command output. Reuse full names such as
  `accounts/<account>/datasets/<id>`, `accounts/<account>/models/<id>`,
  `accounts/<account>/deployments/<id>`, and full fine-tuning job resources.
- Prefer structured output with `-o json` when it is available, and keep command
  evidence in the run workspace.
- Stay on public/customer surfaces: public `firectl`, public Fireworks APIs, and
  customer-visible resources.

## Workflow Scope

This skill covers public `firectl` account/resource help, supervised
fine-tuning workflows, and the deployment lifecycle needed to serve, scale,
observe, and tear down a model. For DPO, RL/RFT/GRPO, RLHF reward-model
training, continued pretraining, native distillation, or other training modes,
route to separate guidance or Fireworks Support. Routers, reservations, and
deployment-shape management are outside the paved path; use the relevant
`firectl ... --help` surface or route to Fireworks Support.

## Account And Resource Questions

Account-state questions are read-only unless the customer explicitly asks to
create, update, delete, train, deploy, or run inference.

- Current balance: use `firectl ... account get -o json`.
- Month-to-date spend: use UTC start-of-month through tomorrow's UTC date as
  the exclusive end:
  `firectl ... billing get-usage --start-time <YYYY-MM-01> --end-time <tomorrow-YYYY-MM-DD> --account-costs-only -o json`.
- Quotas and remaining capacity: use
  `firectl ... quota list -o json --no-paginate`; for a named quota use
  `firectl ... quota get <quota-name> -o json`.
- For "room left before the monthly spend limit", use the `monthly-spend-usd`
  quota output when it includes limit, usage, or remaining capacity. Report
  month-to-date billing usage separately. If the quota output does not expose
  headroom, say that exact headroom is unavailable from the public output.
- Account balance is current amount owed or prepaid balance state; report it
  separately from the monthly spend limit.
- Exact resources: when the customer provides a full dataset, model,
  deployment, or job name, prefer a direct `get` over list/search. If the CLI
  rejects a full name but accepts an id, retry with the id portion and record
  both forms.
- Lists are bounded evidence, not absence proof. Read list help before relying
  on filters, search, pagination, or all-pages behavior.

Keep current account billing state separate from planned spend for future work.
Answer current-spend questions with account evidence, not planned-cost
estimates.
When checking auth, report whether `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT`
are set while keeping secret values private.

## Spend And Execution Boundary

On a customer machine, the customer controls execution. For spendful or
resource-creating work, prefer showing the command plan and cost drivers over
taking action directly. Continue into protected commands only when the customer
has explicitly asked this agent to execute that specific work in the current
conversation.

Protected work includes dataset creation/upload, fine-tuning jobs, deployments,
batch inference jobs, and inference probes. Before protected work, give a terse
preview:

- account;
- model and dataset resources;
- actions that will create or run resources;
- key levers such as epochs, learning rate, LoRA rank, sample counts, replicas,
  and deployment uptime;
- estimated cost by component when available;
- any component whose cost is unknown;
- low-spend posture, such as scale-to-zero for validation deployments.

For guidance-only or planning-only requests, inspect only read-only facts that
change the answer, describe the public path, name missing facts, and keep the
work at the planning stage. For cost-preview-only requests, provide the estimate
and state that running the commands would require a separate customer request.

When spend or resource creation is possible, use a concrete preview instead of
a generic yes/no question. Summarize what the commands would create or run, what
may cost money, and what is unknown. Let the customer decide whether to run the
commands.

## Cost Estimation

For planned work, estimate the spend surface before protected execution. Keep
the estimate simple and explicit; label uncertainty instead of implying false
precision.

- For specific current prices, refer to
  `https://fireworks.ai/pricing#fine-tuning-pricing` rather than hard-coding
  rates into the plan.
- Training cost is driven by base model, billing mode, train tokens, epochs, and
  number of candidate runs. Ordinary LoRA SFT uses token-billed `sft_lora`
  training.
- Inference and eval cost are driven by model, sample count, average input
  tokens, average output tokens, and number of passes.
- Deployment cost is driven by deployment shape or accelerator, replica count,
  and planned uptime. If the hourly rate or shape is unknown, say that instead
  of guessing.
- Include paid eval inference and validation deployment uptime when they are
  part of the requested outcome.
- If a required price is unavailable from public information or current account
  evidence, label that line as unknown and offer to continue, revise, or stop.

## Supported SFT Models

Use this smaller model set as the paved path for public self-serve LoRA SFT.
`firectl model list` may show additional models, but CLI visibility is not full
SFT support proof. Other models may require Fireworks Support coordination,
especially when capacity or serving resources are constrained.

Pick the model from the customer's task and data: task family, output contract,
dataset size, and input/output length profile. Recommend one model by default;
offer a comparison only when the customer asks or the data clearly warrants it.

| Model | Use when |
| --- | --- |
| `accounts/fireworks/models/qwen3-4b` | Lowest-cost small text SFT or smoke. |
| `accounts/fireworks/models/qwen3-8b` | Default small/standard text SFT. |
| `accounts/fireworks/models/qwen3p5-9b` | Standard text SFT when quality matters more than the smallest model. |
| `accounts/fireworks/models/qwen3p5-35b-a3b` | Larger Qwen text SFT with stronger quality target. |
| `accounts/fireworks/models/gemma-4-26b-a4b-it` | Non-Qwen comparison when requested. |
| `accounts/fireworks/models/qwen3-vl-8b-instruct` | Vision-language SFT when the dataset requires multimodal behavior. |

For a named model outside this set, route to Fireworks Support or a separate
experimental plan. Treat slow placement as a capacity signal first; preserve
the job or deployment state before escalating.

## Data Quality

Data quality is the main driver of SFT quality. Check simple format errors in
isolation: unreadable JSONL, missing roles, missing outputs, malformed tool
calls, invalid labels, or train/eval leakage. Inspect the broader dataset
separately for distribution, difficulty, coverage, label quality, and whether
the held-out split measures the customer goal.

Before running SFT, capture the useful data facts: train and eval row counts,
rough input/output token lengths, output schema or label set, split provenance,
and the metric that will decide whether the tuned model helped. If no held-out
split exists, say so and avoid treating training-set behavior as evidence of
generalization.

## Hyperparameter Defaults

If the customer does not pin LoRA SFT hyperparameters, use conservative public
defaults as a low-surprise way to get started and learn whether SFT helps on
the customer's task.

For one unpinned first-pass LoRA SFT run, use:

```text
lora_rank: 8
learning_rate: 1.5e-4
```

For a small first SFT hyperparameter search, use up to this three-cell grid.
The grid gives a small spread of conservative candidate runs.

| Config | LoRA rank | Learning rate |
| --- | ---: | ---: |
| 1 | 8 | 1.5e-4 |
| 2 | 16 | 1.0e-4 |
| 3 | 32 | 5.0e-5 |

Each grid cell is another training run and should appear in the cost preview.

Use the customer's explicit hyperparameters when supplied, unless they are
incompatible with the public create surface.

## Supervised Fine-Tuning

For SFT requests:

1. Clarify or inspect the dataset only as much as needed. Prefer an existing
   full Fireworks dataset resource when one is supplied. If no dataset resource
   or local dataset path is supplied, name that missing fact and keep execution
   planning out of the answer.
2. Choose a supported public base model for the task and preserve the exact
   model id.
3. Read `firectl supervised-fine-tuning-job create --help` before mapping plan
   fields to flags.
4. If a local dataset must be uploaded, read `firectl dataset create --help`
   and create the dataset only after the customer explicitly asks this agent to
   execute dataset creation.
5. Before creating the job, preserve the planned base model, dataset, output
   model target, job id/display id, epochs, learning rate, LoRA rank, context
   length, and evaluation-data choice if one is in scope.
6. Submit through `firectl supervised-fine-tuning-job create` only after the
   customer explicitly asks this agent to run the SFT job.
7. Poll the job in the foreground until a terminal state. Give a final answer
   only after the foreground wait has ended.
8. Read the final job state and take `output_model` from that state. Confirm the
   output model with a direct model read before deployment or final reporting.

For LoRA SFT, include a positive `--lora-rank` from the selected plan so the
LoRA intent is explicit.

## Deployment And Inference Proof

Deploy the promoted SFT output model, not the base model. Preserve the full
deployment resource returned by create, and use that exact
`accounts/<account>/deployments/<id>` value as the inference `model` target.
Keep `firectl deployment create` non-interactive. If you pass a deployment
shape, prefer a full shape version so the installed CLI does not open a picker.

Deployment state alone is not serving proof. When inference is explicitly
requested and in scope, poll a real request to the same endpoint and model
target until HTTP 200 before running eval or reporting serving readiness.
Choose the chat or completions endpoint to match the dataset/output shape.
Treat early routing errors after deployment creation as transient within the
planned readiness window. Run the readiness wait in the foreground.

Inspect deployments read-only with `firectl deployment list` and
`firectl deployment get <deployment>`.

Set replica count with `firectl deployment scale <deployment>
--replica-count N`. Set autoscaling at create or update time:
`--min-replica-count` and `--max-replica-count` bound load-based scaling, and
`--max-with-revocable-replica-count` fills spare capacity with revocable
replicas the platform reclaims when it is needed elsewhere. Tune with
`--scale-up-window`, `--scale-down-window`, `--scale-to-zero-window`, and
`--scaling-schedules`.

Read serving behavior with `firectl deployment-metrics list --metric <name>`
(for example `load` or `latency`) before reporting readiness or scaling
decisions.

Tear down with `firectl deployment delete <deployment>`, or leave a
scale-to-zero deployment in place. Keep teardown a foreground step and record
the final state.

## Customer Report

Keep the final answer compact and customer-facing. Include resource ids, final
state, output model, deployment target when applicable, inference proof when
run, cost when known, metric provenance, and a recommended next step. Keep
private file paths, harness jargon, and internal methodology names out of the
customer report.
