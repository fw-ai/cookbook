# Getting started with Fireworks fine-tuning

*Source of truth: [Fine-tuning intro](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md) · [Python SDK](https://github.com/fw-ai-external/python-sdk) · [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl.md) — defer to the live docs and installed SDK for current calls/flags.*

The fastest path from a dataset to a deployed fine-tuned model: authenticate, pick a launch surface, run a minimal first SFT job, and check your quota.

The agent does every step through the Python SDK with `FIREWORKS_API_KEY`.
`firectl` blocks agent mutations, so each `firectl` block below is for the user
to run themselves; the agent may use `firectl` only for read-only checks when
it is installed.

## 1. Authenticate

Resolve the account, then build an account-scoped client (never print the key).
`Fireworks()` reads the key from `FIREWORKS_API_KEY` but the account only from
`FIREWORKS_ACCOUNT_ID` or `account_id=`; dataset, job, and deployment calls
fail without one:

```python
import os
from fireworks import Fireworks

account = os.environ.get("FIREWORKS_ACCOUNT_ID")
if not account:
    names = [a.name.split("/")[-1] for a in Fireworks().accounts.list(page_size=10)]
    if len(names) != 1:
        raise SystemExit(f"Key reaches {names}; ask the user which account to use.")
    account = names[0]
client = Fireworks(account_id=account)
```

The later snippets assume this `client` and `account`.

**For a user who runs commands themselves (`firectl`):** install the Fireworks CLI ([docs](https://docs.fireworks.ai/tools-sdks/firectl/firectl.md)) and sign in:

```bash
brew tap fw-ai/firectl && brew install firectl   # macOS (Homebrew)
firectl signin            # add <ACCOUNT_ID> if you use custom SSO
firectl whoami            # confirm the signed-in account
```

## 2. Get an API key (use scoped keys by default)

Create a key in the [dashboard](https://app.fireworks.ai/settings/users/api-keys) (or `firectl api-key create`), then `export FIREWORKS_API_KEY=...` or put it in a gitignored `.env`.

- **Always** prefer a least-privilege key over a personal admin key for automation/agents — use a **service account** with a scoped permission preset. See [Service Accounts](https://docs.fireworks.ai/accounts/service-accounts.md).
- **Never** commit a key — use `.env` (gitignored) or a secret manager.

## 3. Training vs serving: which resource is which

"Serverless" and "dedicated" each name two different things on Fireworks: one
for **training**, one for **serving** (inference). Training on a surface does
not decide how the result is served. Always say the full name.

**Training** (pick one per run; see [overview](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md)):

| Name | What it is | Use when | What it creates | Cost |
| --- | --- | --- | --- | --- |
| **Managed fine-tuning** | A job you submit (UI, SDK, `firectl`, REST); Fireworks runs the loop. SFT, DPO, ORPO. New Managed RFT creates are paused. | Plain SFT or DPO on a dataset; easiest path | Job + output model | Per training token |
| **Serverless Training** (Training API) | Your Python loop on a shared, always-on pooled trainer. LoRA SFT, DPO, ORPO, RL, distillation, custom loops. GA. | Your own loop, in-session base-vs-tuned evaluation, RL | A session (closes with your script); a model only if you promote a checkpoint | Per token; in-session sampling is its own line |
| **Dedicated Training** (Training API) | Your loop on a provisioned trainer, plus provisioned deployments for rollouts or evaluation. [Request access](https://fireworks.ai/contact-training). | Full-parameter, a model not in the serverless pool, sustained runs, explicit resume or deployment control | Trainer job + any deployments; you delete them | Per GPU-hour while running |

**Serving** a model:

| Name | What it is | Serves | Cost |
| --- | --- | --- | --- |
| **Serverless inference** | Shared per-token endpoint | Selected base models only; never your trained LoRA | Per token |
| **On-demand deployment** (the docs also say "dedicated deployment") | GPUs reserved for your deployment; live merge (one LoRA) or multi-LoRA | Base models and your trained models | Per GPU-hour while replicas run, even idle; delete or scale to zero |

Rules that prevent the common mix-ups:

- A trained LoRA is served only from an on-demand deployment, whichever
  training surface produced it. Training serverless does not make it servable
  serverless.
- To compare base and tuned without paying for a deployment, sample inside a
  Serverless Training session. Managed jobs report loss only; their task-level
  evaluation needs an on-demand deployment.
- A Dedicated Training run can create deployments for rollouts. Those are
  on-demand deployments and bill until deleted.
- Never write bare "serverless" or "dedicated" in a plan. Write "Serverless
  Training", "serverless inference", "Dedicated Training", or "on-demand
  deployment".

The coding agent, UI, CLI, REST API, and Python SDK are interaction surfaces.
Use the path and method routing in `../../configure/SKILL.md`; serving is in
`deploy-and-troubleshoot.md`.

## 4. Preflight — verify before you create a job

Creating a job is protected, spend-gated work. Verify all of these first. A rejected create is almost always one of these prerequisites, not a platform bug:

```python
model = client.models.get("<MODEL_ID>", account_id="fireworks").model_dump()  # ID, not full path
print(model.get("supervisedLoraTunable"), model.get("useTrainingV2"))  # LoRA SFT supported?
client.datasets.get("<DATASET_ID>")                                    # state READY, format, rows
```

**For a user who runs commands themselves (`firectl`):**

```bash
firectl whoami                               # signed in? (else: firectl signin)
firectl model get -a fireworks <MODEL_ID>    # model support
firectl quota list                           # GPU quota headroom (no SDK equivalent)
```

- **Model support:** read `supervisedLoraTunable` and `useTrainingV2`; the older `tunable` field is retired. See `execution-notes.md` for the full-parameter flag.
- **Quota:** there is no SDK quota call. If `firectl` is installed, `firectl quota list` is a safe read-only check; otherwise ask the user to check quota in the dashboard when a create is rejected for capacity.
- **Upgrade an outdated CLI (when you use `firectl`):** if `firectl version` is behind the latest, run `sudo firectl upgrade`. An older CLI can send fields the backend has removed — e.g. 1.7.16 sends `gradient_accumulation_steps`, which **Training V2 rejects** (`InvalidArgument … deprecated and not supported by Training V2`).
- **Auth:** `FIREWORKS_API_KEY` for the agent (use a scoped service-account key, never a personal admin key); `firectl signin` only for the user's own CLI.
- **Billing:** the account needs an active payment method on file. Without one, job create is rejected up front, and credits alone do not satisfy it. Add a payment method in the billing settings at https://app.fireworks.ai before your first job. Exception: the temporary Muse Glimmer promotion in `../../configure/references/cost-estimation.md` states no credit card is required.
- **Full model path:** always pass the full base-model path `accounts/fireworks/models/<MODEL_ID>`, never a bare name. A bare name resolves against your own account, so the public base model fails to resolve when your acting account is not `fireworks`.
- **Quota vs billing are different gates:** quota is a GPU ceiling; billing is a payment-method / spend control. Check both.

Only after these pass, show the complete resolved plan and receive explicit confirmation before dataset upload or job creation.

## 5. Minimal first SFT job

Datasets use the OpenAI-compatible chat format — JSONL, one example per line, a `messages` array (min 3 examples). → see `references/choose-method.md`.

The agent uploads and launches through the SDK (full dataset upload and polling
are in `training-api.md#quick-reference-managed-and-serverless-sft`):

```python
job = client.supervised_fine_tuning_jobs.create(
    supervised_fine_tuning_job_id="my-first-tune",
    dataset=f"accounts/{account}/datasets/my-first-dataset",
    base_model="accounts/fireworks/models/qwen3-8b",
    output_model=f"accounts/{account}/models/my-first-tune",
    lora_rank=8,              # always explicit: unset means full-parameter
    eval_auto_carveout=True,
)
client.supervised_fine_tuning_jobs.get("my-first-tune")  # state, job_progress.percent
```

**For a user who runs commands themselves (`firectl`):** the same job (check `firectl sftj create --help`
for the LoRA rank flag and pass it explicitly):

```bash
firectl dataset create my-first-dataset /path/to/data.jsonl
firectl sftj create \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset my-first-dataset \
  --output-model my-first-tune
firectl sftj get <JOB_ID>                      # monitor
firectl model list                             # the new LoRA appears here
```

For a free first run, use the Muse Glimmer starter in
`training/examples/sft/muse_glimmer_starter.py`.

- **Always** verify the base model supports LoRA SFT (`supervisedLoraTunable`) first.
- Never infer a free training tier from model size. Confirm the exact model,
  method, surface, and deadline on the live
  [pricing page](https://fireworks.ai/pricing) or the promotion's source page.

## 6. Check your quota

Quota is read with `firectl` (read-only) or in the dashboard; the SDK has no quota call.

```bash
firectl quota list      # GPU quotas, rate limits, spend limit, usage
```

- On-demand GPU quota is a **ceiling on concurrent GPUs — not a reservation**; you pay only for what runs, capacity isn't held.
- A **monthly spend limit** pauses all usage when hit. Need more GPUs → [contact Fireworks](https://fireworks.ai/company/contact-us).

## First-successful-job checklist

- [ ] The client is built for the intended account (`client.accounts.get()`; `firectl whoami` for the user).
- [ ] Dataset uploaded + passes validation (valid JSONL, every line has `messages` with `role`+`content`).
- [ ] Job reaches `COMPLETED`; training loss trends down (read the job's `metrics.jsonl`).
- [ ] Output model exists with `kind` `HF_PEFT_ADDON` (`client.models.get(...)`; `firectl model list` for the user).
- [ ] Deploy it on-demand and get a sensible completion → `references/deploy-and-troubleshoot.md`.

> Wrong text learned or assistant turn ignored? Use Render Samples and `renderer-verification.md`.
> For multimodal jobs, image-bearing records include ordered `rendered_chunks`,
> an `image_count`, and the number of visual tokens contributed by each image.
> Image ranges appear as explicit `<image … visual tokens>` markers rather than
> blank tokenizer output; the artifact intentionally does not duplicate source
> image bytes or asset locations.
