# Getting started with Fireworks fine-tuning

*Source of truth: [Fine-tuning intro](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md) · [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl.md) — defer to the live docs for current commands/flags.*

The fastest path from a dataset to a deployed fine-tuned model: install + auth, pick a launch surface, run a minimal first SFT job, and read your quota.

## 1. Install `firectl` and sign in

`firectl` is the Fireworks CLI ([docs](https://docs.fireworks.ai/tools-sdks/firectl/firectl.md)):

```bash
# macOS (Homebrew)
brew tap fw-ai/firectl && brew install firectl
```

```bash
firectl signin            # add <ACCOUNT_ID> if you use custom SSO
firectl whoami            # confirm the signed-in account
```

## 2. Get an API key (use scoped keys by default)

Create a key in the [dashboard](https://app.fireworks.ai/settings/users/api-keys) or via `firectl api-key create`, then `export FIREWORKS_API_KEY=...`.

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

```bash
firectl version                              # record the installed version
firectl whoami                               # signed in? (else: firectl signin)
firectl quota list                           # GPU quota headroom for the shape you will use
firectl model get -a fireworks <MODEL_ID>    # base model shows Tunable: true
```

- **Upgrade an outdated CLI first:** if `firectl version` is behind the latest, run `sudo firectl upgrade` before creating jobs. An older CLI can send fields the backend has removed — e.g. 1.7.16 sends `gradient_accumulation_steps`, which **Training V2 rejects** (`InvalidArgument … deprecated and not supported by Training V2`), blocking `sftj create` until you upgrade.
- **Auth:** either `firectl signin` (interactive) or a `FIREWORKS_API_KEY` environment variable (preferred for agents; use a scoped service-account key, never a personal admin key).
- **Billing:** the account needs an active payment method on file. Without one, job create is rejected up front, and credits alone do not satisfy it. Add a payment method in the billing settings at https://app.fireworks.ai before your first job. Exception: the temporary Muse Glimmer promotion in `../../configure/references/cost-estimation.md` states no credit card is required.
- **Full model path:** always pass the full base-model path `accounts/fireworks/models/<MODEL_ID>`, never a bare name. A bare name resolves against your own account, so the public base model fails to resolve when your acting account is not `fireworks`.
- **Quota vs billing are different gates:** quota is a GPU ceiling; billing is a payment-method / spend control. Check both.

Only after these pass, show the complete resolved plan and receive explicit confirmation before dataset upload or job creation.

## 5. Minimal first SFT job

Datasets use the OpenAI-compatible chat format — JSONL, one example per line, a `messages` array (min 3 examples). → see `references/choose-method.md`.

```bash
firectl model get -a fireworks <MODEL_ID>     # confirm Tunable: true
firectl dataset create my-first-dataset /path/to/data.jsonl
firectl sftj create \
  --base-model accounts/fireworks/models/qwen3-8b \
  --dataset my-first-dataset \
  --output-model my-first-tune
firectl sftj get <JOB_ID>                      # monitor
firectl model list                             # the new LoRA appears here
```

Coding agents run the same managed job through the Python SDK, because
`firectl` blocks agent mutations:

```python
from fireworks import Fireworks

client = Fireworks()  # FIREWORKS_API_KEY; account from FIREWORKS_ACCOUNT_ID or the key
job = client.supervised_fine_tuning_jobs.create(
    supervised_fine_tuning_job_id="my-first-tune",
    dataset=f"accounts/{account}/datasets/my-first-dataset",
    base_model="accounts/fireworks/models/qwen3-8b",
    output_model=f"accounts/{account}/models/my-first-tune",
    eval_auto_carveout=True,
)
client.supervised_fine_tuning_jobs.get("my-first-tune")  # state, job_progress.percent
```

For a free first run, use the Muse Glimmer starter in
`training/examples/sft/muse_glimmer_starter.py`.

- **Always** verify the base model is `Tunable: true` first.
- Never infer a free training tier from model size. Confirm the exact model,
  method, surface, and deadline on the live
  [pricing page](https://fireworks.ai/pricing) or the promotion's source page.

## 6. Check your quota

```bash
firectl quota list      # GPU quotas, rate limits, spend limit, usage
```

- On-demand GPU quota is a **ceiling on concurrent GPUs — not a reservation**; you pay only for what runs, capacity isn't held.
- A **monthly spend limit** pauses all usage when hit. Need more GPUs → [contact Fireworks](https://fireworks.ai/company/contact-us).

## First-successful-job checklist

- [ ] `firectl whoami` = correct account.
- [ ] Dataset uploaded + passes validation (valid JSONL, every line has `messages` with `role`+`content`).
- [ ] Job reaches `COMPLETED`; training loss trends down.
- [ ] New fine-tuned LoRA in `firectl model list`.
- [ ] Deploy it on-demand and get a sensible completion → `references/deploy-and-troubleshoot.md`.

> Wrong text learned or assistant turn ignored? Use Render Samples and `renderer-verification.md`.
> For multimodal jobs, image-bearing records include ordered `rendered_chunks`,
> an `image_count`, and the number of visual tokens contributed by each image.
> Image ranges appear as explicit `<image … visual tokens>` markers rather than
> blank tokenizer output; the artifact intentionally does not duplicate source
> image bytes or asset locations.
