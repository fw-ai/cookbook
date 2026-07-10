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

## 3. Choose your launch surface

Managed fine-tuning is **GA**; the **Training API is private preview** ([request access](https://fireworks.ai/contact-training)). See [overview](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md).

| Surface | What it is | Use when |
| --- | --- | --- |
| **Your coding agent + this skill + `firectl`** | Claude Code / Cursor drives the CLI primitives directly, using this skill as the orchestration layer | Default going forward. See `references/orchestrate-from-agent.md` |
| **Managed Fine-Tuning** | Data + config via UI / `firectl` / REST; runs SFT, DPO, RFT | Standard production tuning |
| **Training API** | Custom Python training loops on Fireworks GPUs | Research / custom loss / RL (preview) |
| ~~**Fireworks Agent (Pilot)**~~ | ~~Plain-English server-side agent (`firectl session ...`)~~ | **Deprecated.** Being decommissioned in favor of the coding-agent + skill path above |

Default: drive the run from your coding agent with `firectl` (see `references/orchestrate-from-agent.md`); use the dashboard for a first look at loss curves. **Do not build on `firectl session *` (Pilot) — it is deprecated.**

## 4. Preflight — verify before you create a job

Creating a job is protected, spend-gated work. Verify all of these first. A rejected create is almost always one of these prerequisites, not a platform bug:

```bash
firectl version                              # installed; some flags (e.g. --preemptible) need >= 1.7.26
firectl whoami                               # signed in? (else: firectl signin)
firectl quota list                           # GPU quota headroom for the shape you will use
firectl model get -a fireworks <MODEL_ID>    # base model shows Tunable: true
```

- **Auth:** either `firectl signin` (interactive) or a `FIREWORKS_API_KEY` environment variable (preferred for agents; use a scoped service-account key, never a personal admin key).
- **Billing:** the account needs an active payment method on file. Without one, job create is rejected up front, and credits alone do not satisfy it. Add a payment method in the billing settings at https://app.fireworks.ai before your first job.
- **Full model path:** always pass the full base-model path `accounts/fireworks/models/<MODEL_ID>`, never a bare name. A bare name resolves against your own account, so the public base model fails to resolve when your acting account is not `fireworks`.
- **Quota vs billing are different gates:** quota is a GPU ceiling; billing is a payment-method / spend control. Check both.

Only after these pass, show the plan and create the job.

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

- **Always** verify the base model is `Tunable: true` first.
- RFT is **free for models under 16B**; SFT/DPO bill per training token ([pricing](https://fireworks.ai/pricing)).

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

> Wrong text learned / assistant turn ignored? Use Render Samples / [Debug SFT tokenization](https://docs.fireworks.ai/fine-tuning/debug-sft-tokenization.md).
