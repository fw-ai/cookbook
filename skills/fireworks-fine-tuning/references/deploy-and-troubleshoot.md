# Deploy your fine-tuned model + troubleshooting

*Source of truth: [Deploying fine-tuned models](https://docs.fireworks.ai/fine-tuning/deploying-loras.md) · [numerics alignment](https://docs.fireworks.ai/fine-tuning/rl-rollout-integration.md) — defer to the live docs for current behavior.*

Take a trained adapter live, smoke-test it, tear it down to stop spend, keep numerics aligned, and recover from common failures.

## Deploy after training (LoRA → on-demand only)

A fine-tuned LoRA **cannot run on serverless** — it needs an **on-demand (dedicated) deployment**. Two methods:

| | **Live merge** | **Multi-LoRA** |
|---|---|---|
| How | LoRA merged into base at deploy → one model | Base + addons; adapters loaded per request |
| # LoRAs | One per deployment | Many per deployment |
| Perf | Matches base | Slightly higher TTFT; lower max throughput |
| Best for | Single model in prod | Experiments / many variants |

**One adapter → live merge** (simplest):
```bash
firectl deployment create "accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>"
```
**Multi-LoRA:**
```bash
firectl deployment create "accounts/fireworks/models/<BASE_MODEL_ID>" --enable-addons
firectl load-lora <FINE_TUNED_MODEL_ID> --deployment <DEPLOYMENT_ID>
# route per request: model="<model_name>#<deployment_name>"
```
> **Addon gotcha:** `--enable-addons` works only on **BF16** shapes (FP8/FP4 reject addons). Use a BF16 shape or live merge.

Docs: [Deploying fine-tuned models](https://docs.fireworks.ai/fine-tuning/deploying-loras), [On-demand deployments](https://docs.fireworks.ai/guides/ondemand-deployments).

## Preemptible deployment (eval / batch — borrow idle capacity)

**The recommended way to evaluate a model without holding dedicated on-demand capacity.** `--preemptible` opts the deployment into *borrowing idle reserved GPUs* rather than reserving capacity for you exclusively, so you don't pay to hold dedicated capacity for the eval. It can be reclaimed (preempted) at any time and is **not guaranteed** — but in practice it usually stays up long enough to run an eval end to end, and it carries no *unique* availability risk (if there's no capacity for a preemptible deployment, there's none for an on-demand one either). Safer than the older eval script, which leaned on client-side pieces that are hard to fix without customer coordination.

> **Eval / batch only.** It can disappear mid-request — never point production or latency-sensitive traffic at it.

Requirements:
- **`firectl` ≥ 1.7.26** (`firectl version`) — `--preemptible` is a newer flag and is **silently ignored** on older builds. Upgrade if below.

```bash
firectl deployment create accounts/fireworks/models/<MODEL> \
  -a <ACCOUNT_ID> \
  --deployment-id <NAME> --display-name <NAME> \
  --deployment-shape accounts/fireworks/deploymentShapes/<SHAPE> \
  --min-replica-count 1 --max-replica-count 1 \
  --preemptible --wait
# e.g. <MODEL>=glm-5p2, <SHAPE>=glm-5p2-minimal
# --wait blocks until ready (1h default; tune with --wait-timeout)
```

- **`--preemptible` is immutable** — set at create, can't be toggled on/off later. To change it, delete + recreate.
- Check status: `firectl get deployment <NAME> -a <ACCOUNT_ID>`, then send eval requests like any model endpoint.
- **Tear down when done** to release the borrowed capacity: `firectl delete deployment <NAME> -a <ACCOUNT_ID>`.

## Smoke-test (a READY state is not serving proof)

`State: READY` does not prove the model serves. Send a **real request and confirm HTTP 200** before reporting success or running an eval. Right after `deployment create`, early routing errors are usually transient within the readiness window, so poll with a short backoff rather than failing on the first error.

Addressing the inference `model` depends on the serving path:
- **Implicit live-merge** (served on the base model's serverless): call by the **fine-tuned model id** directly (`accounts/<acct>/models/<tuned>`). The base merges the adapter per request; no explicit deployment needed.
- **Dedicated live-merge deployment:** target the **deployment** you created. Calling a fresh dedicated deployment by the bare model id can return `NOT_FOUND`; use the deployment's model reference.
- **Multi-LoRA:** route `model="<model_name>#<deployment_name>"`.

- **Playground:** open the deployment in the [dashboard](https://app.fireworks.ai), send prompts.
- **API:** `POST https://api.fireworks.ai/inference/v1/chat/completions`. A base-only response usually means the adapter wasn't promoted/loaded; a `NOT_FOUND` usually means wrong addressing for the serving path above.

## CLEAN UP — stop the spend

On-demand bills by **GPU-second while replicas are active, even with no traffic**:
```bash
firectl deployment list
firectl deployment delete <DEPLOYMENT_ID>
```
Lighter: `scale_to_zero` (min/max replicas = 0). Defaults: scale to zero after ~1h idle; min-0 deployments auto-deleted after 7 days idle. A scaled-to-zero deployment returns **`503 DEPLOYMENT_SCALING_UP`** on the first request — add retry/backoff.

## Numerics alignment (why outputs drift)

Training and inference are different code paths; mismatched numerics cause logprob/output drift (and in RL, wasted rollouts):
- **Match precision/quantization** (FP8 / BF16 / FP4) between trainer checkpoints and the deployment shape.
- **Measure logprob divergence** on the same tokens.
- **MoE → Router Replay (R3):** divergence often comes from the router picking different top-K experts; pass `include_routing_matrix: true` + `logprobs: true`. Docs: [Numerics alignment](https://docs.fireworks.ai/fine-tuning/rl-rollout-integration#numerics-alignment), [MoE Router Replay](https://docs.fireworks.ai/guides/rollout-inference#moe-router-replay).

## Common issues (field-observed)

Distilled from recurring training-support cases. **First triage question: platform-side (retry / escalate, not your config) vs user-side (fix your dataset/config).** A bare error string is often *not* diagnostic on its own; classify by symptom. When a job is stuck or failing, `sftj get <id>` for the status message, then match below.

| Symptom | What it usually is | What to do |
|---|---|---|
| **Stuck in `RUNNING` at 0%, no loss** | Startup, not training: a job can be marked `RUNNING` before the trainer actually starts, so `RUNNING` ≠ progress. Usually the model download or GPU placement hasn't finished. **Platform-side.** | Don't wait indefinitely. No loss after a few minutes → retry once; if it recurs on the same model/shape, escalate. It is not your config. |
| **Fails after a readiness/startup timeout (~3600s)** | Trainer/rollout couldn't become ready in the window (large download, tight capacity). **Platform-side**, often transient. | Retry; if it repeats, it's capacity or a known infra issue → escalate with the job id. |
| **`RESOURCE_EXHAUSTED` / "no capacity" / unschedulable** | Scheduler can't place the job on the requested GPU class/region (large shapes and some regions run hot). **Platform-side (infra)** — not quota, not config. | Retry (capacity frees up), try a smaller shape / different region, or request capacity. Distinct from a 429 quota ceiling. |
| **Bare "Internal error", no detail** | Masking gap: the trainer crashed before writing a status file, so you see a generic message. Cause is **either** platform (failed to download the base model) **or** user (dataset rejected — next row). | Not diagnostic alone. Re-validate the dataset first (cheapest); if it's clean, treat as a platform startup/download crash → retry + escalate. |
| **Dataset rejected / validation error** | Malformed JSONL — rows missing `messages`, bad roles. **User-side.** Can surface as a plain "Internal error" above. | Fix the JSONL (every line needs `messages` with `role`+`content`), re-upload (`dataset create` validates on upload). See `references/choose-method.md`. |
| **400 on a warm-start / reused config** | Known issue where jobs created with validations skipped can 400 when warm-starting or reusing a shape. **Platform-side**, fix rolling out. | Create against a **validated shape** (don't skip validations); retry. If blocked, escalate for a rollout ETA. |
| **429 on job create** | Quota — a **ceiling on concurrent GPUs**, not billing. Retry storms make it worse. **User-side (limits).** | Back off; raise quota ([account quotas](https://docs.fireworks.ai/guides/quotas_usage/account-quotas)) or run fewer concurrent jobs / a smaller model. |
| **Account suspended / "spending limit reached"** | **Billing-side**, distinct from quota — budget cap (even with credits), no payment method, or risk review. | Fix in [Billing](https://fireworks.ai/billing). Not a platform bug. ([suspended with credits](https://docs.fireworks.ai/faq-new/billing-pricing/why-might-my-account-be-suspended-even-with-remaining-credits)) |
| **412 / shape unavailable** | The requested training shape isn't enabled on your account. | Pick a listed shape or request enablement. |
| **Context length rejected (`must be divisible by 16`)** | Config validation: requested max context isn't a multiple of the shape's tiling — and a shape's *advertised* max may not itself be divisible. **User-side.** | Round the context length **down** to a multiple of 16. |
| **Deploy after LoRA serves base behavior / "never works"** | Adapter not promoted/loaded, **or** a deployment image/chart version skew (chart newer than the served image). | Confirm the job finished + model exists; for multi-LoRA confirm `load-lora` and route `model#deployment`. Still base-only on a fresh deploy → likely image/chart skew, escalate. |
| **Multi-LoRA / addon load rejected** | FP8/FP4 shapes reject addons; some model families don't support multi-LoRA at all. | Use a **BF16** shape for addons, or live-merge a single adapter. Confirm the family supports multi-LoRA. |
| **Reused / "leaked" deployment fails on redeploy** | A deployment resource reused from a prior run (or swept as a "leaked resource") leaves a stale ledger entry, so the redeploy fails. **Platform-side.** | Delete the old deployment and create fresh; if a rollout (RFT) deploy keeps failing right after a resource sweep, escalate with the deployment id. |
| **SFT much slower than expected (esp. via Training API)** | Sequences not packed → a large fraction of tokens are padding, so throughput drops sharply. | Enable sequence packing / batch tokenization where available; check token utilization. |
| **RL/GRPO reward collapses or KLD explodes** | Almost always **numerics drift** between trainer and rollout inference (especially MoE routing), not the algorithm. | Align precision + measure logprob divergence; for MoE enable **Router Replay (R3)** — see Numerics alignment above. |

**When escalating**, include the job id, base model + shape, and the exact status message. Do not paste account or customer identifiers into shared or customer-facing places.

## Critical rules

- **LoRA = on-demand only** (serverless won't serve fine-tuned LoRAs).
- **One adapter → live merge; many → multi-LoRA on a BF16 shape.**
- **Always tear down / `scale_to_zero` when done** — on-demand bills by GPU-second even when idle.
- **Align numerics before trusting outputs** (precision + logprob divergence + R3 for MoE).
- **Before assuming a Fireworks bug, check quota (a GPU ceiling) and billing (suspension = billing-side)** — different controls.
- **A bare "Internal error" or stuck-at-0% is usually platform-side, not your config** — validate the dataset, then retry / escalate rather than rewriting your setup.
