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

Fine-tuned LoRA serves from an **on-demand deployment** — serverless per-token serving of your *own* fine-tuned LoRA is not available yet. Addressing the inference `model` depends on the deployment type:
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

## Benchmark: base vs fine-tuned

An absolute score means little without the before/after delta. Always compare the fine-tuned model against the base model on the **same held-out split**:

1. Run the **base model** on the eval split first (this is the baseline; do it before or during training so you're not blocked).
2. Run the **fine-tuned model** on the same split. Use a **preemptible deployment** (above) so you don't hold dedicated capacity just to eval.
3. Report the delta, per-label for classification (plus a confusion-matrix summary), and your evaluator/rubric score for open-ended tasks. A fine-tune that doesn't beat base on the held-out split is not ready, regardless of training loss.
4. Tear down the eval deployment when done.

## Troubleshooting a failed or stuck job

Full field-observed error table, platform-vs-user triage, and debug steps are in [`error-reference.md`](error-reference.md). Deep Training-SDK debugging (weight sync, checkpoints, renderer) lives in the `skills/dev` skill.

## Critical rules

- **LoRA = on-demand only** (serverless won't serve fine-tuned LoRAs).
- **One adapter → live merge; many → multi-LoRA on a BF16 shape.**
- **Always tear down / `scale_to_zero` when done** — on-demand bills by GPU-second even when idle.
- **Align numerics before trusting outputs** (precision + logprob divergence + R3 for MoE).
- **Before assuming a Fireworks bug, check quota (a GPU ceiling) and billing (suspension = billing-side)** — different controls.
- **A bare "Internal error" or stuck-at-0% is usually platform-side, not your config** — validate the dataset, then retry / escalate rather than rewriting your setup.
