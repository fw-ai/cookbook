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

**One adapter → live merge** (simplest). You **must** pass a deployment shape (or an accelerator type): a bare `firectl deployment create <model>` drops into an **interactive shape picker**, and choosing "Create without using shape" fails with `accelerator_type must be specified for non-embeddings engines`. The interactive prompt also breaks non-interactive / agent / CI use, so always pass the shape explicitly and add `--wait`:
```bash
# preferred: an explicit compatible (BF16) deployment shape
firectl deployment create "accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>" \
  --deployment-id <DEPLOYMENT_ID> \
  --deployment-shape accounts/<ACCOUNT_ID>/deploymentShapes/<SHAPE> --wait
# fallback (no shape): you MUST set the accelerator explicitly
firectl deployment create "accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>" \
  --deployment-id <DEPLOYMENT_ID> \
  --accelerator-type NVIDIA_H100_80GB --wait
```
**Multi-LoRA:**
```bash
firectl deployment create "accounts/fireworks/models/<BASE_MODEL_ID>" --enable-addons
firectl load-lora <FINE_TUNED_MODEL_ID> --deployment <DEPLOYMENT_ID>
# route per request: model="<model_name>#<deployment_name>"
```
> **Addon gotcha:** `--enable-addons` works only on **BF16** shapes (FP8/FP4 reject addons). Use a BF16 shape or live merge.

Docs: [Deploying fine-tuned models](https://docs.fireworks.ai/fine-tuning/deploying-loras), [On-demand deployments](https://docs.fireworks.ai/guides/ondemand-deployments).

## Preemptible deployment (optional eval / batch feature)

Use this only when the installed CLI exposes `--preemptible` in `firectl deployment create --help`. It borrows idle capacity and can be reclaimed at any time. If the flag is absent, use a bounded on-demand deployment and tear it down immediately after evaluation.

> **Eval / batch only.** It can disappear mid-request — never point production or latency-sensitive traffic at it.

Feature probe:

```bash
firectl deployment create --help | grep -q -- '--preemptible'
```

Do not use the flag when this probe fails.

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
- **Dedicated live-merge deployment:** the bare model id returns `NOT_FOUND` on a fresh deployment, and so does `model="<model>#<short-deployment-name>"`. Use the **full** reference `model="<full-model-path>#<full-deployment-resource-path>"` — e.g. `accounts/acme/models/my-tune#accounts/acme/deployments/my-tune-dep`.
- **Multi-LoRA:** route `model="<model_name>#<deployment_name>"`.

- **Playground:** open the deployment in the [dashboard](https://app.fireworks.ai), send prompts.
- **API:** `POST https://api.fireworks.ai/inference/v1/chat/completions`. A base-only response usually means the adapter wasn't promoted/loaded; a `NOT_FOUND` usually means wrong addressing for the serving path above.

> **Reasoning base models (e.g. Qwen3):** the model emits `<think>…</think>` before the answer. Budget enough `max_tokens` for the thinking span — a small value like 16 truncates before the answer (`finish_reason=length`) and makes serving look broken — or disable reasoning, and parse the label **after** `</think>`.

## CLEAN UP — stop the spend

On-demand bills by **GPU-second while replicas are active, even with no traffic**:
```bash
firectl deployment list
firectl deployment delete <DEPLOYMENT_ID>
# If the deployment received inference in the last hour, delete is blocked with
# FailedPrecondition ("pass --ignore-checks to skip this check"). Force teardown now:
firectl deployment delete <DEPLOYMENT_ID> --ignore-checks
```
Lighter: `scale_to_zero` (min/max replicas = 0). Defaults: scale to zero after ~1h idle; min-0 deployments auto-deleted after 7 days idle. A scaled-to-zero deployment returns **`503 DEPLOYMENT_SCALING_UP`** on the first request — add retry/backoff.

### Recover orphaned spend (you lost the deployment name)

If a run's session ended and there is no manifest / no recorded ID, do not assume nothing is billing — scan the account. This is independent of any manifest:

```bash
firectl deployment list -a <ACCOUNT_ID>                 # enumerate ALL deployments
firectl deployment get <DEPLOYMENT_ID> -o json -a <ACCOUNT_ID>   # per hit: State + replica counts
#   -> a deployment with Replica Count / Ready Replica Count > 0 is actively billing
firectl model list -a <ACCOUNT_ID>                      # correlate the fine-tuned model created around that time
firectl rlor-trainer-job list -a <ACCOUNT_ID>           # a dedicated Training API trainer may also still be live
```

Then stop the spend (confirm first; may hit the agent guard → run manually):

```bash
firectl deployment update <DEPLOYMENT_ID> --min-replica-count 0 --max-replica-count 0   # park
firectl deployment delete <DEPLOYMENT_ID> --ignore-checks                                # hard stop
```

## Numerics alignment (why outputs drift)

Training and inference are different code paths; mismatched numerics cause logprob/output drift (and in RL, wasted rollouts):
- **Match precision/quantization** (FP8 / BF16 / FP4) between trainer checkpoints and the deployment shape.
- **Measure logprob divergence** on the same tokens.
- **MoE → Router Replay (R3):** divergence often comes from the router picking different top-K experts; pass `include_routing_matrix: true` + `logprobs: true`. Docs: [Numerics alignment](https://docs.fireworks.ai/fine-tuning/rl-rollout-integration#numerics-alignment), [MoE Router Replay](https://docs.fireworks.ai/guides/rollout-inference#moe-router-replay).

## Benchmark: base vs fine-tuned

An absolute score means little without the before/after delta. Always compare the fine-tuned model against the base model on the **same held-out split**:

1. Run the **base model** on the eval split first (this is the baseline; do it before or during training so you're not blocked).
2. Run the **fine-tuned model** on the same split. Use preemptible only when the installed CLI exposes it; otherwise use a bounded on-demand deployment.
3. Report the delta, per-label for classification (plus a confusion-matrix summary), and your evaluator/rubric score for open-ended tasks. A fine-tune that doesn't beat base on the held-out split is not ready, regardless of training loss.
4. Tear down the eval deployment when done.

## Troubleshooting a failed or stuck job

Full field-observed error table, platform-vs-user triage, and debug steps are in [`error-reference.md`](error-reference.md). Deep Training API debugging lives in `references/` within this skill.

## Critical rules

- **LoRA = on-demand only** (serverless won't serve fine-tuned LoRAs).
- **One adapter → live merge; many → multi-LoRA on a BF16 shape.**
- **Always tear down / `scale_to_zero` when done** — on-demand bills by GPU-second even when idle.
- **Align numerics before trusting outputs** (precision + logprob divergence + R3 for MoE).
- **Before assuming a Fireworks bug, check quota (a GPU ceiling) and billing (suspension = billing-side)** — different controls.
- **A bare "Internal error" or stuck-at-0% is unknown until evidence supports a side** — validate locally, gather method-specific status/progress/request evidence, then classify.
