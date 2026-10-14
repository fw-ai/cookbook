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

The agent creates, checks, and deletes deployments through the Python SDK.
`firectl` blocks agent mutations; the `firectl` blocks below are for the user to
run themselves. A deployment is a paid resource: get separate approval first.

**One adapter → live merge** (simplest). Always set a deployment shape. Find a
deployable one, or pass `"default"` to let the server pick a validated one:

```python
import time

# client = Fireworks(account_id=account), resolved as in getting-started.md
model = f"accounts/{account}/models/<FINE_TUNED_MODEL_ID>"
shapes = client.deployment_shape_versions.match_for_model(model)
print([v.name for v in shapes.deployment_shape_versions or []])  # .../deploymentShapes/<SHAPE>/versions/<v>

dep = client.deployments.create(
    base_model=model,
    deployment_id="<DEPLOYMENT_ID>",
    deployment_shape="accounts/fireworks/deploymentShapes/<SHAPE>",  # or "default"
    min_replica_count=0, max_replica_count=1,
)
deadline = time.monotonic() + 60 * 60   # the SDK does not wait; bound it like firectl --wait
while (d := client.deployments.get("<DEPLOYMENT_ID>")).state != "READY":
    if d.state in ("FAILED", "DELETING", "DELETED") or time.monotonic() > deadline:
        raise SystemExit(f"Deployment {d.name} is {d.state}: {d.status}. Diagnose it, "
                         "then delete it with approval so it stops billing.")
    time.sleep(30)
```

**Multi-LoRA** needs a **BF16** shape (FP8/FP4 reject addons):

```python
shapes = client.deployment_shape_versions.match_for_model(
    "accounts/fireworks/models/<BASE_MODEL_ID>", enable_addons=True)
client.deployments.create(
    base_model="accounts/fireworks/models/<BASE_MODEL_ID>",
    deployment_id="<DEPLOYMENT_ID>",
    deployment_shape="accounts/fireworks/deploymentShapes/<SHAPE>",
    enable_addons=True,
)
# after it is READY:
client.lora.load(model=f"accounts/{account}/models/<FINE_TUNED_MODEL_ID>",
                 deployment=f"accounts/{account}/deployments/<DEPLOYMENT_ID>")
# route per request: model="<model_name>#<deployment_name>"
```

**For a user who runs commands themselves (`firectl`).** A bare `firectl deployment create <model>` opens an
interactive shape picker (choosing "Create without using shape" fails with
`accelerator_type must be specified for non-embeddings engines`), so pass the
shape and `--wait`:

```bash
firectl deployment-shape-version match --model "accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>"
firectl deployment create "accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>" \
  --deployment-id <DEPLOYMENT_ID> \
  --deployment-shape accounts/fireworks/deploymentShapes/<SHAPE> --wait
# Multi-LoRA:
firectl deployment-shape-version match --model accounts/fireworks/models/<BASE_MODEL_ID> --enable-addons
firectl deployment create "accounts/fireworks/models/<BASE_MODEL_ID>" \
  --deployment-shape accounts/fireworks/deploymentShapes/<SHAPE> --enable-addons --wait
firectl load-lora <FINE_TUNED_MODEL_ID> --deployment <DEPLOYMENT_ID>
```
> **Addon gotcha:** addons work only on **BF16** shapes (FP8/FP4 reject addons). Use a BF16 shape or live merge.

Docs: [Deploying fine-tuned models](https://docs.fireworks.ai/fine-tuning/deploying-loras), [On-demand deployments](https://docs.fireworks.ai/guides/ondemand-deployments).

## Preemptible deployment (optional eval / batch feature)

Preemptible is a `firectl` flag the SDK `deployments.create` does not expose, so
it is a **user-run** step: give the user the command below. Use it only
when the installed CLI exposes `--preemptible` in `firectl deployment create --help`. It borrows idle capacity and can be reclaimed at any time. Otherwise the agent creates a bounded on-demand deployment through the SDK and tears it down immediately after evaluation.

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
- Check status: `client.deployments.get("<NAME>")` (or `firectl get deployment <NAME> -a <ACCOUNT_ID>`), then send eval requests like any model endpoint.
- **Tear down when done** to release the borrowed capacity: `client.deployments.delete("<NAME>")` (or `firectl delete deployment <NAME> -a <ACCOUNT_ID>`).

## Smoke-test (a READY state is not serving proof)

`State: READY` does not prove the model serves. Send a **real request and confirm HTTP 200** before reporting success or running an eval. Right after `deployment create`, early routing errors are usually transient within the readiness window, so poll with a short backoff rather than failing on the first error.

Fine-tuned LoRA serves from an **on-demand deployment** — serverless per-token serving of your *own* fine-tuned LoRA is not available yet. Addressing the inference `model` depends on the deployment type:
- **Live-merge deployment:** the bare model id returns `NOT_FOUND` on a fresh deployment, and so does `model="<model>#<short-deployment-name>"`. Use the **full** reference `model="<full-model-path>#<full-deployment-resource-path>"` — e.g. `accounts/acme/models/my-tune#accounts/acme/deployments/my-tune-dep`.
- **Multi-LoRA:** route `model="<model_name>#<deployment_name>"`.

- **Playground:** open the deployment in the [dashboard](https://app.fireworks.ai), send prompts.
- **API:** `POST https://api.fireworks.ai/inference/v1/chat/completions`. A base-only response usually means the adapter wasn't promoted/loaded; a `NOT_FOUND` usually means wrong addressing for the serving path above.

> **Reasoning base models (e.g. Qwen3):** the model emits `<think>…</think>` before the answer. Budget enough `max_tokens` for the thinking span — a small value like 16 truncates before the answer (`finish_reason=length`) and makes serving look broken — or disable reasoning, and parse the label **after** `</think>`.

## CLEAN UP — stop the spend

On-demand bills by **GPU-second while replicas are active, even with no traffic**:
```python
import fireworks

try:
    client.deployments.delete("<DEPLOYMENT_ID>")
except fireworks.APIStatusError as exc:
    # Deployments that served traffic in the last hour fail with FailedPrecondition
    # ("pass --ignore-checks"). Force only after the user approves:
    if "ignore" not in str(exc).lower():
        raise
    client.deployments.delete("<DEPLOYMENT_ID>", ignore_checks=True)
try:
    state = client.deployments.get("<DEPLOYMENT_ID>").state
except fireworks.NotFoundError:
    state = "DELETED"
print(state)  # DELETING or DELETED means teardown went through; anything else, keep going
```
**For a user who runs commands themselves (`firectl`):**
```bash
firectl deployment list
firectl deployment delete <DEPLOYMENT_ID>                  # add --ignore-checks if blocked
```
Lighter: `scale_to_zero` (min/max replicas = 0). Defaults: scale to zero after ~1h idle; min-0 deployments auto-deleted after 7 days idle. A scaled-to-zero deployment returns **`503 DEPLOYMENT_SCALING_UP`** on the first request — add retry/backoff.

### Recover orphaned spend (you lost the deployment name)

If a run's session ended and there is no manifest / no recorded ID, do not assume nothing is billing — scan the account. This is independent of any manifest:

```python
for d in client.deployments.list(page_size=100):   # enumerate ALL deployments
    print(d.name, d.state, d.base_model, d.replica_count)
#   -> replica_count > 0 is actively billing
for m in client.models.list(page_size=100, order_by="create_time desc"):
    print(m.name, m.create_time)                    # correlate the fine-tuned model
```

A dedicated Training API trainer may also still be live; list it read-only with
`firectl rlor-trainer-job list -a <ACCOUNT_ID>` when `firectl` is installed.

Then stop the spend after the user confirms:

```python
dep = client.deployments.get("<DEPLOYMENT_ID>")
client.deployments.update("<DEPLOYMENT_ID>", base_model=dep.base_model,
                          min_replica_count=0, max_replica_count=0)    # park
client.deployments.delete("<DEPLOYMENT_ID>", ignore_checks=True)       # hard stop
```

**For a user who runs commands themselves (`firectl`):**

```bash
firectl deployment list -a <ACCOUNT_ID>
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
