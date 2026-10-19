# Training API (custom training loops)

*Source of truth: [Training API intro](https://docs.fireworks.ai/fine-tuning/training-api/introduction.md) · [current serverless comparison](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md#serverless-vs-dedicated) · [public cookbook](https://github.com/fw-ai/cookbook/tree/main) — defer to the live docs/repo for current SDK + recipes, then record the checkout commit used by the run.*

Write or fork a Python training loop; Fireworks runs forward/backward on distributed GPUs. The SDK is **Tinker-compatible**, so Tinker code ports over with minimal changes. For standard supported jobs, prefer managed training.

Docs: https://docs.fireworks.ai/fine-tuning/training-api/introduction · Cookbook (start here): https://github.com/fw-ai/cookbook

> Status: Serverless Training is generally available. Dedicated Training needs access; request it at https://fireworks.ai/contact-training

## Access preflight

Before selecting a Training API path, confirm that the target account was enabled through onboarding or account settings. For dedicated work, read the account and shared shape inventories with `firectl training-shape list`; this proves shape visibility but does not by itself prove serverless entitlement. If no authoritative read-only entitlement signal is available, mark access unverified and ask the user to confirm enablement. Do not discover access by creating a trainer or serverless session.

## Managed training vs Training API

Use **managed training** for standard SFT/DPO/ORPO jobs. Reach for the **Training API** when you need a custom **loss/reward**, **RL with rollouts** (inference-in-the-loop), forward-pass internals (for example MoE routing for R3), distillation, or multi-turn/agentic trajectories.

## Training API infrastructure

| Path | Use when | Cookbook |
|---|---|---|
| **Serverless** | LoRA SFT, DPO, ORPO, RL, distillation, and custom loops on models in the shared pool; per-token billing; no trainer or deployment to provision; session-based training and in-session sampling | `sft_loop` / `dpo_loop` with `serverless=True` (`training/examples/sft/train_sft_serverless.py`, `training/examples/dpo/train_dpo_serverless.py`), the [external SDFT cookbook](https://github.com/fw-ai-external/tinker-cookbook) for SDFT, RL via `training/examples/rl/*/train_*_serverless.py` on `training/recipes/experiment/async_rl_loop_serverless.py` |
| **Dedicated** | Full-parameter training, models not in the serverless pool, sustained workloads, explicit trainer, deployment, checkpoint, and teardown control | The matching `train_*.py` examples and `training/recipes/` with `TrainerConfig` / `DeployConfig` |

Default to Serverless when the model is in the pool and LoRA is enough; that
includes DPO, ORPO, distillation, and custom losses. Read the live
[serverless](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md) and
[dedicated](https://docs.fireworks.ai/fine-tuning/training-api/dedicated.md) pages before choosing.

## How to run RL

RL runs on the Training API. Supply an inline `reward_fn(completion, row) -> float`. It may
read `ground_truth`, another declared reference field, tool outcomes, environment state, or a
judge result. There is no evaluator resource — same shape as Tinker's reward-in-the-loop. This
is fully agent-drivable.

- **Serverless:** start from `training/examples/rl/*/train_*_serverless.py` (on
  `recipes/experiment/async_rl_loop_serverless.py`). Rollouts sample from the session's
  sampler snapshots; nothing is provisioned.
- **Dedicated:** fork `training.recipes.rl_loop` / `async_rl_loop`; the recipe provisions a
  trainer plus a rollout deployment from `TrainerConfig` / `DeployConfig`.

**Managed RFT (`firectl reinforcement-fine-tuning-job create --evaluator <id>`) is deprecated**
and accepts no new jobs. The evaluator material below applies only to jobs that already exist.

### Managed RFT: authoring the eval3 evaluator (deprecated)

`firectl reinforcement-fine-tuning-job create` needs an **eval3 evaluator with an `entry_point`** — legacy evaluators are rejected (`InvalidArgument: managed RFT requires an eval3 evaluator`), and `firectl evaluator create` is deprecated. The code-first way to make one is **eval-protocol** (no UI). Check current evaluator authorization in the live docs and handle the observed role gate below:

```bash
pip install eval-protocol          # provides @reward / EvaluationRow (needs Python 3.10+)
```
```python
from eval_protocol import reward, EvaluationRow          # write the reward as code
@reward
def category_match(row: EvaluationRow) -> float:
    return 1.0 if parse(row.output).get("category") == row.ground_truth else 0.0
```
Running the eval-protocol `pytest` flow **auto-registers** the reward as an eval3 evaluator + the dataset with Fireworks. Treat that command as protected registration and run it only after approval; offline tests should import and call the reward directly without authentication. Do **not** follow it with `firectl rftj create` — managed RFT create is rejected. Use the Training API rollout/reward path above for new work. If you are only inspecting an existing evaluator on an existing job, keep using its resource ID.

**Observed authorization behavior on 2026-07-15:** eval3 creation is a multi-RPC flow (`CreateEvaluatorV2` → `GetEvaluatorUploadEndpoint` → upload → `ValidateEvaluatorUpload` → e2b build). An admin-role identity authored the evaluator with a plain `fw_` API key, while a user-role identity received **`403: admin role required to call CreateEvaluatorV2`**. Treat live docs and current API behavior as authoritative because scoped permissions may change. Do not broaden credentials or work around a 403. Ask an admin to register the evaluator once, then launch managed RFT with its resource ID.

> **But the run is capacity-gated.** RFT provisions a trainer + a rollout deployment; on capacity-constrained accounts this readiness-times-out or fails (0/6 success observed on the shared account). RFT reliability is a **platform** matter, not a skill one — see `references/deploy-and-troubleshoot.md`.

## Ways to supply the reward

Whichever RFT route you use, the reward can come from three places:

1. **An existing evaluator ID** (`accounts/<acct>/evaluators/<id>`) — reuse a registered eval3 evaluator. Managed route.
2. **An inline `reward_fn`** — write the reward as Python in the forked recipe. The agent-drivable default; no evaluator resource, any key.
3. **An LLM-judge rubric** — when correctness isn't a clean 0/1 (tone, quality, open-ended), score completions with a judge model against a rubric. Implement it inside `reward_fn` (call a judge model, map its verdict to a float), or register it as an evaluator. If the user gives no evaluator and no rubric, infer a data-grounded rubric from the training samples and confirm it with them before spending.

## Default workflow: fork a cookbook recipe

> **Cookbook implementation:** use the exact recipe and SDK references routed
> from `../../configure/SKILL.md`. This file explains managed versus Training
> API choices; this reference directory and `training/` contain implementation
> depth.

**Approval gate still applies.** Training API cookbook recipes can create trainers, rollout deployments, checkpoints, promoted models, and paid inference. Before running a recipe or any create/promote/deploy action, show the account, recipe, dataset, full resolved trainer and deployment config, cost drivers, evaluation plan, and teardown plan; get explicit approval for this run. A deployment or materially expanded sweep gets its own approval.

Don't write a loop from scratch — fork a recipe in the `training/` tree of `fw-ai/cookbook`:
- `training/recipes/` — loop scripts (e.g. `async_rl_loop`)
- `training/recipes/experiment/` — experimental infrastructure variants, including async serverless RL and SAO/PPO
- `training/examples/` — worked RL / SFT / DPO / ORPO
- `training/utils/` — config, data loading, losses, metrics

Recipes cover SFT, DPO/ORPO, and RL. The RL loops train GRPO; DAPO, GSPO, CISPO, DRO, DPPO and others are objective modules in `training/utils/rl/algorithm/` that a forked loop calls (see `rl-custom-loss.md`). SAO/PPO and IGPO have their own recipes.

## Quick reference: Managed and Serverless SFT

Both blocks ran live on Muse Glimmer 30B. When the cookbook checkout has the
Muse starter or `sft_loop`, run those and fork them for custom loops; use these
blocks when they are missing. Load `.env` in-process first with an
explicit search from the working directory:
`from dotenv import find_dotenv, load_dotenv; load_dotenv(find_dotenv(usecwd=True))`.

Managed SFT on a dataset already in the account:

```python
from fireworks import Fireworks
c = Fireworks(account_id=ACCOUNT)
job = c.supervised_fine_tuning_jobs.create(
    supervised_fine_tuning_job_id=JOB_ID,
    base_model="accounts/fireworks/models/muse-glimmer-30b",
    dataset=f"accounts/{ACCOUNT}/datasets/{DATASET_ID}",
    output_model=f"accounts/{ACCOUNT}/models/{JOB_ID}",
    epochs=1, lora_rank=8, eval_auto_carveout=True,
)
j = c.supervised_fine_tuning_jobs.get(JOB_ID)   # state, job_progress.percent, status.message
urls = j.model_dump()                           # metricsFileSignedUrl, renderSamplesSignedUrl
```

Serverless Training API loop with base and tuned snapshots (run from
`$COOKBOOK_DIR/training` so the `training.*` helpers import):

```python
import os
from fireworks.training import tinker
from fireworks.training.sdk import FiretitanServiceClient
from training.utils.supervised import build_renderer, render_messages_to_datum
from training.utils.tokenizers import load_tokenizer
from training.renderer import get_text_content

tok = load_tokenizer("meta-models/Muse-Glimmer-30B")
renderer = build_renderer(tok, "meta-models/Muse-Glimmer-30B", "muse_glimmer")
datums = [render_messages_to_datum(r["messages"], renderer=renderer, max_seq_len=2048).datum for r in rows]

service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                 base_url="https://api.fireworks.ai/training/v1/serverless")
try:
    tc = service.create_lora_training_client(base_model="accounts/fireworks/models/muse-glimmer-30b", rank=8)
    for batch in batches(datums):
        fb = tc.forward_backward(batch, "cross_entropy").result()          # fb.metrics["loss:sum"]
        opt = tc.optim_step(tinker.AdamParams(learning_rate=1e-4),
                            emit_grad_norm_metrics=True).result()          # opt.metrics grad norms
    path = tc.save_weights_for_sampler("tuned").result().path
    sampler = service.create_sampling_client(model_path=path, tokenizer=tok)
    out = sampler.sample(prompt=renderer.build_generation_prompt(msgs[:-1]), num_samples=1,
                         sampling_params=tinker.SamplingParams(max_tokens=256, temperature=0.0,
                                                               stop=renderer.get_stop_sequences())).result()
    msg, termination = renderer.parse_response(out.sequences[0].tokens)
    answer = get_text_content(msg)   # drops reasoning (thinking) parts
    sampler.close()
finally:
    service.close()
```

`sample` returns a future: for an eval set, submit every request first, then
collect results, instead of waiting on each one in turn.

Snapshot the base adapter the same way before the first step to compare base
and tuned on one holdout, and keep each example's raw output, parsed answer,
and `termination` for both. Muse Glimmer reasons before answering: a base
sample that ends `MALFORMED` usually ran out of `max_tokens` before the answer,
which is a budget problem, not a capability score. Raise the budget or report
those rows separately. Use the same `max_tokens`, temperature, and stop
sequences for base and tuned, and read a few raw outputs before reporting a
surprising number.

To keep the tuned adapter after the session, promote its sampler checkpoint
before closing; session checkpoints are unreachable once the session ends:

```python
from fireworks.training.sdk import FireworksClient
fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
rows = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
        if r.get("promotable") and "tuned" in r["checkpointName"]]   # ids are prefixed/suffixed; never match by equality
fw.promote_session_checkpoint(name=max(rows, key=lambda r: r["createTime"])["name"],
                              output_model_id="my-tuned-lora", base_model="accounts/fireworks/models/muse-glimmer-30b")
```

`service.close()` ends your use of the session, but the run keeps its slot in
the account's concurrent-run quota until the session expires. Do not tell the
user nothing is in use until then; a new run (Managed or Serverless) on the
same model may queue or be rejected. More in
[`execution-notes.md`](execution-notes.md).

To release the slot immediately, call `service.delete_training_session()`
before `service.close()`, after promoting anything you want to keep. The
`sft_loop` / `dpo_loop` `close_session_on_exit=True` option does this at
teardown, including when the run fails. Either way the session's checkpoints
can no longer be resumed or promoted, so leave it off for runs that a later
process may resume.

## Core SDK primitives

- `forward_backward` — built-in losses by id (e.g. `"cross_entropy"`), no extra forward pass.
- `forward_backward_custom(datums, loss_fn)` — your Python loss; returns per-token logprobs with gradients. **Loss runs locally; forward/backward run on remote GPUs.**
- `forward` — forward-only (e.g. reference-model logprobs); optional `loss_fn_config={"top_k": K}` returns per-token top-K logprobs/indices — see below.
- `forward_projection(datums)` — forward-only raw projection-head outputs (`loss_fn_outputs[i]["projection"]`, shape `[tokens, projection_head_dim]`) for a trainer created with a positive `projection_head_dim`. Any compatible validated training shape can be used; the trainer setting configures the head dimension. Accumulates no gradients; train the head with `forward_backward_custom(..., output="projection")`. See `recipes/experiment/ppo_value_head_loop.py`.
- `optim_step(...)` — optimizer update after gradient accumulation.
- `save_weights_for_sampler()` + `create_sampling_client()` — export a checkpoint + stand up a sampler (weight sync for eval/rollouts).

Loss and datum routing: `training-api-losses.md`.

```python
def loss_fn(data, logprobs_list):   # logprobs_list: per-token, requires_grad
    # return (scalar differentiable loss, {metrics for logging})
    ...
```

### Gradient norm metrics (SFT and RL)

Telemetry defaults to off. Enable it on each optimizer update:

```python
result = training_client.optim_step(
    params, emit_grad_norm_metrics=True,
).result()
print(result.metrics)
```

Log `result.metrics` with the optimizer step. `True` emits global L2
`grad_norm` and `grad_norm_rms`; `"detailed"` adds parameter-bucket norms.
`grad_norm` is after accumulation normalization and before clipping; preserve
the run's normalization setting. Keys may have a `:last` suffix.
The RL loops always request `True`; a fork that needs bucket norms passes
`"detailed"`. `sft_loop` exposes `grad_norm_metrics` (`"off"`, `"basic"`, or
`"detailed"`) and defaults it to `"basic"`. Verify the first step's logs; missing metrics do not mean zero
gradients.

### Top-K logprobs from `forward` (entropy / distributional KL)

`forward` (forward-only, no gradient) accepts an optional `loss_fn_config={"top_k": K}`. When set, the trainer returns per-token top-K logprobs and their vocab ids alongside the requested-token `logprobs`, in the same pass — no separate full-vocab forward needed:

```python
result = training_client.forward(
    data,
    "cross_entropy",                        # loss_fn id; ignored in forward-only mode
    loss_fn_config={"top_k": 20},
).result()

for item in result.loss_fn_outputs:
    logprobs = item["logprobs"].data             # requested-token logprobs, shape [seq_len]
    top_k_logprobs = item["top_k_logprobs"].data  # shape [seq_len, 20], float32, sorted descending
    top_k_indices = item["top_k_indices"].data    # shape [seq_len, 20], int64 vocab ids, same order
```

Use it for per-token distribution statistics a single requested-id logprob can't give you — entropy (`-Σ p·log p`), collision (`Σ p²`), or an approximate KL against a reference model's own top-K. Fields are populated only when `top_k > 0` (`top_k` must be `>= 0`, else `ValueError`). They're for analysis — no built-in loss reads them.

## RL: async loop + rollouts

RL recipe: https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl

`training.recipes.rl_loop` (sync) and `async_rl_loop` (rollout/train overlap; superset, preferred for new work) are the fork-and-customise recipes. The one thing you customise is the **inline reward**, exactly like Tinker:
```python
def reward_fn(completion: str, row: dict) -> float:   # fork this
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    return 1.0 if predicted == truth else 0.0
```
The dataset is JSONL of prompts (`messages`) plus exactly the fields the selected reward reads. `ground_truth` is common for exact-match rewards, but it is not universal. Declare and validate the reward's required fields before launch. The recipe samples rollouts, scores each with `reward_fn`, and trains with the trainer's built-in GRPO loss, clipped against the rollout logprobs, with no reference KL. Key knobs: `completions_per_prompt`, `prompt_groups_per_step`, `eps_clip`/`eps_clip_high`, and `max_head_offpolicy_versions` (async; 0 = strict on-policy). To change the objective, fork the loop's train step (`rl-custom-loss.md`). On Dedicated, the recipe provisions the trainer + rollout deployment and tears them down on close; on Serverless, the serverless examples sample from session snapshots instead. **No registered evaluator needed on this path.**

## Numerics alignment & MoE Router Replay (R3)

Trainer↔inference divergence silently wrecks RL. Required reading: https://docs.fireworks.ai/fine-tuning/rl-rollout-integration#numerics-alignment
- Match **precision/quantization** between trainer checkpoints and the deployment shape.
- Measure **logprob divergence** between trainer forward and rollout inference on the same tokens.
- **MoE → Router Replay (R3):** align the top-K experts the router picks. Inference returns them via `include_routing_matrix: true` + `logprobs: true`, and rollout assembly attaches them to the trainer datums. The RL loops request completion-only R3 automatically for MoE base models. https://docs.fireworks.ai/guides/rollout-inference#moe-router-replay

## Critical rules

- **Start from a cookbook recipe**, not a blank loop.
- **Align numerics** before trusting any RL signal; keep **R3** on for MoE.
- **Dedicated RL only:** use a separate rollout deployment (sized for sampling), not your prod endpoint. Serverless RL needs no deployment.
- **Dedicated RL only: set an explicit deployment shape for the rollout.** RFT provisions a rollout/sampling deployment, and the accelerator is owned by the *deployment shape*. With only a base model and `training_shape=auto`, creation fails: `ValueError: Cannot create a managed deployment without a deployment shape`. Pass a `deployment_shape` (or a `training_shape_id` whose shape references one) — e.g. via the recipe's `--training-shape`. The RFT preflight should check this before launch. (Observed live on qwen3-4b, 2026-07-15.)
- Use `forward_backward_custom` only for a genuinely custom objective; otherwise the built-in path is cheaper.
- Loss must be differentiable w.r.t. `logprobs_list`; the metrics dict is logging-only.
