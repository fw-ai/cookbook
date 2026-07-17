# Training API (custom training loops)

*Source of truth: [Training API intro](https://docs.fireworks.ai/fine-tuning/training-api/introduction.md) · [current serverless comparison](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md#serverless-vs-dedicated) · [pinned cookbook](https://github.com/fw-ai/cookbook/tree/e3dca98ea6363b7ed7d2ea1f7203b16489451407) — defer to the live docs/repo for current SDK + recipes.*

Write or fork a Python training loop; Fireworks runs forward/backward on distributed GPUs. The SDK is **Tinker-compatible**, so Tinker code ports over with minimal changes. For standard supported jobs, prefer managed training.

Docs: https://docs.fireworks.ai/fine-tuning/training-api/introduction · Cookbook (start here): https://github.com/fw-ai/cookbook

> Status: private preview — request access at https://fireworks.ai/contact-training

## Managed training vs Training API

Use **managed training** for standard SFT/DPO/ORPO/RFT jobs. Reach for the **Training API** when you need a custom **loss/reward**, **RL with rollouts** (inference-in-the-loop), forward-pass internals (for example MoE routing for R3), distillation, or multi-turn/agentic trajectories.

## Training API infrastructure

| Path | Use when | Cookbook |
|---|---|---|
| **Serverless** | Supported-model LoRA SFT or RL, shared pool, per-token billing, no provisioning | `training/examples/serverless_rl/` |
| **Dedicated** | Full-parameter, DPO, broader model/method support, sustained runs, explicit trainer/deployment/checkpoint control | `training/recipes/` |

Read the live [serverless](https://docs.fireworks.ai/fine-tuning/training-api/serverless.md) and [dedicated lifecycle](https://docs.fireworks.ai/fine-tuning/training-api/training-and-sampling.md) pages before choosing.

## Two agent-drivable ways to run RFT/RL

There are two RFT paths, and they differ in **where the reward lives**. This matters a lot when a coding agent is driving:

| Path | Reward | Agent-drivable? |
|---|---|---|
| **Managed RFT** — `firectl reinforcement-fine-tuning-job create --evaluator <id>` | A **registered evaluator resource** (server-side, built in an e2b sandbox, eval v3) | **Yes once the evaluator exists.** Register via **eval-protocol** (`pytest` auto-registers) or the **UI**. Evaluator authoring may require an admin role; a scoped key can still launch with an evaluator it can access. `firectl evaluator create` (V1) is **deprecated**. |
| **Training-API RL** — fork `training.recipes.rl_loop` / `async_rl_loop` | An **inline `reward_fn(completion, row) -> float`** in the forked recipe. It may read `ground_truth`, another declared reference field, tool outcomes, environment state, or a judge result. | **Yes.** No evaluator resource. Same shape as Tinker's reward-in-the-loop. The SDK provisions the trainer + rollout deployment. |

**Prefer the managed path for standard RFT** (same as the managed UI): `firectl reinforcement-fine-tuning-job create --dataset <ds> --evaluator <id>` — it resolves the training shape for you and is proven live (qwen3-4b, 2026-07-15). Reuse an existing evaluator or author one via eval-protocol. **Use the inline-reward recipe (below) for users with Training API access** who need a custom loop/reward, rollouts, or agentic trajectories. Both paths are agent-drivable; they differ in reward location, access, billing, and capability.

### Managed RFT: authoring the eval3 evaluator

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
Running the eval-protocol `pytest` flow **auto-registers** the reward as an eval3 evaluator + the dataset with Fireworks. Treat that command as protected registration and run it only after approval; offline tests should import and call the reward directly without authentication. After registration, run `firectl rftj create --base-model <m> --dataset <ds> --evaluator accounts/<acct>/evaluators/<id>`. The flow is fully code-drivable via eval-protocol. If the current account role cannot register it, an admin registers it once and the scoped agent continues with the evaluator resource.

**Observed authorization behavior on 2026-07-15:** eval3 creation is a multi-RPC flow (`CreateEvaluatorV2` → `GetEvaluatorUploadEndpoint` → upload → `ValidateEvaluatorUpload` → e2b build). An admin-role identity authored the evaluator with a plain `fw_` API key, while a user-role identity received **`403: admin role required to call CreateEvaluatorV2`**. Treat live docs and current API behavior as authoritative because scoped permissions may change. Do not broaden credentials or work around a 403. Ask an admin to register the evaluator once, then launch managed RFT with its resource ID.

> **But the run is capacity-gated.** RFT provisions a trainer + a rollout deployment; on capacity-constrained accounts this readiness-times-out or fails (0/6 success observed on the shared account). RFT reliability is a **platform** matter, not a skill one — see `references/deploy-and-troubleshoot.md`.

## Ways to supply the reward

Whichever RFT route you use, the reward can come from three places:

1. **An existing evaluator ID** (`accounts/<acct>/evaluators/<id>`) — reuse a registered eval3 evaluator. Managed route.
2. **An inline `reward_fn`** — write the reward as Python in the forked recipe. The agent-drivable default; no evaluator resource, any key.
3. **An LLM-judge rubric** — when correctness isn't a clean 0/1 (tone, quality, open-ended), score completions with a judge model against a rubric. Implement it inside `reward_fn` (call a judge model, map its verdict to a float), or register it as an evaluator. If the user gives no evaluator and no rubric, infer a data-grounded rubric from the training samples and confirm it with them before spending.

## Default workflow: fork a cookbook recipe

> **Cookbook implementation:** use the exact recipe and SDK references routed from the root `SKILL.md`. This file explains managed-versus-Training-API choices; `references/sdk/` and `training/` contain the implementation depth.

**Approval gate still applies.** Training API cookbook recipes can create trainers, rollout deployments, checkpoints, promoted models, and paid inference. Before running a recipe or any create/promote/deploy action, show the account, recipe, dataset, full resolved trainer and deployment config, cost drivers, evaluation plan, and teardown plan; get explicit approval for this run. A deployment or materially expanded sweep gets its own approval.

Don't write a loop from scratch — fork a recipe in the `training/` tree of `fw-ai/cookbook`:
- `training/recipes/` — loop scripts (e.g. `async_rl_loop`)
- `training/examples/` — worked RL / SFT / DPO / ORPO
- `training/utils/` — config, data loading, losses, metrics

Recipes cover SFT, DPO/ORPO, and RL (GRPO, DAPO, GSPO, CISPO).

## Core SDK primitives

- `forward_backward` — built-in losses by id (e.g. `"cross_entropy"`), no extra forward pass.
- `forward_backward_custom(datums, loss_fn)` — your Python loss; returns per-token logprobs with gradients. **Loss runs locally; forward/backward run on remote GPUs.**
- `forward` — forward-only (e.g. reference-model logprobs).
- `optim_step(...)` — optimizer update after gradient accumulation.
- `save_weights_for_sampler()` + `create_sampling_client()` — export a checkpoint + stand up a sampler (weight sync for eval/rollouts).

Loss docs: https://docs.fireworks.ai/fine-tuning/training-api/loss-functions

```python
def loss_fn(data, logprobs_list):   # logprobs_list: per-token, requires_grad
    # return (scalar differentiable loss, {metrics for logging})
    ...
```

## RL: async loop + rollouts

RL recipe: https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl

`training.recipes.rl_loop` (sync) and `async_rl_loop` (rollout/train overlap; superset, preferred for new work) are the fork-and-customise recipes. The one thing you customise is the **inline reward**, exactly like Tinker:
```python
def reward_fn(completion: str, row: dict) -> float:   # fork this
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    return 1.0 if predicted == truth else 0.0
```
The dataset is JSONL of prompts (`messages`) plus exactly the fields the selected reward reads. `ground_truth` is common for exact-match rewards, but it is not universal. Declare and validate the reward's required fields before launch. The recipe samples rollouts, scores each with `reward_fn`, and trains. Key knobs: `policy_loss` (`grpo`/`dapo`/`gspo`/`cispo`/…), `max_head_offpolicy_versions` (0 = strict on-policy), `completions_per_prompt`. The SDK provisions the trainer + rollout deployment and tears them down on close. **No registered evaluator needed on this path.**

## Numerics alignment & MoE Router Replay (R3)

Trainer↔inference divergence silently wrecks RL. Required reading: https://docs.fireworks.ai/fine-tuning/rl-rollout-integration#numerics-alignment
- Match **precision/quantization** between trainer checkpoints and the deployment shape.
- Measure **logprob divergence** between trainer forward and rollout inference on the same tokens.
- **MoE → Router Replay (R3):** align the top-K experts the router picks. Inference returns them via `include_routing_matrix: true` + `logprobs: true`; feed them back through `loss_fn_inputs`. https://docs.fireworks.ai/guides/rollout-inference#moe-router-replay

## Critical rules

- **Start from a cookbook recipe**, not a blank loop.
- **Align numerics** before trusting any RL signal; turn on **R3** for MoE.
- **For RL, use a dedicated rollout deployment** (sized for sampling), not your prod endpoint.
- **Set an explicit deployment shape for the rollout.** RFT provisions a rollout/sampling deployment, and the accelerator is owned by the *deployment shape*. With only a base model and `training_shape=auto`, creation fails: `ValueError: Cannot create a managed deployment without a deployment shape`. Pass a `deployment_shape` (or a `training_shape_id` whose shape references one) — e.g. via the recipe's `--training-shape`. The RFT preflight should check this before launch. (Observed live on qwen3-4b, 2026-07-15.)
- Use `forward_backward_custom` only for a genuinely custom objective; otherwise the built-in path is cheaper.
- Loss must be differentiable w.r.t. `logprobs_list`; the metrics dict is logging-only.
