# Training API (custom training loops)

*Source of truth: [Training API intro](https://docs.fireworks.ai/fine-tuning/training-api/introduction.md) · [cookbook](https://github.com/fw-ai/cookbook) — defer to the live docs/repo for current SDK + recipes.*

Write your **own** Python training loop; Fireworks runs forward/backward on distributed GPUs. The SDK is **Tinker-compatible**, so Tinker code ports over with minimal changes. Low-level path — for standard runs prefer the managed UI.

Docs: https://docs.fireworks.ai/fine-tuning/training-api/introduction · Cookbook (start here): https://github.com/fw-ai/cookbook

> Status: private preview — request access at https://fireworks.ai/contact-training

## API vs managed UI

Use the **managed UI** for standard SFT/DPO (data + hyperparameters, no code). Reach for the **Training API** when you need: custom **loss/reward**, **RL with rollouts** (inference-in-the-loop), access to forward-pass internals (e.g. MoE routing for R3), or multi-turn/agentic trajectories.

## Two ways to run RFT/RL — pick the inline-reward recipe for agent-driven work

There are two RFT paths, and they differ in **where the reward lives**. This matters a lot when a coding agent is driving:

| Path | Reward | Agent-drivable? |
|---|---|---|
| **Managed RFT** — `firectl reinforcement-fine-tuning-job create --evaluator <id>` | A **registered evaluator resource** (server-side, built in an e2b sandbox, eval v3) | **Only with an admin-role key.** Register via **eval-protocol** (`pytest` auto-registers) or the **UI**; authoring calls `CreateEvaluatorV2`, which **requires admin role** (a user-role key gets 403). `firectl evaluator create` (V1) is **deprecated**. Use when an admin authors the evaluator. |
| **Training-API RL** — fork `training.recipes.rl_loop` / `async_rl_loop` | An **inline `reward_fn(completion, row) -> float`** in the forked recipe (`row["ground_truth"]` carries the label) | **Yes.** No evaluator resource. Same shape as Tinker's reward-in-the-loop. The SDK provisions the trainer + rollout deployment. |

**For an agent running RFT via this skill, use the inline-reward recipe path** (below). It needs no deprecated evaluator, mirrors Tinker, and is fully code-driven. Reserve the managed `--evaluator` path for when a human sets up the sandboxed evaluator via eval-protocol/UI.

### Managed RFT: authoring the eval3 evaluator (admin-role-gated)

`firectl reinforcement-fine-tuning-job create` needs an **eval3 evaluator with an `entry_point`** — legacy evaluators are rejected (`InvalidArgument: managed RFT requires an eval3 evaluator`), and `firectl evaluator create` is deprecated. The code-first way to make one is **eval-protocol** (no UI), but **authoring requires an admin-role identity** — see the gate below:

```bash
pip install eval-protocol          # provides @reward / EvaluationRow (needs Python 3.10+)
```
```python
from eval_protocol import reward, EvaluationRow          # write the reward as code
@reward
def category_match(row: EvaluationRow) -> float:
    return 1.0 if parse(row.output).get("category") == row.ground_truth else 0.0
```
Running the eval-protocol `pytest` flow **auto-registers** the reward as an eval3 evaluator + the dataset with Fireworks; then `firectl rftj create --base-model <m> --dataset <ds> --evaluator accounts/<acct>/evaluators/<id>`. So the "you need the UI" limitation is not fundamental: the flow is fully code-drivable via eval-protocol. But **authoring still requires an admin role** (below), so a standard user-role key cannot do it unattended.

**It's an admin-role gate, not a credential-type gate (tested).** eval3 creation is a multi-RPC flow (`CreateEvaluatorV2` → `GetEvaluatorUploadEndpoint` → upload → `ValidateEvaluatorUpload` → e2b build). eval-protocol drives exactly this. The gate is on **account role, not credential type**: an **admin-role** identity authors the evaluator fine (even with a plain `fw_` API key), while a **user-role** identity gets **`403: admin role required to call CreateEvaluatorV2`**. The UI wizard works only because a console session carries admin. So an agent whose key has admin role **can** author eval3 in code; an agent with a user-role key **cannot**, and needs an admin (UI or admin-role key) to author it once. This is a platform authz gate, not something the skill can route around. (`firectl evaluator create` is separately deprecated/V1.) → For managed RFT with a non-admin key, an admin authors the evaluator once; the agent then drives `rftj create --evaluator <id>`. This admin gate is exactly why agent-driven RFT defaults to the inline-reward path.

> **But the run is capacity-gated.** RFT provisions a trainer + a rollout deployment; on capacity-constrained accounts this readiness-times-out or fails (0/6 success observed on the shared account). RFT reliability is a **platform** matter, not a skill one — see `references/deploy-and-troubleshoot.md`.

## Default workflow: fork a cookbook recipe

> **In this repo:** the companion `fireworks-training` skill ([`../../dev/SKILL.md`](../../dev/SKILL.md)) is the SDK reference implementation — route there for recipe internals (RL loss paths, hotload, shapes, distillation, infra migration). This reference is the managed-orchestration view; `dev` is the code.

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
The dataset is JSONL of prompts (`messages`) with a `ground_truth` field per row; the recipe samples rollouts, scores each with `reward_fn`, and trains. Key knobs: `policy_loss` (`grpo`/`dapo`/`gspo`/`cispo`/…), `max_head_offpolicy_versions` (0 = strict on-policy), `completions_per_prompt`. The SDK provisions the trainer + rollout deployment and tears them down on close. **No registered evaluator needed on this path.**

## Numerics alignment & MoE Router Replay (R3)

Trainer↔inference divergence silently wrecks RL. Required reading: https://docs.fireworks.ai/fine-tuning/rl-rollout-integration#numerics-alignment
- Match **precision/quantization** between trainer checkpoints and the deployment shape.
- Measure **logprob divergence** between trainer forward and rollout inference on the same tokens.
- **MoE → Router Replay (R3):** align the top-K experts the router picks. Inference returns them via `include_routing_matrix: true` + `logprobs: true`; feed them back through `loss_fn_inputs`. https://docs.fireworks.ai/guides/rollout-inference#moe-router-replay

## Critical rules

- **Start from a cookbook recipe**, not a blank loop.
- **Align numerics** before trusting any RL signal; turn on **R3** for MoE.
- **For RL, use a dedicated rollout deployment** (sized for sampling), not your prod endpoint.
- Use `forward_backward_custom` only for a genuinely custom objective; otherwise the built-in path is cheaper.
- Loss must be differentiable w.r.t. `logprobs_list`; the metrics dict is logging-only.
