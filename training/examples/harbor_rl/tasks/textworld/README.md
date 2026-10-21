# TextWorld RL

This recipe freezes seeded TextWorld cooking games as Harbor tasks, then
trains the pinned Pi harness in disposable E2B sandboxes. The model uses the
dedicated `textworld_action` and `textworld_reset` tools and receives reward
`1` only when a replayed action history reaches TextWorld's `won` state.
General Bash, read, edit, and write tools are disabled for this recipe.

The calibrated default suite uses five ingredients, twelve rooms, held-out
test foods/preparations, containers, cooking, cutting, and limited inventory.
It is designed to produce both successes and failures before training, leaving
room for a group-relative learning signal.

## Install

From `training/`:

```bash
uv sync
uv pip install --python .venv/bin/python \
  'harbor[e2b]==0.21.0' 'e2b==2.55.0'
```

Generate and freeze a suite:

```bash
uv run python -m training.examples.harbor_rl.tasks.textworld.dataset \
  --output "./data/textworld-cooking-24k" \
  --seed 42 \
  --train-tasks 256 \
  --evaluation-tasks 32 \
  --recipe 5 \
  --take 5 \
  --go 12 \
  --open \
  --cook \
  --cut \
  --drop \
  --split test \
  --generator-venv "./.venv-textworld-generator"
```

The output path must not already exist. The generator automatically creates
the specified pinned TextWorld 1.6.2 / NumPy 1.26.4 environment; generation
stays isolated from the newer dependencies used by TITO. Pass `--tw-make` only
to use an already-installed compatible executable instead. The generated
manifest records every game seed and hashes every Harbor task. Training rejects
added, removed, or modified tasks.

## Check for headroom first

Score the 32 evaluation games against an existing deployment before training.
A mean reward near `1.0` means the split is already solved and training will
show nothing; near `0.0` means the games are too hard to produce any learning
signal. Aim for a rate somewhere in between.

```bash
export FIREWORKS_API_KEY=...
export E2B_API_KEY=...

uv run python -m training.examples.harbor_rl.tasks.textworld.train \
  --sampling-only \
  --deployment-id accounts/example/deployments/abc123 \
  --base-model accounts/fireworks/models/qwen3p8-27b \
  --textworld-dataset "./data/textworld-cooking-24k" \
  --run-dir "./runs/textworld-cooking-eval" \
  --tokenizer-model Qwen/Qwen3.8-27B \
  --renderer-name qwen3_8
```

The run writes `sampling-result.json` with the mean reward
(`sampling/reward`) and the per-game spread (`sampling/task_reward_min`,
`_mean`, `_max`). No trainer or deployment is created.

## Train

```bash
export FIREWORKS_API_KEY=...
export E2B_API_KEY=...
export WANDB_API_KEY=...  # only when --wandb-entity is set

uv run python -m training.examples.harbor_rl.tasks.textworld.train \
  --textworld-dataset "./data/textworld-cooking-24k" \
  --run-dir "./runs/textworld-cooking-pi" \
  --shuffle-seed 7 \
  --base-model accounts/fireworks/models/qwen3p8-27b \
  --tokenizer-model Qwen/Qwen3.8-27B \
  --renderer-name qwen3_8
```

The launcher copies each frozen task, bakes in the pinned Pi harness, prebuilds
content-addressed E2B templates, records the resolved `Config` in
`launch.json`, and then starts server-side GRPO. Pass `--trainer-job-id` and
`--deployment-id` to reattach existing resources and `--skip-cleanup` to keep
them after the run. Training and
evaluation games are disjoint. Each rollout gets a fresh sandbox while E2B
templates are reused. The recipe defaults to the calibrated 24,576-token
sequence limit; reducing `--max-seq-len` can remove the task's learning signal,
so recalibrate the baseline after changing it.

The recipe trains with the trainer's built-in GRPO loss. To compare another
objective, such as the sequence-ratio GSPO surrogate, fork
`recipes/async_rl_loop.py`, import `main` from the fork in `train.py`, and swap
the loss in `train_group` and `_run_optimizer_step`. Each objective in
`training.utils.rl.algorithm` exports its loss name, config and metrics:

```python
from training.utils.rl.algorithm.gspo import (
    GSPO_LOSS_FN,
    GSPO_NORMALIZATION,
    GSPOConfig,
    gspo_loss_fn_config,
    gspo_metrics,
)

gspo = GSPOConfig(clip_ratio_low=0.001, clip_ratio_high=0.001)

# train_group
result = policy.forward_backward(
    datums, GSPO_LOSS_FN, loss_fn_config=gspo_loss_fn_config(gspo)
)
result.metrics.update(gspo_metrics(batch, result, gspo))

# _run_optimizer_step
policy.optim_step(
    adam_params,
    grad_accumulation_normalization=GSPO_NORMALIZATION,
    emit_grad_norm_metrics=True,
)
```

Launch the comparison run from the same base model and shuffled dataset order,
and do not switch losses inside an existing run.

## Async off-policy sweeps

Omit `--full-sync` and set the maximum rollout lead explicitly to study
staleness:

```bash
uv run python -m training.examples.harbor_rl.tasks.textworld.train \
  ...same arguments as above... \
  --max-head-offpolicy-versions 8 \
  --epochs 2
```

`--max-head-offpolicy-versions` controls how many optimizer versions rollout
production may run ahead of training. `--epochs 2` reuses the same frozen,
shuffled task order for a second pass. Use a fresh run directory for every
setting, and keep the seed and task manifest fixed when comparing values.

The recipe uses AdamW with `beta1=0.9`, `beta2=0.95`, `eps=1e-12`, and weight
decay `0.01`. Gradient clipping is disabled by default (`--grad-clip-norm 0`).
Each optimizer step logs `train/grad_norm_pre_norm`, `train/grad_norm`, and
`train/grad_norm_rms`; when clipping changes the gradient,
`train/grad_norm_post_clip` is logged as well.

Inside a trial, typical commands are:

```text
textworld_action({"action": "look"})
textworld_action({"action": "inventory"})
textworld_action({"action": "go north"})
textworld_reset({})
```

The verifier independently replays `/workspace/.textworld-actions.json`
against the frozen game and emits a binary reward. Editing the action file
cannot manufacture success: the replay still has to reach the game's win
state. Each tool executes the game binary directly without a shell and has a
120-second hard timeout.
