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

From `public-repos/cookbook/training`:

```bash
uv sync
uv pip install --python .venv/bin/python \
  'harbor[e2b]==0.21.0'
```

Generate and freeze a suite:

```bash
uv run python -m training.examples.rl.harbor.recipes.textworld.dataset \
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

uv run python -m training.examples.rl.harbor.recipes.textworld.train \
  --sampling-only \
  --deployment-id accounts/example/deployments/abc123 \
  --base-model accounts/fireworks/models/qwen3p8-27b \
  --textworld-dataset "./data/textworld-cooking-24k" \
  --run-dir "./runs/textworld-cooking-eval" \
  --tokenizer-model Qwen/Qwen3.8-27B \
  --renderer-name qwen3_8
```

The run writes `sampling-result.json` with the mean reward and a per-game
breakdown under `sampling/task/<task-id>`. No trainer or deployment is created.

## Train

```bash
export FIREWORKS_API_KEY=...
export E2B_API_KEY=...
export WANDB_API_KEY=...  # only when --wandb-entity is set

uv run python -m training.examples.rl.harbor.recipes.textworld.train \
  --textworld-dataset "./data/textworld-cooking-24k" \
  --run-dir "./runs/textworld-cooking-pi" \
  --shuffle-seed 7 \
  --base-model accounts/fireworks/models/qwen3p8-27b \
  --tokenizer-model Qwen/Qwen3.8-27B \
  --renderer-name qwen3_8
```

The launcher copies each frozen task, bakes in the pinned Pi harness, prebuilds
content-addressed E2B templates, and then starts server-side GRPO. Training and
evaluation games are disjoint. Each rollout gets a fresh sandbox while E2B
templates are reused. The recipe defaults to the calibrated 24,576-token
sequence limit; reducing `--max-seq-len` can remove the task's learning signal,
so recalibrate the baseline after changing it.

To compare the sequence-ratio GSPO surrogate against token-level PPO, launch a
separate run from the same base model and shuffled dataset order with:

```bash
uv run python -m training.examples.rl.harbor.recipes.textworld.train \
  ...same arguments as above... \
  --policy-loss gspo \
  --gspo-clip-ratio-low 0.001 \
  --gspo-clip-ratio-high 0.001
```

GSPO uses the portable two-pass custom loss and forces optimizer normalization
by `num_sequences`. It applies the paper's response mean and
defaults to clipping at `1−3e-4` and `1+4e-4`; the example widens the range to
symmetrical `±0.1%`. Do not switch losses inside an existing run.

Additional client-side policy-loss experiments use the same rollout and
full-sync setup:

```bash
# DAPO: asymmetric PPO clipping [0.8, 1.28].
... --policy-loss dapo

# DRO: smooth quadratic trust region.
... --policy-loss dro --dro-beta 0.05

# CISPO: detached clipped-ratio weighting.
... --policy-loss cispo

# Divergence PPO: binary total-variation trust-region mask.
... --policy-loss dppo --dppo-divergence binary_tv --dppo-threshold 0.15

# Score centering: additive top-5 training-inference drift correction.
... --policy-loss score_centering --score-centering-top-k 5
```

DAPO, DRO, CISPO, and DPPO return raw token-sum losses and do not request
gradient-accumulation normalization. DPPO here means Divergence Proximal
Policy Optimization from arXiv:2602.04879; the binary-KL variant is available
with `--dppo-divergence binary_kl --dppo-threshold 0.05`.

Score centering implements the top-k tail reconstruction from
arXiv:2609.20807. TITO records the sampler's raw top-k distribution for every
policy-generated token, then the trainer evaluates those same token IDs via
multi-target custom loss. Its raw token loss is normalized by
`num_loss_tokens`, matching the paper's shared objective. Top-k collection
uses the public inference limit of five candidates and requires integer
`token_id` values in `top_logprobs`. The paper evaluated larger heads, so this
more aggressive approximation should be validated for the target model. This
recipe uses the shared AdamW optimizer rather than the paper's SGD setup; it
isolates the score-centering correction while preserving the TextWorld
comparison contract.

All policy losses use AdamW with `beta1=0.9`, `beta2=0.95`, `eps=1e-12`, and weight
decay `0.01`. Gradient clipping is disabled by default (`--grad-clip-norm 0`).
The recipe requests basic trainer-side gradient telemetry by default and logs
`train/grad_norm_pre_norm`, `train/grad_norm`, and `train/grad_norm_rms`;
`--grad-norm-metrics detailed` also logs parameter-category norms. When
clipping changes the gradient, `train/grad_norm_post_clip` is logged as well.

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
