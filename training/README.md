# Fireworks Training Cookbook

Ready-to-run training recipes for reinforcement learning, preference optimization, and supervised fine-tuning on [Fireworks](https://fireworks.ai).
Each recipe is a single Python file you can fork and customize.

## Install

```bash
cd cookbook
pip install -e .

export FIREWORKS_API_KEY="..."
export FIREWORKS_ACCOUNT_ID="..."
```

## Quick start (GRPO)

```python
from training.recipes.rl_loop import Config, main
from training.utils import InfraConfig, DeployConfig, HotloadConfig, WandBConfig

cfg = Config(
    base_model="accounts/fireworks/models/qwen3-8b",
    dataset="path/to/your_data.jsonl",
    completions_per_prompt=4,
    policy_loss="grpo",
    deployment=DeployConfig(
        tokenizer_model="Qwen/Qwen3-8B",
        # deployment_id="my-existing-deploy",    # omit to auto-create
        # deployment_region="US_OHIO_1",
        # sample_timeout=600,
    ),
    infra=InfraConfig(
        training_shape_id="your-training-shape",
        # ref_training_shape_id="your-ref-shape",  # if different from policy
        # region="US_OHIO_1",
        # skip_validations=False,
    ),
    hotload=HotloadConfig(hot_load_interval=1),
    # learning_rate=1e-5,
    # kl_beta=0.001,                              # set 0 to skip reference model
    # temperature=1.0,
    # max_completion_tokens=1024,
    # epochs=1,
    # lora_rank=0,
    # wandb=WandBConfig(project="my-project"),
)

main(cfg)
```

When no `deployment_id` is provided, a new inference deployment is created automatically.
To reuse an existing deployment, pass `deployment_id="my-deploy"`.

## Recipes

| Recipe | File | Description |
| --- | --- | --- |
| GRPO / DAPO / GSPO / CISPO | `recipes/rl_loop.py` | On-policy RL with streaming rollouts. Set `policy_loss="grpo"`, `"dapo"`, `"gspo"`, or `"cispo"`. |
| DPO | `recipes/dpo_loop.py` | Direct preference optimization with cached reference logprobs. |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization -- no reference model needed. |
| SFT | `recipes/sft_loop.py` | Supervised fine-tuning with response-only cross-entropy loss. |

## Directory layout

```
recipes/            Training loop scripts (fork these)
utils/              Shared config, data loading, loss functions, metrics
examples/deepmath/  Worked example: math reasoning with GRPO
tests/              Unit and end-to-end tests
```

## Configuration

All recipes use composable dataclass configs:

- **`DeployConfig`** -- inference deployment. Set `deployment_id` to use an existing deployment, or leave it unset to auto-create one.
- **`InfraConfig`** -- region, accelerators, training shapes. When `training_shape_id` is set, config is auto-derived from the shape.
- **`HotloadConfig`** -- weight sync cadence and checkpoint settings.
- **`WandBConfig`** -- optional Weights & Biases logging.
- **`ResumeConfig`** -- resume training from a checkpoint.

## What to customize

1. **Reward function** -- `reward_fn` in `recipes/rl_loop.py`. Replace with your task-specific reward logic.
2. **Dataset format** -- each row needs a `messages` list (OpenAI chat format) and any fields your reward function reads.
3. **Loss tuning** -- adjust `kl_beta`, `temperature`, `completions_per_prompt`, or switch `policy_loss`.
4. **Infrastructure** -- set `InfraConfig(training_shape_id="...")` to use pre-configured training shapes.

## Tests

```bash
pip install -e ".[dev]"
pytest training/tests/
```
