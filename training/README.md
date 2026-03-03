# Fireworks Training Cookbook

Ready-to-run training recipes for reinforcement learning, preference optimization, and supervised fine-tuning on [Fireworks](https://fireworks.ai).
Each recipe is a single Python file you can fork and customize.

## Install

```bash
pip install -r requirements.txt

export FIREWORKS_API_KEY="..."
export FIREWORKS_ACCOUNT_ID="..."
```

## Quick start (GRPO)

```python
from fireworks.training.cookbook.recipes.rl_loop import Config, main
from fireworks.training.cookbook.utils import InfraConfig, DeployConfig, HotloadConfig

cfg = Config(
    base_model="accounts/fireworks/models/qwen3-8b",
    dataset="https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl",
    completions_per_prompt=4,
    policy_loss="grpo",
    deployment=DeployConfig(
        tokenizer_model="Qwen/Qwen3-8B",
    ),
    hotload=HotloadConfig(hot_load_interval=1),
)

main(cfg)
```

When no `deployment_id` is provided, a new inference deployment is created automatically.
To reuse an existing deployment, pass `deployment_id="my-deploy"`.

### Server-side tokenization (no local tokenizer)

```python
from fireworks.training.cookbook.recipes.rl_loop_multiturn import Config, main

cfg = Config(
    base_model="accounts/fireworks/models/qwen3-8b",
    dataset="path/to/multiturn.jsonl",
)

main(cfg)
```

This variant uses `/v1/chat/completions` with `return_token_ids=True` --
the server handles tokenization, so no `tokenizer_model` is needed for sampling.

## Recipes

| Recipe | File | Description |
| --- | --- | --- |
| GRPO / DAPO / GSPO / CISPO | `recipes/rl_loop.py` | On-policy RL with streaming rollouts. Set `policy_loss="grpo"`, `"dapo"`, `"gspo"`, or `"cispo"`. |
| GRPO (server-side tokenization) | `recipes/rl_loop_multiturn.py` | Same RL loop using chat completions API. No local tokenizer required. |
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
  Set `use_chat_completions=True` for server-side tokenization.
- **`InfraConfig`** -- region, accelerators, training shapes.
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
pip install -r requirements-dev.txt
pytest tests/
```
