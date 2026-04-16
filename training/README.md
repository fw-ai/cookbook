# Fireworks Training Cookbook

Ready-to-run training recipes for reinforcement learning, preference optimization, and supervised fine-tuning on [Fireworks](https://fireworks.ai).

> **Full documentation**: [Training SDK docs](https://docs.fireworks.ai/fine-tuning/training-sdk/introduction)

## Setup

See [`skills/dev/references/setup.md`](../skills/dev/references/setup.md) for install, credentials, and verification.

Quick version:

```bash
cd cookbook/training
conda create -n cookbook python=3.12 -y && conda activate cookbook
pip install --pre -e .
```

## Recipes

| Recipe | File | Description |
| --- | --- | --- |
| GRPO / IS / DAPO / DRO / GSPO / CISPO | `recipes/rl_loop.py` | On-policy RL with streaming rollouts |
| DPO | `recipes/dpo_loop.py` | Direct preference optimization |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization |
| SFT | `recipes/sft_loop.py` | Supervised fine-tuning |

Each recipe has a `Config` dataclass — edit it and run `python -m recipes.<name>`.

## Directory Layout

```
recipes/              Training loop scripts (fork these)
utils/                Shared config, data loading, losses, metrics
examples/
  rl/deepmath/        Worked example: math reasoning with GRPO
  rl/frozen_lake/     Worked example: Frozen Lake with tool-use RL
  orpo/ifeval/        Worked example: IFEval with ORPO
  sft/                Worked example: SFT getting started
  dpo/                Worked example: DPO
  tools/              Standalone utility scripts
tests/                Unit and end-to-end tests
```

## Skills (primary reference)

For configuration, debugging, shapes, hotload, checkpoints, and all task-specific guidance, see **[`skills/dev/SKILL.md`](../skills/dev/SKILL.md)** — it is the single source of truth and maps every task to its reference doc.
