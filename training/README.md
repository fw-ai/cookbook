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

**Quick start** — run the SFT example end-to-end:

```bash
export FIREWORKS_API_KEY="your-api-key"
python examples/sft/train_sft.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --tokenizer-model Qwen/Qwen3-8B \
    --dataset-path examples/sft/text2sql_dataset.jsonl \
    --region US_VIRGINIA_1 \
    --max-examples 100 \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --output-model-id sft-text-qwen3-8b-$(date +%Y%m%d%H%M)
```

See [`skills/dev/references/examples.md`](../skills/dev/references/examples.md) for all worked examples.

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
