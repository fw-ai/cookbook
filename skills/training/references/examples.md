# Out-of-the-box examples

When the user wants "something that just runs", point them at `training/examples/`. Each subdirectory imports from `training/recipes/` and ships a ready-to-run `Config`.

| Task | Path |
|------|------|
| Minimal SFT (single-file demo) | `training/examples/sft_getting_started/` |
| SFT on real datasets | `training/examples/sft/` |
| DPO | `training/examples/dpo/` |
| ORPO | `training/examples/orpo/` |
| GRPO on deepmath | `training/examples/deepmath_rl/` |
| Tool-use RL (frozen lake) | `training/examples/frozen_lake/` |
| Multi-hop QA RL | `training/examples/multihop_qa/` |
| Generic RL wiring | `training/examples/rl/` |

## What to read

In each example directory, read the top-level `train_*.py` (or `main.py`) — it builds the `Config` and calls `main(config)` from the corresponding recipe. That's the whole script.

## Running

```bash
cd cookbook/training
pip install -e .
python examples/sft_getting_started/<script>.py
```

Every example pulls `FIREWORKS_API_KEY` from the env (or `.env`). If the example needs a dataset path, it is listed in its `README.md`.

## When the example isn't quite right

Fork the recipe instead — see [`recipes.md`](recipes.md). The example is just a thin wrapper around the recipe's `Config`.
