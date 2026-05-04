# Out-of-the-box examples

When the user wants "something that just runs", point them at `training/examples/`. Most examples import from `training/recipes/` and ship a ready-to-run `Config`; the larger agentic examples may own a custom loop plus rollout code.

| Task | Path |
|------|------|
| SFT on real datasets | `training/examples/sft/` |
| DPO | `training/examples/dpo/` |
| ORPO | `training/examples/orpo/` |
| GRPO on DeepMath | `training/examples/rl/deepmath/` |
| Tool-use RL (Frozen Lake) | `training/examples/rl/frozen_lake/` |
| Async single-turn RL (renderer-backed) | `training/examples/rl/single_turn_async/` |
| Async single-turn RL on GSM8K | `training/examples/rl/gsm8k_async/` |
| Multi-turn RL (token-native, no renderer) | `training/examples/rl/multi_turn_minimal/` |
| Multi-turn renderer RL | `training/examples/rl/multi_turn_minimal_renderer/` |
| Multi-turn tool RL | `training/examples/rl/multi_turn_tool/` |
| Remote rollout service RL | `training/examples/rl/remote_rollout/` |
| Eval-protocol remote grader RL | `training/examples/rl/ep_remote_grader/` |
| Multi-hop QA IGPO | `training/examples/multihop_qa/` |

## What to read

In each example directory, read the top-level `train_*.py`, `train.py`, or `rollout.py` — it builds the `Config` or rollout function and calls the corresponding recipe/helper.

## Running

```bash
cd cookbook/training
pip install --pre -e .
python examples/sft/train_sft.py --output-model-id <model-id>
```

Every example pulls `FIREWORKS_API_KEY` from the env (or `.env`). If an example needs a dataset path or extra arguments, check that script's `argparse` block, `README.md`, or `run.sh`.

## When the example isn't quite right

Fork the recipe instead — see [`recipes.md`](recipes.md). The example is just a thin wrapper around the recipe's `Config`.
