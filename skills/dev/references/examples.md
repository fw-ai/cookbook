# Out-of-the-box examples

When the user wants "something that just runs", point them at `training/examples/`. Most subdirectories import from `training/recipes/` and ship a ready-to-run `Config`; the agentic examples may own a custom loop plus rollout code.

| Task | Path |
|------|------|
| SFT | `training/examples/sft/` |
| DPO | `training/examples/dpo/` |
| ORPO (IFEval) | `training/examples/orpo/ifeval/` |
| GRPO on DeepMath | `training/examples/rl/deepmath/` |
| Tool-use RL (Frozen Lake) | `training/examples/rl/frozen_lake/` |
| Async RL single-turn (token-in rollout) | `training/examples/rl/single_turn_token_in/` |
| Async RL multi-turn (message-in rollout) | `training/examples/rl/multi_turn_message_in/` |
| Multi-hop QA async RL (+ optional IGPO) | `training/examples/multihop_qa/` |
| Manual hotload-scope tests (PER_TRAINER re-attach, PER_DEPLOYMENT) | `training/examples/manual/` |

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
