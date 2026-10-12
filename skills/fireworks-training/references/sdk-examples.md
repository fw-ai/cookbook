# Out-of-the-box examples

When the user wants "something that just runs", point them at `training/examples/`. Most subdirectories import from `training/recipes/` and ship a ready-to-run `Config`; the agentic examples may own a custom loop plus rollout code.

Each task folder pairs a dedicated entry point (`train_*.py`) with a
serverless twin (`train_*_serverless.py`) where both exist.

| Task | Path |
|------|------|
| SFT (text, vision via `run_vl.sh`; dedicated + serverless) | `training/examples/sft/` |
| DPO (dedicated + serverless) | `training/examples/dpo/` |
| ORPO (IFEval) | `training/examples/orpo/ifeval/` |
| Text single-turn GRPO (DeepMath; dedicated + serverless) | `training/examples/rl/deepmath/` |
| Vision multi-turn tool GRPO (VisualToolBench; dedicated + serverless) | `training/examples/rl/visual_toolbench/` |
| Tool-use RL environment (Frozen Lake, text or image) | `training/examples/rl/frozen_lake/` |
| Multi-hop QA with IGPO turn-level rewards | `training/examples/rl/multihop_qa/` |
| Agentic RL in sandboxes (Harbor + TITO; OpenCode, Pi, Mini-SWE-Agent) | `training/examples/harbor_rl/` |
| Distillation (dedicated + serverless) | `training/examples/distillation/` |

`training/examples/archive/` is deprecated; do not start new work from it.

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

Fork the recipe instead — see [`sdk-recipes.md`](sdk-recipes.md). The example is just a thin wrapper around the recipe's `Config`.
