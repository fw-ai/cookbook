# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/stainless-sdks/fireworks-ai-python) (`fireworks.training.sdk`).

**Only `training/` is relevant.** Other top-level directories in this repo are unrelated to the training SDK -- ignore them.

## Directory layout

```
training/
  recipes/          Training loop scripts (fork and customize these)
  utils/            Shared config, infra, losses, data loading, logging
  examples/         Worked examples with ready-to-run scripts
  examples/snippets/ Standalone utility scripts (not training loops)
  renderer/         Token-level renderers for supervised/multi-modal data
  tests/            Unit, smoke, and end-to-end tests
```

## Recipes

Each recipe is a single Python file with a `Config` dataclass and a `main()` function. Fork and customize.

| Recipe | File | Description |
|--------|------|-------------|
| RL (GRPO/DAPO/DRO/GSPO/CISPO) | `recipes/rl_loop.py` | On-policy RL with streaming rollouts, concurrent sampling, weight sync |
| IGPO | `recipes/igpo_loop.py` | RL with turn-level intrinsic rewards for multi-turn agentic trajectories |
| DPO | `recipes/dpo_loop.py` | Direct preference optimization with cached reference logprobs |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization (no reference model) |
| SFT | `recipes/sft_loop.py` | Supervised fine-tuning with response-only cross-entropy loss |

## Worked examples

Full training scripts with datasets, reward functions, and bash runners. Use these as starting points.

| Example | Directory | What it shows |
|---------|-----------|---------------|
| DeepMath GRPO | `examples/rl/deepmath/` | Math reasoning RL with verification reward |
| Frozen Lake | `examples/rl/frozen_lake/` | Tool-use RL in a grid-world environment |
| Multi-hop QA (IGPO) | `examples/multihop_qa/` | Multi-turn agentic QA with information-gain rewards |
| DPO | `examples/dpo/` | Preference optimization entry point |
| ORPO (IFEval) | `examples/orpo/ifeval/` | Instruction-following with ORPO |
| SFT | `examples/sft/` | Getting started with supervised fine-tuning |
| Hotload re-attach | `examples/hotload_reattach/` | Test for switching deployments between trainer jobs |

## Snippets (standalone tools)

These are **not training loops** -- they are operational utility scripts for specific tasks.

| Script | What it does |
|--------|-------------|
| `examples/snippets/promote_checkpoint.py` | Read `checkpoints.jsonl`, find a sampler checkpoint, promote it to a deployable Fireworks model |
| `examples/snippets/reconnect_and_adjust_lr.py` | Reconnect to an already-running trainer and adjust learning rate mid-run |
| `examples/snippets/verify_logprobs.py` | Create trainer + deployment, sample completions, verify per-token logprob alignment |

## Utils

| Module | What it provides |
|--------|-----------------|
| `utils/config.py` | `InfraConfig`, `DeployConfig`, `WeightSyncConfig`, `WandBConfig`, `RunnerConfig` |
| `utils/infra.py` | `create_trainer_job`, `setup_deployment`, `setup_or_reattach_deployment` |
| `utils/client.py` | `ReconnectableClient` -- wraps training client with dispatch/wait logic |
| `utils/losses.py` | DPO/ORPO/SFT loss functions with microbatch validation |
| `utils/data.py` | `RLPromptDataset` -- batch-indexed prompt loading |
| `utils/rl/` | RL losses (GRPO, DAPO, DRO, GSPO, CISPO, REINFORCE), training loop, metrics, TIS, R3 |
| `utils/checkpoint_utils.py` | Resume from checkpoint, GCS-transparent I/O |
| `utils/logging.py` | WandB setup and metrics logging |
| `utils/training_shapes.py` | Training shape auto-selection and profile resolution |

## Install

```bash
cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook
uv pip install -e .
```

The `--pre` flag is required to get the `fireworks.training.sdk` package.

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/unit tests/test_smoke_imports.py    # unit tests (no API)
pytest tests/                                     # full suite (needs API key)
```
