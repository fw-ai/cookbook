---
name: training
description: Use the Fireworks Training Cookbook -- find recipes, run examples, understand utilities, and customize training loops. Use this skill when the user wants to train a model (SFT, RL, DPO, ORPO), run a worked example, customize a reward function, understand cookbook utilities, promote a checkpoint, reconnect to a running trainer, verify logprobs, or debug a training script.
---

# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/stainless-sdks/fireworks-ai-python) (`fireworks.training.sdk`). Each recipe is a single Python file you can fork and customize.

**Only `training/` matters.** Other top-level directories in this repo are unrelated to the training SDK -- ignore them.

---

## Directory layout

```
training/
  recipes/            Training loop scripts (fork these)
  utils/              Shared config, infra, losses, data loading, logging
  utils/rl/           RL-specific losses, training loop, metrics, TIS, R3
  examples/           Worked examples with datasets and bash runners
  examples/snippets/  Standalone tool scripts (not training loops)
  renderer/           Token-level renderers for supervised/multi-modal data
  tests/              Unit, smoke, and end-to-end tests
```

---

## Recipes

Fork and customize. Each has a `Config` dataclass at the top and `main()` entry point.

| Recipe | File | Description |
|--------|------|-------------|
| RL (GRPO/DAPO/DRO/GSPO/CISPO) | `recipes/rl_loop.py` | On-policy RL with streaming rollouts, concurrent sampling, weight sync. Set `policy_loss=` to choose algorithm. |
| IGPO | `recipes/igpo_loop.py` | RL with turn-level intrinsic rewards for multi-turn agentic trajectories |
| DPO | `recipes/dpo_loop.py` | Direct preference optimization with cached reference logprobs |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization (no reference model) |
| SFT | `recipes/sft_loop.py` | Supervised fine-tuning with response-only cross-entropy loss |

### Minimum config for any recipe

| Field | What to set |
|-------|-------------|
| `dataset` | Path to JSONL training data |
| `base_model` | Fireworks model ID (e.g. `accounts/fireworks/models/qwen3-8b`) |
| `max_seq_len` | Max token length |
| `infra` | `InfraConfig()` for auto shape selection, or `InfraConfig(training_shape_id="...")` to override |

RL recipes also need `deployment` (`DeployConfig`) and `weight_sync` (`WeightSyncConfig`). SFT/DPO/ORPO also need `tokenizer_model`.

---

## Worked examples

Full training scripts with datasets, reward functions, and bash runners. Start from the closest example and modify.

| Example | Directory | What it shows |
|---------|-----------|---------------|
| DeepMath GRPO | `examples/rl/deepmath/` | Math reasoning RL with verification reward |
| Frozen Lake | `examples/rl/frozen_lake/` | Tool-use RL in a grid-world environment (multi-turn, action masking) |
| Multi-hop QA (IGPO) | `examples/multihop_qa/` | Multi-turn agentic QA with information-gain rewards |
| DPO | `examples/dpo/` | Preference optimization entry point |
| ORPO (IFEval) | `examples/orpo/ifeval/` | Instruction-following with ORPO |
| SFT | `examples/sft/` | Getting started with supervised fine-tuning (text2sql dataset) |
| Hotload re-attach | `examples/hotload_reattach/` | Test for switching deployments between trainer jobs |

---

## Snippets (standalone tool scripts)

These are **not** training loops. They are operational utilities for specific tasks.

| Script | What it does |
|--------|-------------|
| `examples/snippets/promote_checkpoint.py` | Read `checkpoints.jsonl`, find a sampler checkpoint, promote it to a deployable Fireworks model |
| `examples/snippets/reconnect_and_adjust_lr.py` | Reconnect to an already-running trainer job and adjust learning rate mid-run |
| `examples/snippets/verify_logprobs.py` | Create trainer + deployment, sample completions, verify per-token logprob alignment between training and inference |

---

## Utils reference

### Core

| Module | What it provides |
|--------|-----------------|
| `utils/config.py` | `InfraConfig`, `DeployConfig`, `WeightSyncConfig`, `WandBConfig`, `RunnerConfig`, `ConcurrencyConfig` |
| `utils/infra.py` | `create_trainer_job`, `setup_deployment`, `setup_or_reattach_deployment` -- orchestrates SDK managers |
| `utils/client.py` | `ReconnectableClient` -- wraps `FiretitanTrainingClient` with dispatch/wait logic and reconnection |
| `utils/training_shapes.py` | Training shape auto-selection and profile resolution from control-plane data |
| `utils/validation.py` | Pre-flight config validation (credentials, base_model, dataset) |

### Data & losses

| Module | What it provides |
|--------|-----------------|
| `utils/data.py` | `RLPromptDataset` -- batch-indexed RL prompt loading from JSONL |
| `utils/losses.py` | DPO, ORPO, SFT loss functions with microbatch validation |
| `utils/supervised.py` | Token-level rendering helpers for masks and weights |

### RL

| Module | What it provides |
|--------|-----------------|
| `utils/rl/losses.py` | Loss registration, `PromptGroup`, advantage computation |
| `utils/rl/grpo.py` | GRPO loss (PPO-style clipped ratio with TIS) |
| `utils/rl/dapo.py` | DAPO loss (asymmetric clipping + TIS) |
| `utils/rl/gspo.py` | GSPO loss (sequence-level geometric mean) |
| `utils/rl/dro.py` | DRO loss (distributionally robust, quadratic penalty) |
| `utils/rl/cispo.py` | CISPO loss (clipped IS as detached weight) |
| `utils/rl/train.py` | `run_rl_loop` -- pipelined on-policy training with weight sync |
| `utils/rl/tis.py` | Train-Inference Sampling weight correction |
| `utils/rl/router_replay.py` | R3: align MoE routing from inference to training |
| `utils/rl/metrics.py` | RL metric helpers (response lengths, training perf) |

### I/O & logging

| Module | What it provides |
|--------|-----------------|
| `utils/checkpoint_utils.py` | Resume from checkpoint, GCS-transparent I/O, `checkpoints.jsonl` logging |
| `utils/fileio.py` | Transparent local/GCS file I/O (abstracts `gs://` URIs) |
| `utils/logging.py` | WandB setup, metrics logging, offline fallback |
| `utils/timer.py` | Singleton timer for per-step operation timing |
| `utils/runner.py` | Runner contract: status/metadata/metrics files for orchestration |

---

## Install

```bash
cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook
uv pip install -e .
```

The `--pre` flag is required -- without it pip resolves to stable `0.x` which does not include `fireworks.training.sdk`.

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/unit tests/test_smoke_imports.py    # unit tests (no API key needed)
pytest tests/                                     # full suite (needs FIREWORKS_API_KEY)
```
