---
name: training
description: Use the Fireworks Training Cookbook to train models on Fireworks. Covers what fields to pass into recipes, how to reuse trainer jobs and deployments across runs, promoting a checkpoint to a deployable model, re-attaching a deployment to a new trainer, and the shared utilities under training/utils/. Use this skill when the user wants to configure or run a training recipe, iterate quickly across runs by reusing trainers/deployments, promote a checkpoint, re-attach a deployment, or debug a cookbook config.
---

# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/stainless-sdks/fireworks-ai-python) (`fireworks.training.sdk`). Each recipe is a single Python file in `training/recipes/` that you fork and customize.

**Only `training/` is relevant.** Ignore other top-level directories.

This skill focuses on the operational questions: what to pass in, what gets reused, which tool to use for a given task. For *what each training algorithm does conceptually*, see the Fireworks docs, not this skill.

---

## The two operational questions

### "What do I pass in?"

Every recipe has a `Config` dataclass at the top of the file. Always required:

- `dataset` (JSONL path or `gs://` URL)
- `base_model` (Fireworks model ID)
- `max_seq_len`
- `log_path` (where `checkpoints.jsonl`, `metrics.jsonl`, `status.json` go)
- `infra` (`InfraConfig`; auto-selects a validated shape if `training_shape_id` is unset)

Recipe-specific additions (see [`references/recipes.md`](references/recipes.md) for the full table):

- **SFT, DPO, ORPO**: `tokenizer_model` (HF name)
- **RL, IGPO**: `deployment: DeployConfig` (`tokenizer_model` required), `weight_sync: WeightSyncConfig`, `reward_fn`, `policy_loss`

### "Will my trainer / deployment be reused?"

| You pass | Trainer behavior | Deployment behavior |
|----------|-----------------|----------------------|
| Neither `job_id` nor `deployment_id` | New trainer created | New deployment created (auto-generated ID) |
| `job_id` on existing RUNNING job | **Reused** via `reconnect_and_wait` | — |
| `job_id` on FAILED/CANCELLED/PAUSED | Resumed and waited on | — |
| `deployment_id` on READY/UPDATING deployment | — | **Reused** and re-attached to the new trainer (PATCH `hotLoadTrainerJob` + wait for rolling restart) |
| `deployment_id` on FAILED/DELETED | — | Created fresh with that ID |
| Both set, both live | Trainer reused, deployment re-attached | The warm-iteration case |

Full mechanics in [`references/recipes.md`](references/recipes.md) and [`references/reattach.md`](references/reattach.md).

---

## Operational flows

The skill has focused pages for the two most common operational tasks:

- **Promote a checkpoint to a deployable model** → read [`references/promote.md`](references/promote.md). Includes a decision tree that inspects the user's `checkpoints.jsonl` and recipe version to pick trainer-first vs legacy deployment-first; asks the user explicitly only when the flow can't be determined.
- **Re-attach a running deployment to a new trainer** → read [`references/reattach.md`](references/reattach.md). Covers `setup_or_reattach_deployment()`, when NOT to re-attach, and the raw SDK pattern.

Other one-off tools live in `training/tools/`:

- `reconnect_and_adjust_lr.py` -- reconnect to a running trainer and change LR mid-run
- `verify_logprobs.py` -- check train-inference numerical alignment

See [`references/tools.md`](references/tools.md).

---

## Directory layout

```
training/
  recipes/     Fork these: sft_loop.py, rl_loop.py, dpo_loop.py, orpo_loop.py, igpo_loop.py
  utils/       Shared config, infra, client, losses, data loading, logging
  utils/rl/    RL-specific: losses, training loop, TIS, R3, IGPO
  tools/       Standalone operational scripts (promote, reconnect, verify_logprobs)
  examples/    Worked examples with datasets and bash runners
  renderer/    Custom renderers (Gemma4, Minimax M2, Nemotron)
  tests/       Unit, smoke, e2e tests
```

---

## References

- [`references/recipes.md`](references/recipes.md) -- Config fields, trainer/deployment reuse semantics, manual vs shape path
- [`references/reattach.md`](references/reattach.md) -- Re-attach workflow and when not to use it
- [`references/promote.md`](references/promote.md) -- Checkpoint promotion flow-detection decision tree
- [`references/examples.md`](references/examples.md) -- Where each worked example lives and what to copy from it
- [`references/tools.md`](references/tools.md) -- Standalone tools in `training/tools/`
- [`references/utils.md`](references/utils.md) -- `utils/` modules: configs, infra, client, checkpoints, logging
- [`references/rl.md`](references/rl.md) -- RL loss selection, TIS, R3, training loop parameters, IGPO
- [`references/renderers.md`](references/renderers.md) -- Custom renderer interface, HF parity testing

---

## Install

```bash
cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre "fireworks-ai>=1.0.0a61" tinker-cookbook
uv pip install -e .
```

`--pre` is required; stable `0.x` does not include `fireworks.training.sdk`.

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/unit tests/test_smoke_imports.py    # no API key needed
pytest tests/                                     # needs FIREWORKS_API_KEY
```
