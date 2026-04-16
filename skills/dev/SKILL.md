---
name: training-cookbook-dev
description: Develop with the Fireworks Training Cookbook. Covers configuring and running the SFT / DPO / ORPO / RL recipes, customizing losses and rewards, reusing trainer jobs and deployments across iterations, chat-template renderers, and the shared utilities under training/utils/. Use this skill when the user wants to pick a recipe, fork it, fill in a `Config`, or wire up a new training run. For recovering from a broken run (promote failed, hotload stalled, deployment unhealthy), use the sibling `debug` skill instead.
---

# Fireworks Training Cookbook — Dev

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/stainless-sdks/fireworks-ai-python) (`fireworks.training.sdk`). Use this skill when **building** a training run. For **fixing** a run that's broken, see the sibling [`../debug/SKILL.md`](../debug/SKILL.md).

**Only `training/` is relevant.** Ignore other top-level directories.

**Docs:** <https://docs.fireworks.ai/fine-tuning/training-api/cookbook> — the cookbook guide mirrors this skill's structure. Prefer this skill for current-branch details.

This skill focuses on the operational questions: what to pass in, what gets reused, which recipe to pick. For *what each training algorithm does conceptually*, see the Fireworks docs, not this skill.

---

## The two operational questions

### "What do I pass in?"

Every recipe has a `Config` dataclass at the top of the file. Always required:

- `base_model` — `accounts/fireworks/models/<name>`
- `dataset` — path to JSONL
- `tokenizer_model` — HF model name
- `log_path` — directory for `checkpoints.jsonl` and logs
- `infra.training_shape_id` — `accounts/fireworks/trainingShapes/<shape>`

Recipe-specific fields live alongside. See [`references/recipes.md`](references/recipes.md) for each recipe's required fields.

### "What gets reused?"

If you rerun with the same `log_path`, the recipe picks up the last checkpoint and resumes. If you set `init_from_checkpoint="<job>:<step>"`, it loads weights from another job and resets the step counter.

`training/utils/infra.py` provides `setup_or_reattach_deployment(...)` so consecutive training runs can share a warm deployment — see [`references/utils.md`](references/utils.md) (and the debug skill for when re-attach is the right escape hatch).

---

## Recipe selection

| Task | Recipe |
|------|--------|
| Single-turn supervised fine-tuning | `training/recipes/sft_loop.py` |
| RL with outcome rewards (GRPO) | `training/recipes/rl_loop.py` |
| Preference pairs (DPO) | `training/recipes/dpo_loop.py` |
| ORPO (paired preference w/ no reference) | `training/recipes/orpo_loop.py` |
| Frozen-lake tool-use RL demo | `training/examples/rl/frozen_lake/` |

See [`references/recipes.md`](references/recipes.md) for the Config surface of each. See [`references/rl.md`](references/rl.md) for reward-function patterns. See [`references/renderers.md`](references/renderers.md) for chat templates.

---

## Quick start

```bash
# Clone and install
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training
pip install -e .

# Copy and edit a recipe
cp recipes/sft_loop.py my_run.py
# Edit the Config at the top of my_run.py

# Run
python my_run.py
```

---

## References

- [`references/recipes.md`](references/recipes.md) — each recipe's `Config` surface and required fields
- [`references/examples.md`](references/examples.md) — `training/examples/` walkthrough
- [`references/rl.md`](references/rl.md) — RL loop, reward functions, sampling, GRPO specifics
- [`references/renderers.md`](references/renderers.md) — chat templates, message rendering, tool calls
- [`references/utils.md`](references/utils.md) — shared utilities under `training/utils/`

For promoting a checkpoint, re-attaching a deployment, or diagnosing a failed run, see the sibling [debug skill](../debug/SKILL.md).
