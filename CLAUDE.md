# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/stainless-sdks/fireworks-ai-python) (`fireworks.training.sdk`). Only `training/` is relevant -- ignore other top-level directories.

**Skills:** This repo ships two Claude Code skills under `skills/`:

- [`skills/dev/`](skills/dev/SKILL.md) — picking a recipe, configuring a run, customizing losses and rewards, reusing trainers and deployments across iterations.
- [`skills/debug/`](skills/debug/SKILL.md) — recovering from a broken run: promoting a checkpoint, re-attaching a deployment, verifying logprobs, the standalone tools under `training/tools/`.

Both skills cross-link to the public docs at <https://docs.fireworks.ai/fine-tuning/training-api>; prefer the skill for current-branch details.
