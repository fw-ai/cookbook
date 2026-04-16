# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/stainless-sdks/fireworks-ai-python) (`fireworks.training.sdk`). Only `training/` is relevant -- ignore other top-level directories.

**Skills:** this repo ships the only maintained training skills.

- [`skills/dev/`](skills/dev/SKILL.md) — picking a recipe, running an example, resolving shapes from a profile, running `promote_checkpoint.py` / `reconnect_and_adjust_lr.py`, where checkpoint state lives.
- [`skills/debug/`](skills/debug/SKILL.md) — common mistakes: mixing trainer-first with deployment-first, forgetting `--hot-load-deployment-id` on legacy promote, skipping training / deployment shapes, recovering a promote orphan.

Both skills point the agent at specific files; they are not a tutorial.
