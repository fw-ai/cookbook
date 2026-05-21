# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/fw-ai-external/python-sdk) (`fireworks.training.sdk`). Only `training/` is relevant -- ignore other top-level directories.

**Skills:**
- [`skills/dev/`](skills/dev/SKILL.md) — day-to-day training work (greenfield setup, debugging, hotload, RL recipe internals, checkpoint promotion). Entry in `SKILL.md` maps tasks and error signals to specific reference files under `skills/dev/references/`. The SDK repo points here; do not maintain a parallel skill there.
- [`skills/research/`](skills/research/SKILL.md) — research-grade training work (new objectives, train-inference alignment, custom reward pipelines). **Coming soon.**

**Protocol changes:** Any change to the Training SDK, Tinker protocol, trainer/deployment payloads, checkpoint semantics, hotload flow, optimizer-step semantics, or recipe/SDK compatibility contract must update the relevant skill docs in `skills/`. Do not only add new skills; also update or delete stale skill guidance so agents do not preserve outdated protocol behavior.
