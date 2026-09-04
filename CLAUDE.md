# Fireworks Training Cookbook

Ready-to-run Training API recipes built on the [Fireworks Python SDK](https://github.com/fw-ai-external/python-sdk) (`fireworks.training.sdk`). Only `training/` is relevant -- ignore other top-level directories.

**Skills (v2.2.0):** Start here → [`skills/GETTING-STARTED.md`](skills/GETTING-STARTED.md)

- [`skills/research/`](skills/research/SKILL.md) — interview-driven planning: method, data, eval, cookbook
- [`skills/configure/`](skills/configure/SKILL.md) — plan, run, monitor, evaluate, deploy, and tear down training
- [`skills/debug/`](skills/debug/SKILL.md) — triage stuck, failed, or low-quality runs
- [`skills/discover/`](skills/discover/SKILL.md) and [`skills/fireworks-training/`](skills/fireworks-training/SKILL.md) — redirect stubs for older installs

**Protocol changes:** Any change to the Training API SDK, Tinker protocol, trainer/deployment payloads, checkpoint semantics, hotload flow, optimizer-step semantics, or recipe/SDK compatibility contract must update the relevant skill docs in `skills/`. Update or delete stale guidance so agents do not preserve outdated protocol behavior.

**Region placement:** Cookbook config classes retain an explicit `region` field
for compatibility with existing runners, but cookbook code must not add
`deployment_region`, hard-code default regions, infer region from
accelerator/shape, or copy a trainer region into a hot-load deployment. Leave
unset values unset so the backend RLOR trainer/deployment gateway can select
defaults and enforce colocation.
