# Fireworks Training Cookbook

Ready-to-run Training API recipes built on the [Fireworks Python SDK](https://github.com/fw-ai-external/python-sdk) (`fireworks.training.sdk`). Only `training/` is relevant -- ignore other top-level directories.

**Skills:**
- [`skills/fireworks-training/`](skills/fireworks-training/SKILL.md) — the single training skill for managed SFT/DPO/ORPO/RFT and Training API serverless or dedicated workflows. The root skill owns routing, confirmation, and lifecycle; progressive references cover cookbook recipes, SDK internals, resume, deployment, and debugging.

**Protocol changes:** Any change to the Training API SDK, Tinker protocol, trainer/deployment payloads, checkpoint semantics, hotload flow, optimizer-step semantics, or recipe/SDK compatibility contract must update the relevant skill docs in `skills/`. Update or delete stale guidance so agents do not preserve outdated protocol behavior.

**Region placement:** Cookbook config classes retain an explicit `region` field
for compatibility with existing runners, but cookbook code must not add
`deployment_region`, hard-code default regions, infer region from
accelerator/shape, or copy a trainer region into a hot-load deployment. Leave
unset values unset so the backend RLOR trainer/deployment gateway can select
defaults and enforce colocation.
