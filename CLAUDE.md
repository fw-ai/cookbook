# Fireworks Training Cookbook

Ready-to-run training recipes built on the [Fireworks Training SDK](https://github.com/fw-ai-external/python-sdk) (`fireworks.training.sdk`). Only `training/` is relevant -- ignore other top-level directories.

**Skills:**
- [`skills/fireworks-fine-tuning/`](skills/fireworks-fine-tuning/SKILL.md) — **managed fine-tuning** router (SFT/DPO/RFT via public `firectl`): method choice, dataset validation, launch + monitor, training shapes, deploy, troubleshoot. Successor to the Pilot agent; start here for any managed fine-tuning task.
- [`skills/dev/`](skills/dev/SKILL.md) — day-to-day training-SDK work (greenfield setup, debugging, hotload, RL recipe internals, checkpoint promotion). Entry in `SKILL.md` maps tasks and error signals to specific reference files under `skills/dev/references/`. The SDK repo points here; do not maintain a parallel skill there.
- _Deprecated (stubs kept for link stability):_ [`skills/fireworks-agent/`](skills/fireworks-agent/SKILL.md) and [`skills/research/fireworks-auto-tune/`](skills/research/fireworks-auto-tune/SKILL.md) — superseded by `skills/fireworks-fine-tuning/`.

**Protocol changes:** Any change to the Training SDK, Tinker protocol, trainer/deployment payloads, checkpoint semantics, hotload flow, optimizer-step semantics, or recipe/SDK compatibility contract must update the relevant skill docs in `skills/`. Do not only add new skills; also update or delete stale skill guidance so agents do not preserve outdated protocol behavior.

**Region placement:** Cookbook config classes retain an explicit `region` field
for compatibility with existing runners, but cookbook code must not add
`deployment_region`, hard-code default regions, infer region from
accelerator/shape, or copy a trainer region into a hot-load deployment. Leave
unset values unset so the backend RLOR trainer/deployment gateway can select
defaults and enforce colocation.
