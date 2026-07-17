---
name: fireworks-agent
description: Deprecated. The Pilot agent (`firectl session`) is being decommissioned; use skills/fireworks-fine-tuning/ instead. Not for new work.
---

# Fireworks Agent — deprecated

**This skill is deprecated.** The server-side Pilot agent (`firectl session ...`, `--scope optimize`) is being decommissioned in favor of the coding-agent + skill model, where your agent drives the `firectl` training primitives directly.

**Use [`skills/fireworks-fine-tuning/`](../fireworks-fine-tuning/SKILL.md) instead** — specifically [`references/orchestrate-from-agent.md`](../fireworks-fine-tuning/references/orchestrate-from-agent.md), which maps every old `session` verb to the direct `firectl` command that replaces it.

| You were using | Now use |
|---|---|
| `session create -n "..."` | Local preflight -> plan/confirm -> the matching `sftj` / `dpo-job` / `rftj create` command |
| `session events --wait` | Poll the matching job resource and its available metrics or linked W&B run |
| `session update` | The agent decides inline / asks the user |
| `session get` / `list` | The matching `sftj` / `dpo-job` / `rftj get` or `list`, plus deployment resources |
| `session cancel` / `delete` | The matching job `cancel`; delete artifacts through their own resource commands |

For the SDK power-user path (custom loops, RL recipes), use the separately installed [`fireworks-training` skill](https://github.com/fw-ai/cookbook/blob/main/skills/dev/SKILL.md).
