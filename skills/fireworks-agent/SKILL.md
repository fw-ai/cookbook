---
name: fireworks-agent
description: Deprecated. The Pilot agent (`firectl session`) is being decommissioned; use skills/fireworks-fine-tuning/ instead. Not for new work.
---

# Fireworks Agent — deprecated

**This skill is deprecated.** The server-side Pilot agent (`firectl session ...`, `--scope optimize`) is being decommissioned in favor of the coding-agent + skill model, where your agent drives the `firectl` training primitives directly.

**Use [`skills/fireworks-fine-tuning/`](../fireworks-fine-tuning/SKILL.md) instead** — specifically [`references/orchestrate-from-agent.md`](../fireworks-fine-tuning/references/orchestrate-from-agent.md), which maps every old `session` verb to the direct `firectl` command that replaces it.

| You were using | Now use |
|---|---|
| `session create -n "..."` | Preflight -> plan/confirm -> `firectl sftj create ...` |
| `session events --wait` | Poll `firectl sftj get` + `sftj export-metrics` |
| `session update` | The agent decides inline / asks the user |
| `session get` / `list` | `firectl sftj get` / `sftj list`, `deployment get` / `list` |
| `session cancel` / `delete` | `firectl sftj cancel`; delete artifacts via their own resource commands |

For the SDK power-user path (custom loops, RL recipes), see [`skills/dev/SKILL.md`](../dev/SKILL.md).
