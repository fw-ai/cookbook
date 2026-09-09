---
name: fireworks-training
description: >-
  Compatibility redirect and shared reference carrier for the Fireworks
  training skills. Use research to choose method, data, evaluation, and a
  cookbook entry; use configure to plan and run training; use debug for stuck,
  failed, or low-quality runs. Keep this skill installed so the three entry
  skills can load its progressive references.
---

# Fireworks training

This workflow is split into three focused skills:

| Goal | Skill |
|---|---|
| Choose method, data, evaluation, and cookbook entry | `research` |
| Plan, run, monitor, deploy, or resume training | `configure` |
| Diagnose a failed, stuck, or low-quality run | `debug` |

Do not execute a training workflow from this file. Route to the matching skill.

This directory also carries shared detailed references used by `configure` and
`debug`. Cursor and Codex installs must include `fireworks-training` alongside
the three entry skills.
