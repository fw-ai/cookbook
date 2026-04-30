# Renderer verifier

Validates that a cookbook renderer produces the same tokens the live
Fireworks gateway emits, and that loss weights are consistent with
the "hard-append → weight 0, native-generated → weight 1" rule. Ships
a probe CLI, a batch triage runner, a single-file React viewer that
highlights every audit-table row by provenance and inspection-rule
match, and a YAML-driven rule engine.

> **How to use it**: see [`cookbook/skills/verifier/SKILL.md`](../../skills/verifier/SKILL.md).
>
> **Implementing a new renderer to validate**: see [`cookbook/skills/renderer/SKILL.md`](../../skills/renderer/SKILL.md).

<!-- screenshot of the GUI goes here -->

## Layout

```
training/verifier/
├── cli.py                python -m training.verifier render | inspect
├── serve.py              python -m training.verifier.serve
├── triage.py             python -m training.verifier.triage
├── spinup_deployment.py  personal-deployment helper
├── utils/                engine modules (probe, inspect_rules, hf_parity, …)
├── rules/                inspect_rules.yaml — single source of truth
└── viewer/               single-file React GUI
```
