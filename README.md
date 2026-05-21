# Fireworks AI Cookbook

Ready-to-run training recipes for reinforcement learning (GRPO, DAPO, GSPO, CISPO), preference optimization (DPO, ORPO), and supervised fine-tuning (SFT) on [Fireworks](https://fireworks.ai).

> **Full documentation**: [Fireworks Training SDK Reference](https://docs.fireworks.ai/fine-tuning/training-sdk/introduction)

## Quick Start

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training
conda create -n cookbook python=3.12 -y && conda activate cookbook
pip install --pre -e .
```

See [`training/README.md`](./training/README.md) for configuration, recipes, and examples.

## For AI Agents

The primary reference for agents working in this repo is **[`skills/dev/SKILL.md`](skills/dev/SKILL.md)** — it maps tasks and error signals to specific reference files. Start there, not the READMEs.

## Repository Structure

Only `training/` is actively developed. Other top-level directories (`integrations/`, `multimedia/`, `archived/`) are kept for backward compatibility.

```
training/           Training SDK recipes, utilities, and examples
  recipes/          Fork-and-customize training loop scripts
  utils/            Shared config, data loading, losses, metrics
  examples/         Worked examples (RL, SFT, DPO, ORPO)
  verifier/         Renderer correctness validator + live React viewer
  tests/            Unit and end-to-end tests
skills/             Agent skills and reference docs
```

## Renderer skills

- [`skills/renderer/SKILL.md`](skills/renderer/SKILL.md) — implementing
  a new renderer (chat template, stop semantics, weight-mask
  invariants, registration, dev loop).
- [`skills/verifier/SKILL.md`](skills/verifier/SKILL.md) — validating
  a renderer against the live Fireworks gateway and the upstream
  HuggingFace chat template. The verifier ships a probe CLI, a batch
  triage runner, and a single-file React viewer that highlights every
  audit-table row by provenance, weight, and inspection-rule match.

## Contributing

See the [Contribution Guide](./Contribution.md).

## Support

- [Documentation](https://fireworks.ai/docs)
- [Discord](https://discord.gg/9nKGzdCk)
- [Open an issue](https://github.com/fw-ai/cookbook/issues/new)
