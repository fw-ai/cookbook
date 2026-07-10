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

Two agent skills cover this repo: **[`skills/fireworks-fine-tuning/SKILL.md`](skills/fireworks-fine-tuning/SKILL.md)** for **managed fine-tuning** (SFT/DPO/RFT driven through `firectl`), and **[`skills/dev/SKILL.md`](skills/dev/SKILL.md)** for the **Training SDK** power-user path (recipes, custom loops, RL internals). Each maps tasks and error signals to specific reference files — start there, not the READMEs.

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

## Fine-tuning skills

- **[`skills/fireworks-fine-tuning/SKILL.md`](skills/fireworks-fine-tuning/SKILL.md)** — the primary skill for **managed fine-tuning** (SFT / DPO / RFT). Your coding agent drives the `firectl` primitives directly: choose a method, prepare + validate a dataset, launch + monitor a job, pick a training shape, deploy, and troubleshoot. Successor to the Pilot agent.
- [`skills/dev/SKILL.md`](skills/dev/SKILL.md) — the Training SDK power-user path (fork a recipe, custom training loop, RL internals, hotload, distillation).
- _Deprecated:_ [`skills/fireworks-agent/`](skills/fireworks-agent/SKILL.md) (Pilot, `firectl session`) and [`skills/research/fireworks-auto-tune/`](skills/research/fireworks-auto-tune/SKILL.md) (customer `firectl` SFT) — both superseded by `skills/fireworks-fine-tuning/`.

## Contributing

See the [Contribution Guide](./Contribution.md).

## Support

- [Documentation](https://fireworks.ai/docs)
- [Discord](https://discord.gg/9nKGzdCk)
- [Open an issue](https://github.com/fw-ai/cookbook/issues/new)
