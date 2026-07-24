# Fireworks AI Cookbook

Ready-to-run training recipes for reinforcement learning (GRPO, DAPO, GSPO, CISPO), preference optimization (DPO, ORPO), and supervised fine-tuning (SFT) on [Fireworks](https://fireworks.ai).

> **Full documentation**: [Fireworks Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction)

## Quick Start

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training
conda create -n cookbook python=3.12 -y && conda activate cookbook
pip install --pre -e .
```

See [`training/README.md`](./training/README.md) for configuration, recipes, and examples.

## For AI Agents

One canonical skill covers the full training product: **[`skills/fireworks-training/SKILL.md`](skills/fireworks-training/SKILL.md)**. It routes managed SFT/DPO/ORPO/RFT and Training API serverless or dedicated workflows, then progressively loads the relevant cookbook, operations, or troubleshooting references.

### Claude Code

```bash
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
```

### Cursor

```bash
npx --yes skills add fw-ai/cookbook -g -s fireworks-training -a cursor -y
```

### Codex

```bash
npx --yes skills add fw-ai/cookbook -g -s fireworks-training -a codex -y
```

The repository also includes [`.codex-plugin/plugin.json`](.codex-plugin/plugin.json)
for packaging the same skill as a Codex plugin. The skill is portable Agent
Skills Markdown; Cursor and Codex installation is validated with the `skills`
CLI and is not limited to the Claude compact interface. `firectl` may still
require mutating commands to be run manually in the user's terminal when its
AI-agent safety guard is active.

## Repository Structure

`training/` is the primary development surface. `eval/` contains reproducible
evaluation packages. Legacy integrations, standalone customer scripts,
multimedia examples, and earlier cookbook content live under `archived/`.

```
training/           Training API recipes, utilities, and examples
  recipes/          Fork-and-customize training loop scripts
  utils/            Shared config, data loading, losses, metrics
  examples/         Worked examples (RL, SFT, DPO, ORPO)
  renderer/         Local renderers and correctness verifier
  tests/            Unit and end-to-end tests
eval/               Reproducible evaluation packages and benchmark adapters
skills/             One Fireworks training skill and progressive references
archived/           Legacy integrations, multimedia, and cookbook content
  tools/            Archived standalone customer scripts
```

## Evaluations

- [`eval/healthbench_professional/`](./eval/healthbench_professional/) — run
  OpenAI's HealthBench Professional through Harbor, preserve exact Fireworks
  input/output token IDs and behavior-policy logprobs, and export validated
  trajectories for RL workflows.

## Contributing

See the [Contribution Guide](./Contribution.md).

## Support

- [Documentation](https://fireworks.ai/docs)
- [Discord](https://discord.gg/9nKGzdCk)
- [Open an issue](https://github.com/fw-ai/cookbook/issues/new)
