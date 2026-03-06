# Fireworks AI Cookbook

The Fireworks AI Cookbook provides ready-to-run recipes and utilities for training models on [Fireworks](https://fireworks.ai). It covers supervised fine-tuning (SFT), reinforcement learning (GRPO, DAPO, GSPO, CISPO), and preference optimization (DPO, ORPO) — all driven by the Fireworks Training SDK.

For full SDK documentation, see the [Fireworks Training SDK Reference](https://docs.fireworks.ai/api-reference/training-sdk/overview).

## Getting Started

Head to the [`training/`](./training) directory for installation instructions, recipe configuration, and runnable examples.

## Repository Structure

```
training/           Training SDK recipes, utilities, and examples
  recipes/          Fork-and-customize training loop scripts
  utils/            Shared config, data loading, losses, metrics
  examples/         Worked examples (e.g. deepmath GRPO)
  tests/            Unit and end-to-end tests
archived/           Legacy cookbook content (see below)
```

## Archived Content

All previous cookbook material — learning tutorials, integration examples, showcase projects, evaluation recipes, and more — has been moved to [`archived/`](./archived). See the [archived README](./archived/README.md) for details on what's there.

## Contributing

We welcome contributions! See the [Contribution Guide](./Contribution.md) for how to get started.

## Feedback & Support

- [Fireworks Documentation](https://fireworks.ai/docs)
- [Discord](https://discord.gg/9nKGzdCk)
- [Open an issue](https://github.com/fw-ai/cookbook/issues/new)
