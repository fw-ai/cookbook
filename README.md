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

Set your API key and run a recipe:

```bash
export FIREWORKS_API_KEY="your-api-key"
python -m recipes.sft_loop
```

See [`training/README.md`](./training/README.md) for recipe configuration and examples.

## Repository Structure

```
training/           Training SDK recipes, utilities, and examples
  recipes/          Fork-and-customize training loop scripts
  utils/            Shared config, data loading, losses, metrics
  examples/         Worked examples (RL, SFT, DPO, ORPO)
  tests/            Unit and end-to-end tests
integrations/       Third-party integrations (AgentCore, SageMaker)
multimedia/         Video and VLM notebooks
archived/           Legacy cookbook content
```

## Contributing

See the [Contribution Guide](./Contribution.md).

## Support

- [Documentation](https://fireworks.ai/docs)
- [Discord](https://discord.gg/9nKGzdCk)
- [Open an issue](https://github.com/fw-ai/cookbook/issues/new)
