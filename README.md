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

**Just installed?** → **[`skills/GETTING-STARTED.md`](skills/GETTING-STARTED.md)** — open a new chat and paste a smoke-test prompt (no spend).

Three skills ship with one plugin install — **research**, **configure**, and **debug**:

| Skill | Role |
|---|---|
| [`skills/research/SKILL.md`](skills/research/SKILL.md) | Interview-driven planning: method, data, eval, cookbook entry |
| [`skills/configure/SKILL.md`](skills/configure/SKILL.md) | Plan, run, monitor, evaluate, deploy, and tear down training |
| [`skills/debug/SKILL.md`](skills/debug/SKILL.md) | Triage stuck, failed, or low-quality runs |

[`skills/discover/SKILL.md`](skills/discover/SKILL.md) and [`skills/fireworks-training/SKILL.md`](skills/fireworks-training/SKILL.md) are redirect stubs for older installs.

### Claude Code

```bash
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
```

### Cursor

```bash
npx --yes skills add fw-ai/cookbook -g -s research -s configure -s debug -a cursor -y
```

### Codex

```bash
npx --yes skills add fw-ai/cookbook -g -s research -s configure -s debug -a codex -y
```

The repository also includes [`.codex-plugin/plugin.json`](.codex-plugin/plugin.json)
for packaging the same skill as a Codex plugin. The skill is portable Agent
Skills Markdown; Cursor and Codex installation is validated with the `skills`
CLI and is not limited to the Claude compact interface. `firectl` may still
require mutating commands to be run manually in the user's terminal when its
AI-agent safety guard is active.

### After install — try this

Open a **new chat** in Cursor, Claude Code, or Codex. You do not @-mention skills.

| You say | Entry |
|---|---|
| *"Which cookbook entry fits prompt routing to small vs big models?"* | research |
| *"SFT qwen3-8b on my JSONL — show the plan, don't start yet"* | configure |
| *"My job is stuck RUNNING at 0%"* | debug |

Full walkthrough: [`skills/GETTING-STARTED.md`](skills/GETTING-STARTED.md).

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
skills/             research, configure, debug (+ discover/fireworks-training redirects)
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
