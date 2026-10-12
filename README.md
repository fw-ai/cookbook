# Fireworks AI Cookbook

Training loops and worked examples for RL (GRPO, IGPO, SAO/PPO), preference
optimization (DPO, ORPO), SFT, distillation and embedding fine-tuning on
[Fireworks](https://fireworks.ai). Full docs: [Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction).

## Quick Start

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

See [`training/README.md`](./training/README.md) for recipes and examples.

## For AI Agents

These skills bring Fireworks training know-how into compatible AI agents through
progressive disclosure. Each entry point loads only the workflow guidance
needed for the task, then follows linked Fireworks documentation and runnable
cookbook examples when deeper detail is needed. The skill set provides three
task-specific skills: **research**, **configure**, and **debug**.

| Skill | What it does | Try |
|---|---|---|
| [Research](skills/research/SKILL.md) | Helps decide whether training is the right intervention. It gathers the task, data, evaluation criteria, and constraints, then recommends a method and the closest runnable cookbook entry. It does not launch training. | *"Which cookbook entry fits prompt routing to small vs big models?"* |
| [Configure](skills/configure/SKILL.md) | Turns a training goal into an executable plan for managed SFT, DPO, ORPO, or RFT, as well as serverless or dedicated Training API setups. It validates inputs, estimates cost, and supports running, monitoring, evaluation, deployment, resume, and teardown. It shows the complete plan and asks for approval before spend or mutation. | *"SFT qwen3-8b on my JSONL. Show the plan, but do not start yet."* |
| [Debug](skills/debug/SKILL.md) | Diagnoses stuck, failed, slow, or low-quality training runs. It gathers runtime evidence, classifies the failure, suggests the safest next action, and does not retry or mutate resources without approval. | *"My job is stuck RUNNING at 0%."* |

### Claude Code

```bash
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
```

### Cursor

```bash
npx --yes skills add fw-ai/cookbook -g \
  -s fireworks-training -s research -s configure -s debug -a cursor -y
```

### Codex

```bash
npx --yes skills add fw-ai/cookbook -g \
  -s fireworks-training -s research -s configure -s debug -a codex -y
```

The repository also includes [`.codex-plugin/plugin.json`](.codex-plugin/plugin.json)
for packaging the skill set as a Codex plugin. The skills use portable Agent
Skills Markdown and can be consumed by other compatible agents. The commands
above cover the three validated installation paths. `firectl` may still require
mutating commands to be run manually in the user's terminal when its AI-agent
safety guard is active.

## Repository Structure

```
training/           Training API cookbook (primary surface)
  recipes/          One training loop per file; fork to change the algorithm
  utils/            Pieces recipes assemble: config, data, losses, rollout, metrics
  examples/         Task code + one recipe call (dedicated and serverless twins)
  case-studies/     End-to-end notebooks
eval/               Reproducible evaluation packages and benchmark adapters
skills/             Research, configure and debug agent workflows
archived/           Legacy integrations and earlier cookbook content
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
