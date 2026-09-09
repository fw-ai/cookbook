# Fireworks Training skills

One installation provides three entry points:

| Skill | Use it for |
|---|---|
| `research` | Choose method, data, evaluation, and cookbook entry |
| `configure` | Plan, run, monitor, deploy, or resume training |
| `debug` | Diagnose a stuck, failed, or low-quality run |

The `fireworks-training` compatibility skill carries the detailed shared
references. Keep it installed with all three entry skills.

## Install

Claude Code:

```bash
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
```

Cursor:

```bash
npx --yes skills add fw-ai/cookbook -g \
  -s fireworks-training -s research -s configure -s debug -a cursor -y
```

Codex:

```bash
npx --yes skills add fw-ai/cookbook -g \
  -s fireworks-training -s research -s configure -s debug -a codex -y
```

## Start

Open a new chat and describe your goal. Do not mention a skill name unless you
want to force a specific workflow.

Examples:

```text
I have customer-support questions and expected policy documents. Help me decide
whether training or retrieval optimization is the right next step.
```

```text
Plan an SFT run for my labeled JSONL dataset. Do not create anything yet.
```

```text
My training job is running but has made no progress for 15 minutes. Diagnose it.
```

The agent asks one structured question at a time. Research and debug are
read-only. Configure must show the complete parameters and cost before asking
for approval.

## Five-minute no-spend smoke

1. Start a new chat with the first example above.
2. Confirm the Research banner appears.
3. Answer one structured question.
4. Confirm no dataset upload or training creation occurs.
5. Ask to hand off to Configure.
6. Confirm Configure asks for path and method details before presenting a plan.
7. Do not approve the final plan.

## Authentication

When Fireworks access is needed, enter a scoped API key in your terminal:

```bash
read -s FIREWORKS_API_KEY
echo
export FIREWORKS_API_KEY
firectl whoami
```

Never paste an API key into chat.

## Journey telemetry

Before the first structured question, the skill explains that Fireworks
product analytics may record the registered question ID, selected option ID,
agent surface, and an optional privacy-validated task summary. It never sends
raw chat messages, datasets, credentials, or paths.

Say `do not track this session` to keep interaction telemetry local. Training
still works.

Remote journey events use `firectl skill-journey record` when supported by the
installed firectl version. Existing BigQuery and CDC data remain authoritative
only for training job and billing outcomes.

## Update

For Cursor or Codex:

```bash
npx skills update
```

Claude Code plugin installs follow the plugin update flow.
