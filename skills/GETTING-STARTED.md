# Fireworks Training skills

The agent acts as an expert ML engineer who helps you post-train models on
Fireworks, and it can do the work itself: pick the method and data, launch the
job, monitor it, evaluate it, and clean up. It aims for the shortest path to a
measured result. Expect short answers and sensible defaults you can override.
It normally asks only the next
decision-critical question, grouping tightly related blockers when that saves
churn. It always shows the complete plan and cost before anything is created.

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

If you are just getting started, ask for something free to try:

```text
Is there something free I can run right now to see how training works?
```

The agent offers a starter, shows one plan, and after you approve it launches
and monitors the job. Built-in starters (see
`research/references/starter-jobs.md`):

| Starter | Data | Easiest surface |
|---|---|---|
| Muse Glimmer prompt-routing SFT | account-relative dataset ID `onboarding-prompt-router` | Managed SFT |
| Code-first SFT with a base-vs-tuned comparison | your own chat JSONL | Serverless Training API |
| Muse Glimmer receipt-extraction SFT | account-relative dataset ID `onboarding-cord-receipts` | Managed SFT |

Muse Glimmer 30B LoRA SFT is free for jobs started before `2026-11-01T00:00:00-07:00`
(Oct 31) on Managed SFT (UI or API) and on the Serverless Training API
([source](https://docs.fireworks.ai/fine-tuning/models)). RL, DPO,
deployments, inference, and Dedicated trainers are not covered. After the
deadline the agent resolves live pricing instead.

To run it yourself, follow the [Muse Glimmer mini cookbook](../training/examples/muse_glimmer/README.md), or in short:

```bash
git clone --depth 1 https://github.com/fw-ai/cookbook && cd cookbook/training
python3 -m venv .venv && source .venv/bin/activate && pip install -e .   # needs Python 3.11+: check python3 --version, else use python3.12
# if venv fails with "ensurepip is not available": sudo apt-get install python3-venv, or
# uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e .
python -m examples.sft.muse_glimmer_starter plan
RUN=router-$(date +%Y%m%d-%H%M%S)   # fresh ID per run
python -m examples.sft.muse_glimmer_starter launch --surface managed --run-id $RUN
python -m examples.sft.muse_glimmer_starter watch --run-id $RUN
```

Other examples:

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

The agent avoids questionnaires and asks only what changes the next action.
Research and debug are read-only. Configure shows the complete parameters and
cost before asking for approval, then runs the job itself.

## Five-minute no-spend smoke

1. Start a new chat with the first example above.
2. Confirm the first reply offers concrete starters and asks one question.
3. Pick the prompt-router starter.
4. Confirm the next reply is a complete plan (Managed SFT, `$0` training line)
   and that nothing has been created.
5. Do not approve the plan.

## Authentication

The agent runs jobs through the Fireworks Python SDK with `FIREWORKS_API_KEY`.
Create a key at <https://app.fireworks.ai/settings/users/api-keys>, then set it
in the terminal the agent uses, with hidden input:

```bash
read -s FIREWORKS_API_KEY
echo
export FIREWORKS_API_KEY
```

Or put `FIREWORKS_API_KEY=...` in a gitignored `.env` at the repository root.
Never paste an API key into chat. `firectl` stays useful for your own
read-only checks; it blocks agent mutations by design.

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
