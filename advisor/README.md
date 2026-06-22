# The Advisor: a frontier reviewer for any coding agent

A tiny, self-contained tool that gives any AI coding agent a **critique-only
second opinion** before it finishes. Your agent does the work; the advisor reads
the **working git diff** and returns scored issues + fixes. It cannot edit files.

This is the harness from our report **"Open-source agents with a frontier
reviewer"** ([Fireworks blog](https://fireworks.ai/blog)) — an open-source worker
model checked by a stronger reviewer matched frontier-only quality at a fraction
of the cost. Here it's packaged as **one file** (`advisor.mjs`) you can drop into
your own agent, using **Fireworks serverless inference** for the reviewer.

## How it works

One model call with one carefully-worded prompt. The whole design is that the
reviewer is a **skeptic, not a cheerleader** — it distrusts the agent's prose and
treats the `git diff` as ground truth:

> "do NOT accept the agent's framing, arithmetic, or boundaries … the `<worktree>`
> section is **ground truth** … the agent's prose claims of edits are not …
> Confidence: score each issue 0–100 … Critical issues — ONLY issues scoring ≥80."

That calibrated, evidence-gated prompt (the full text is `SYSTEM_PROMPT` in
`advisor.mjs`) is what makes it useful instead of noisy. It's anchored on the
diff, so it works the same in **any** harness.

## 1. Setup

You need [Node.js](https://nodejs.org) (18+) and a **Fireworks API key**
([get one](https://app.fireworks.ai)). Point the advisor at Fireworks serverless
inference:

```bash
export ADVISOR_API_KEY=fw_...                                  # your Fireworks key
export ADVISOR_BASE_URL=https://api.fireworks.ai/inference     # (this is the default)
export ADVISOR_MODEL=accounts/fireworks/models/glm-5p2         # the reviewer model
```

Grab the one file:

```bash
curl -O https://raw.githubusercontent.com/fw-ai/cookbook/main/advisor/advisor.mjs
```

## 2. Try it standalone

From inside any git repo with uncommitted changes:

```bash
node advisor.mjs review --question "are there race conditions in the token refresh?"
```

It prints a ~300-word critique: critical issues (≥80 confidence) → suggested
fixes → low-confidence notes → what it could and couldn't verify.

## 3. Wire it into your agent

Two steps — give the agent the command, and tell it *when* to use it.

**Command** the agent runs (from the repo root):

```bash
node advisor.mjs review --question "<what to check>" --files "<key files>"
```

**Nudge** — add one line to your agent's instructions so it actually calls the
advisor. Put it where your harness reads project instructions:

| Agent | File |
|---|---|
| Claude Code | `CLAUDE.md` |
| Codex · OpenCode | `AGENTS.md` |
| Cursor | `.cursor/rules/advisor.md` |

> Before declaring any non-trivial change complete, run
> `node advisor.mjs review` with a focused `--question`, fix any **critical**
> issues it raises, then re-run your tests. Skip it for trivial edits.

Keep it to **one review before finishing** — in our ablations, consulting the
advisor more (e.g. a planning call too) didn't improve results and cost more.

## Choosing the reviewer model

The advisor talks to any Anthropic-compatible `/v1/messages` endpoint, so the
reviewer is your choice:

| Reviewer | Set | Notes |
|---|---|---|
| **Fireworks serverless** (default) | `ADVISOR_MODEL=accounts/fireworks/models/<model>` | one key, all-Fireworks. Use a **large, capable** model — the reviewer should be at least as strong as your worker. |
| **Frontier (Opus)** | `ADVISOR_BASE_URL=https://api.anthropic.com`, `ADVISOR_MODEL=claude-opus-4-8`, `ADVISOR_API_KEY=sk-ant-...` | the report's headline setup — the strongest results. |

**Tip:** the advisor helps most when the *reviewer is stronger than the worker*.
Pointing a model at its own output (same model as worker and reviewer) gave little
lift in our tests — pick a bigger reviewer, or the frontier option, for the
clearest gains.

## What it sends

For full transparency, each call sends the reviewer: your `<question>` and
`<focus>`, any `<sources>` you pass via `--files`, optional `<context>` (e.g. a
short note on what you did), and the `<worktree>` — `git status` + `git log` +
`git diff HEAD`. Nothing else. Read `advisor.mjs` end to end in two minutes.
