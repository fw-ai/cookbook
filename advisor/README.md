# The Advisor: a frontier reviewer for any coding agent

A tiny, self-contained tool that gives any AI coding agent a **critique-only
second opinion** before it finishes. Your agent does the work; the advisor reads
the **working git diff** and returns scored issues + fixes. It cannot edit files.

This is the harness from our report **"Open-source agents with a frontier
reviewer"** ([Fireworks blog](https://fireworks.ai/blog)). The result: a cheap
**open-source worker on Fireworks** (e.g. GLM-5.2), checked by a **frontier
reviewer** (Claude), matches frontier-only quality at a fraction of the cost.

**Two models, two roles:**

| Role | Model | Runs on | Key |
|---|---|---|---|
| **Worker** (does the task) | open-source, e.g. **GLM-5.2** | **Fireworks** serverless | `fw_...` (in your agent harness) |
| **Advisor** (reviews the diff) | **frontier, Claude Opus** | Anthropic | `sk-ant-...` (this tool, `ADVISOR_API_KEY`) |

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

## 1. Worker — run your agent on Fireworks (GLM-5.2)

Point your coding agent's model at a Fireworks open model. First, get a Fireworks
API key:

1. Sign up or log in at [app.fireworks.ai](https://app.fireworks.ai) and verify
   your account.
2. Open your profile menu → **Settings** → **API Keys**.
3. Click **Create API Key**, give it a name, and copy it (Fireworks shows the full
   key only once). It looks like `fw_...`.

GLM-5.2 is served at `accounts/fireworks/models/glm-5p2`. A call looks like:

```bash
curl https://api.fireworks.ai/inference/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -d '{
    "model": "accounts/fireworks/models/glm-5p2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Wire that into your coding agent however your harness configures its model (see the
[Fireworks quickstart](https://docs.fireworks.ai/getting-started/quickstart) for
SDK examples). This `fw_` key powers the **worker** — separate from the
**advisor's** Anthropic key in the next step.

## 2. Advisor — add the frontier reviewer (Claude)

You need [Node.js](https://nodejs.org) 18+ and an **Anthropic API key** (the
reviewer is Claude). Grab the one file and set the key:

```bash
curl -O https://raw.githubusercontent.com/fw-ai/cookbook/main/advisor/advisor.mjs

export ADVISOR_API_KEY=sk-ant-...        # Anthropic key for the Claude reviewer
# defaults: ADVISOR_BASE_URL=https://api.anthropic.com, ADVISOR_MODEL=claude-opus-4-8
```

`ADVISOR_API_KEY` is the **advisor's** key — it pays for the Claude review calls,
not your Fireworks worker.

## 3. Wire it into your agent

Tell your agent (the GLM-5.2 worker) *when* to call the advisor — add one line
where your harness reads project instructions:

> Before declaring any non-trivial change complete, run
> `node advisor.mjs review` with a focused `--question`, fix any **critical**
> issues it raises, then re-run your tests. Skip it for trivial edits.

Keep it to **one review before finishing** — in our ablations, consulting the
advisor more (e.g. a planning call too) didn't improve results and cost more.

That's it — your agent now has an advisor to talk to. 🎉
