---
name: fireworks-agent
description: Run end-to-end fine-tuning on Fireworks via the Fireworks Agent (a.k.a. Pilot Agent, exposed through `firectl session`). Use this skill whenever the user asks to fine-tune, SFT, DPO, ORPO, post-train, custom-train, auto-train, hyperparameter-tune, or hp-sweep a model on Fireworks — including `qwen3-4b`, `qwen3-8b`, `qwen3p5-9b`, `qwen3p5-27b`, `qwen3p5-35b-a3b`, `llama-v3p3-70b-instruct`, `kimi-k2p5`, `kimi-k2p6`, `glm-5p1`, `qwen3p5-397b-a17b`, or any other Fireworks-hosted base model. Also use it for kicking off a training session, monitoring or resuming a session, answering the agent's mid-run questions, approving training plans, troubleshooting `failed` sessions, listing/cancelling/deleting sessions, and figuring out how to call a fine-tuned model after training. Triggers on commands like `firectl session create / get / events / list / update / cancel / delete`, on terms like "Pilot Agent", "Fireworks Agent", "auto-tune on Fireworks", "fine-tune on Fireworks", and on session IDs of the form `1779157376-41e1`. Runs read commands (`create`, `get`, `events`, `list`) autonomously; routes mutating commands (`update`, `cancel`, `delete`) through user confirmation because they spend real money and influence training decisions.
---

# Fireworks Agent

The **Fireworks Agent** is an autonomous fine-tuning agent exposed through `firectl session`. You give it a single natural-language instruction — `"SFT qwen3-8b on my customer-support dataset"` — and it:

1. Inspects the dataset (schema, size, distribution).
2. Picks or confirms a base model from the validated set.
3. Proposes a plan (model, dataset, LoRA config, eval path, cost estimate) and waits for your approval.
4. Runs training — a single LoRA job for smoke runs, a full hyperparameter sweep + promoted winner for larger runs.
5. Registers the fine-tuned model so it's immediately callable via the standard inference API (no separate deployment needed — served through implicit live-merge LoRA).
6. Writes a structured final report with model ID, losses, metrics URL, actual cost, and reproducible Python code.

It is interactive — at decision points (model/dataset/eval choices, plan approval, and for sweep runs the promotion gate) it pauses and asks you a question. Your job (or the assistant's) is to surface that question verbatim, get your answer, and send it back.

This skill is the operator's guide: how to start a session, stream events, answer the agent's questions, recover from failures, and clean up.

> **Two rules that govern everything below.**
>
> 1. **Session continuity.** If you're resuming work on an existing session, **always** send follow-ups to that same `<session-id>` via `session update`. Never create a new session for a follow-up — it loses all prior context, restarts dataset inspection, and re-bills any setup work.
> 2. **Confirmation rule for assistants.** Read and initiation commands (`create`, `get`, `events`, `list`) are safe to run without asking. The three commands that have side effects — `session update`, `session cancel`, `session delete` — should be confirmed with the user first. `update` in particular sends a message directly to the training agent and influences how training proceeds (e.g. approving a plan starts spending real money on GPU time).

---

## For coding agents (Claude / Cursor / Codex / etc.)

If you are an LLM agent reading this skill on a customer's machine, optimize for the patterns below. Humans can skip this section.

### Session status state machine

A session moves through a small set of states. Use them to decide what to do next:

```
                    [user instruction]
                            │
                            ▼
                       ┌─────────┐
   ┌───────────────────▶ running │◀──────────────┐
   │   (session update) └────┬────┘              │
   │                         │                   │
   │       ┌─────────────────┼─────────────────┐ │
   │       │                 │                 │ │
   │       ▼                 ▼                 ▼ │
   │  ┌─────────┐       ┌──────────┐    ┌────────┴───┐
   └──┤ waiting │       │succeeded │    │   failed   │
      └─────────┘       └──────────┘    └────────────┘
       (terminal:        (terminal,      (terminal,
        loop back        Phase 5)         Phase 6)
        with update)
```

- `running` → keep streaming with `events --wait`.
- `waiting` → extract the `ask_user` block, present verbatim, get confirmation, send `update -n` once.
- `succeeded` / `failed` / `cancelled` → terminal. Do **not** try to update the session; create a new one if more work is needed.

### Cost-ceiling heuristic

Every plan-approval `waiting` state contains an estimated-cost table. Parse it and:

- **Estimate < $5**: safe to auto-approve for an explicit smoke/CI flow when the user has already given a one-shot "go" earlier in the conversation.
- **$5 – $50**: always confirm with the user before sending `Approved, proceed.`.
- **> $50**: confirm AND summarize the line items (model size, dataset rows, sweep grid breadth) — the cost is usually high because the agent picked a big sweep grid that can be narrowed.
- **No cost table found in the `waiting` block**: do **not** approve; ask the user or send `"Please re-state the plan with an explicit cost estimate."`.

### Polling cadence and stuck detection

Sessions can stay in `running` for tens of minutes without visible progress (queue waits, dataset downloads, training time). Use:

- **Stream with `events --wait`** as your default — it's a long-poll that returns when state changes. Don't busy-loop `session get`.
- **If `events --wait` returns before terminal state**, the stream just dropped — re-issue it.
- **Heuristic for "stuck"**: more than 10 minutes since the last `status_info:` line AND status is still `running`. Surface this to the user; do not auto-cancel.
- **Hard ceiling**: more than 2 hours total wall time on a small SFT, more than 8 hours on a sweep run — abnormal, surface and ask.

### Don't do these things

| Bad pattern | Why | Do instead |
|---|---|---|
| Create a new session to "retry the same question" | Loses context, double-bills setup | `session update <same-id> -n "<corrected response>"` |
| Auto-approve any plan above a few dollars | User loses cost control | Show the cost table, ask explicitly |
| Use `--dry-run` to "preview" a session | It actually creates and runs the session in the backend | Just print the instruction string to the user |
| Use `-o json` / `--output json` on any session command | Silently ignored; output is still text. The `--output` flag in the help is **non-functional** today | Parse the text with `grep` / `awk` / `sed` |
| Polish or paraphrase the agent's `ask_user` block before showing the user | The agent's wording sometimes encodes options the user needs to see verbatim | Forward the block as-is |
| Run two `session update` calls in parallel on the same session | Race condition in the agent's state | Always wait for the next `waiting` or `[done]` before the next `update` |
| Delete a `failed` session before reading its events | You lose the only forensic trail | `events` first, then optionally `delete` |

### Text-parsing recipes

There is **no JSON output**. Use these shell snippets verbatim:

```bash
# Extract session ID from `session create` stdout.
SESSION_ID=$(firectl session create --api-key $FIREWORKS_API_KEY -n "<instr>" \
              | awk '/^Session ID:/{print $3}')

# Get current status (one of: running, waiting, succeeded, failed, cancelled).
STATUS=$(firectl session get $SESSION_ID --api-key $FIREWORKS_API_KEY \
          | awk '/^Status:/{print $2}')

# Extract the latest `status_info` block (agent's current question or update).
firectl session events $SESSION_ID --api-key $FIREWORKS_API_KEY 2>/dev/null \
  | awk '/\] status_info:/{buf=""} {buf=buf"\n"$0} /\[done\] session status: waiting/{print buf}' \
  | tail -100

# After `succeeded`: extract the final fine-tuned model ID from the completion report.
MODEL_ID=$(firectl session events $SESSION_ID --api-key $FIREWORKS_API_KEY 2>/dev/null \
            | grep -oE 'accounts/[a-z0-9_-]+/models/[a-z0-9-]+' \
            | tail -1)

# After `succeeded`: extract the training job ID (alphanumeric, 8 chars).
JOB_ID=$(firectl session events $SESSION_ID --api-key $FIREWORKS_API_KEY 2>/dev/null \
          | awk -F'`' '/Training Job ID/{print $2}' | tail -1)
```

### Pre-flight before `session create`

Run these three checks before spending compute on a new session. They take < 5 seconds total:

```bash
# 1. Key has Pilot scope (returns 403 if not).
firectl session list --api-key $FIREWORKS_API_KEY > /dev/null 2>&1 \
  || { echo "API key lacks Pilot scope"; exit 1; }

# 2. Account is implicit — discover what account this key is bound to.
# Service-account keys are tied to one account; this prints existing dataset names from that account.
firectl list datasets --api-key $FIREWORKS_API_KEY | head -3

# 3. (If the user named a specific dataset) confirm it's reachable on this account.
firectl get dataset <dataset-id> --api-key $FIREWORKS_API_KEY > /dev/null 2>&1 \
  || { echo "Dataset not reachable on this account"; exit 1; }
```

If your key is scoped to one account and the user wants a different account, pass `-a <account-id>` on every `firectl` call (don't try to rewrite `~/.fireworks/auth.ini`).

### Account discovery

There's no `firectl whoami` command. To figure out what account your key is bound to:

```bash
# Option A — cheapest. Get any dataset and read the full resource name.
DATASET=$(firectl list datasets --api-key $FIREWORKS_API_KEY 2>/dev/null \
            | awk 'NR>2{print $1; exit}')
ACCOUNT=$(firectl get dataset "$DATASET" --api-key $FIREWORKS_API_KEY 2>/dev/null \
            | awk '/^Name:/{split($2, a, "/"); print a[2]}')
echo "ACCOUNT=$ACCOUNT"

# Option B — for interactive `firectl auth login` flows only (service-account keys are opaque).
python3 -c "
import base64, json, configparser, os
c = configparser.ConfigParser()
c.read_string('[d]\n' + open(os.path.expanduser('~/.fireworks/auth.ini')).read())
t = c['d'].get('id_token', '')
if t:
    payload = t.split('.')[1] + '=='
    print(json.loads(base64.urlsafe_b64decode(payload))['fireworks_account'])
"
```

In practice, you can also just read the first event in the stream — the worker name is prefixed `optimize-<account>-<session-id>-...`.

---

## Task → command

| Task | Command |
|---|---|
| Start a new fine-tuning session | [§2 `session create`](#2-create-the-session) |
| Stream events from a running session | [§3 `session events --wait`](#3-stream-and-monitor) |
| Check the current status of a session | `firectl session get <id>` |
| Respond to a question the agent is waiting on | [§4 `session update -n ...`](#4-handle-waiting-states) |
| List all your sessions | [§Reference: list](#reference-list-sessions) |
| Stop a session that's running | [§7 `session cancel`](#7-cancel-or-delete) |
| Delete a session record | [§7 `session delete`](#7-cancel-or-delete) |
| Resume monitoring after a network drop | [§3 fallback poll](#fallback-poll-with-get) |
| Diagnose a `failed` session | [§6 failure recovery](#6-handle-failures) |
| Replay history without retraining | `firectl session events <id>` (without `--wait`) |

---

## 0. Prerequisites

You need:

- **`firectl`** installed and on your `$PATH`. See [Fireworks CLI installation](https://docs.fireworks.ai/tools-sdks/firectl/firectl).
- **A Fireworks API key with Pilot Agent scope.** A regular user API key is **not** enough — calls will fail with `HTTP 403: This scope requires specific permissions not present on your API key`. You need a **service-account key** scoped to your account. Have an admin create one:

  ```bash
  firectl api-key create --service-account=<your-service-account>
  ```

  Save it to a `.env` file in your project (the rest of this skill assumes `$FIREWORKS_API_KEY` is set; some teams use `$FIREWORKS_PILOT_KEY` or `$PI_API_KEY` instead — pick a name your team understands):

  ```bash
  echo 'FIREWORKS_API_KEY=fw_...' >> .env
  source .env
  ```

- **(Optional) A dataset** already uploaded to Fireworks if you want supervised fine-tuning on your own data — the agent can also discover and pick one for you from the account. See [Datasets](https://docs.fireworks.ai/fine-tuning/datasets).
- **(Optional) An account override.** Service-account keys are usually tied to one account. To run against a different account, pass `-a / --account-id <acct>` on any command, or set `account_id` in `~/.fireworks/auth.ini`.

```bash
# Sanity check before doing anything else.
firectl session list --api-key $FIREWORKS_API_KEY
# Expect: a list (possibly empty) of sessions.
# If you get HTTP 403, your key doesn't have Pilot scope — see above.
```

Every command in this skill passes `--api-key $FIREWORKS_API_KEY` explicitly so it works whether your shell has the key in the environment or in a `.env` file you've sourced.

---

## 1. Decide on the instruction

The agent takes a single natural-language instruction. Be specific about three things if you know them; otherwise leave them out and the agent will ask or pick a sensible default:

- **Task** — `"Run supervised fine-tuning"`, `"DPO fine-tune"`, `"Evaluate"`, etc.
- **Model** — e.g. `"qwen3-8b"`, `"llama-v3p3-70b-instruct"`. Use one of the supported names (see the tip below).
- **Dataset** — full resource name, e.g. `"accounts/my-acct/datasets/customer-support-v3"`.

Good example:

> `"Run supervised fine-tuning on qwen3-8b using dataset accounts/my-acct/datasets/customer-support-v3"`

Minimal example (the agent will ask you the rest):

> `"Fine-tune a small model on accounts/my-acct/datasets/customer-support-v3"`

You do **not** need to pre-specify learning rate, batch size, number of epochs, LoRA rank, or hardware shape — the agent fills those in. For larger runs it runs a hyperparameter sweep and picks the winner; for small/smoke runs it uses sensible defaults straight away.

> **Discovering supported models.** The agent only fine-tunes models on a curated list. If you don't know what's supported, just ask in your instruction (`"What text models can you fine-tune?"`) — the agent prints its supported-models reference. As of writing, the smallest/cheapest options are `qwen3-4b`, `qwen3-4b-instruct-2507`, `qwen3-8b`, and `qwen3p5-9b`; large options include `llama-v3p3-70b-instruct`, `kimi-k2p5`, `kimi-k2p6`, `glm-5p1`, and `qwen3p5-397b-a17b`. The list evolves, so let the agent be the source of truth.

---

## 2. Create the session

```bash
firectl session create \
  --api-key $FIREWORKS_API_KEY \
  -n "Run supervised fine-tuning on qwen3-8b using dataset accounts/my-acct/datasets/customer-support-v3"
```

**Flags:**

| Flag | Purpose |
|---|---|
| `-n` / `--instruction` *(required)* | The natural-language instruction. |
| `--scope` *(default: `optimize`)* | The agent scope. The Pilot/optimize scope is the only public one today. |
| `-a` / `--account-id` | Override the account from `~/.fireworks/auth.ini`. Useful if your service-account key is scoped to a different account. |

> **Avoid `--dry-run`.** Despite the help text, `--dry-run` currently still creates and runs the session in the backend. If you want to verify your instruction before launch, just paste it into a comment or read it aloud — do not rely on `--dry-run` to be a no-op.
>
> **`--output / -o` is non-functional.** The help text advertises `text|json|flag` but the flag is silently ignored — output stays in text format. Don't plan on JSON parsing; use the text recipes from the "For coding agents" section above.

The command prints a `<session-id>` (e.g. `1779157376-41e1`). Save it — every subsequent command needs it.

---

## 3. Stream and monitor

Stream events live with `--wait`:

```bash
firectl session events <session-id> \
  --api-key $FIREWORKS_API_KEY --wait
```

> **`--wait` is required.** Without it, the command dumps existing events and exits immediately. With it, the stream stays open until the session reaches a terminal state (`succeeded`, `failed`, `cancelled`) or pauses (`waiting`).

> **The full stream is very verbose** — it includes every bash invocation the agent runs and its full output, often echoed multiple times. If you just want the human-readable narrative, pipe through a noise filter:
>
> ```bash
> firectl session events <session-id> --api-key $FIREWORKS_API_KEY --wait \
>   | grep -E "status_info|ask_user|\[done\]|error|failed|JOB_STATE"
> ```

What you'll see, roughly in order:

1. **Dataset inspection** — the agent lists datasets on the account, downloads candidates, inspects schema / row counts / sample previews.
2. **Model resolution** — confirms which base model and tokenizer will be used; if your requested model isn't supported, this is the first waiting state with a counter-offer (e.g. "qwen3-0.6b isn't supported, use qwen3-4b instead?").
3. **Plan + cost** — pauses with `waiting`, shows a plan table (model, dataset, LoRA rank, LR, epochs, eval path) plus an estimated cost. Approve to proceed.
4. **Training** — launches one or more training jobs and polls them. For a small / smoke run, this is one LoRA job; for a real run with a fresh dataset, the agent runs an HP sweep across LRs / ranks and then promotes the winner to a full training run (a second `waiting` state).
5. **Final report** — a structured markdown block with the fine-tuned model ID, job ID, final training and validation loss, metrics URL, an actual-vs-estimated cost table, and reproducible Python code to call the model.

> Small smoke runs (≤ a few hundred rows, default config) tend to **skip the HP sweep + promotion gate** and go straight from plan approval → single training job → done. Large or unconstrained runs typically include the full sweep cycle.

Outcome routing:

- Stream exits with **`waiting`** → see [§4 handle waiting states](#4-handle-waiting-states).
- Stream exits with **`succeeded`** → see [§5 completion](#5-completion).
- Stream exits with **`failed`** → see [§6 handle failures](#6-handle-failures).

### Fallback: poll with `get`

If the stream drops unexpectedly (network blip, laptop sleep, SSH disconnect), don't recreate the session — just poll status until it changes, then re-stream:

```bash
until firectl session get <session-id> --api-key $FIREWORKS_API_KEY 2>/dev/null \
        | grep -E "waiting|succeeded|failed|cancelled"; do
  sleep 10
done
firectl session get <session-id> --api-key $FIREWORKS_API_KEY
```

Then resume with `firectl session events <session-id> --wait`.

---

## 4. Handle waiting states

A `waiting` state means the agent has a question for you. This phase loops with §3 until the session reaches a terminal state.

### Step 1 — capture the last event timestamp

Before doing anything else, record the timestamp of the last event currently in the stream. You'll use it to filter out history on the next stream so you don't re-read everything:

```bash
LAST_TS=$(firectl session events <session-id> --api-key $FIREWORKS_API_KEY 2>/dev/null \
  | grep -oE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]' | tail -1)
```

### Step 2 — surface the agent's question exactly

Extract the last `status_info` block — everything from the final `status_info` line up to `[done] session status: waiting`:

```bash
firectl session events <session-id> --api-key $FIREWORKS_API_KEY 2>/dev/null \
  | awk '/\] status_info:/{buf=""} {buf=buf"\n"$0} /\[done\] session status: waiting/{print buf}'
```

Read the agent's question **verbatim**. Don't paraphrase — wording sometimes encodes specific options.

### Step 3 — send your response

`session update` sends a message directly to the training agent and influences its decisions. **Confirm the response with the user before running it.**

```bash
firectl session update <session-id> \
  --api-key $FIREWORKS_API_KEY \
  -n "<your response>"
```

> **Note the flag is `-n` / `--instruction`** (same name as on `session create`). There is no `-i` shorthand — calls with `-i` fail with `unknown shorthand flag: 'i'`.

**Flags:** `-n` / `--instruction` *(required)*, `--scope` *(default: `optimize`)*.

> **Heads-up: the agent may bundle multiple questions in one `ask_user` block.** A typical first waiting state asks about model choice, dataset choice, and evaluation path all at once. Answer all of them in a single `session update`, e.g. `"Yes use qwen3-4b. Pick any small dataset in the account. Path A (validation loss only) is fine."`.

Common questions and good responses:

| Question theme | Good response |
|---|---|
| "Which evaluation path — validation loss only, or a separate evaluator?" | `"validation loss is fine"` (skips evaluator setup) |
| "Here's the proposed plan — approve to proceed?" | `"Approved, proceed."` |
| "Sweep complete — promote the winning config?" | `"Proceed with the winning config."` |
| "Which dataset format — chat or text?" | Answer with the actual format of your file. |
| "What's your eval target — accuracy, F1, BLEU, custom?" | Be specific; the agent wires the evaluator to match. |

### Step 4 — resume streaming, filtered

After sending the update, stream again but drop everything at or before `LAST_TS` so the user only sees new traces:

```bash
firectl session events <session-id> --api-key $FIREWORKS_API_KEY --wait 2>/dev/null \
  | awk -v ts="$LAST_TS" '/^\[20/{show=($0>ts)} show{print}'
```

How the `awk` works:

- Lines that start with a timestamp (`[2025-...`): set `show=1` if the line is newer than `LAST_TS`, else `show=0`.
- Continuation lines (no timestamp prefix): inherit `show` from the previous timestamped line — so multi-line `status_info` blocks aren't split.

**This loop — stream → `waiting` → capture timestamp → surface question → get response → `update` → filtered stream — repeats until the session reaches `succeeded`, `failed`, or `cancelled`.**

---

## 5. Completion

When status is `succeeded`, the agent writes a structured markdown report as the final stream output. Don't paraphrase it — it already contains everything you need. The report includes:

- **Fine-tuned model ID** (e.g. `accounts/<acct>/models/qwen3-4b-smoke-lr15e5-lora8-1779158908`).
- **Training job ID** (e.g. `vfes4zbn`) — needed for follow-up actions like `firectl jobs get`.
- **Metrics URL** — a GCS link to `metrics.jsonl` with per-step loss history.
- **Final training and validation loss**.
- **Actual vs estimated cost** with a percentage delta.
- **A reproducible Python snippet** for calling the new model.

### What the report looks like (real run, smoke test, ~$0.37 actual)

```text
## How To Access The Final Model

- Model ID:        accounts/pyroworks/models/qwen3-4b-smoke-lr15e5-lora8-1779158908
- Training Job ID: vfes4zbn
- Usage Note:      Served via implicit live-merge LoRA deployment on B200 x1.

## Training Metrics

| Metric                | Value  |
|-----------------------|--------|
| Final Training Loss   | 0.7044 |
| Final Validation Loss | 0.7771 |
| Training Duration     | ~5 min |

## Cost Summary

| Category  | Estimated | Actual | Delta |
|-----------|-----------|--------|-------|
| Inference | $0.00     | $0.00  | 0%    |
| Training  | $0.42     | $0.37  | -12%  |
| Total     | $0.42     | $0.37  | -12%  |
```

### Calling the fine-tuned model

The agent recommends the [Fireworks Python SDK](https://pypi.org/project/fireworks-ai/) (`pip install fireworks-ai`) — it handles auth and live-merge LoRA deployment transparently:

```python
import os
from fireworks import Fireworks

client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
FINAL_MODEL_ID = "accounts/<acct>/models/<fine-tuned-model-id>"

response = client.chat.completions.create(
    model=FINAL_MODEL_ID,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ],
    temperature=0.0,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

Or via curl:

```bash
curl https://api.fireworks.ai/inference/v1/chat/completions \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "accounts/<acct>/models/<fine-tuned-model-id>",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

> **Note on serving.** Fine-tuned LoRA adapters are served through an **implicit live-merge deployment** on shared base-model hardware — you don't need to explicitly create a deployment for the fine-tuned model. The base model's serverless deployment merges the LoRA at request time. First request after long idle may be slightly slower (adapter warm-up).

For more on calling the deployed model, see [Querying models](https://docs.fireworks.ai/guides/querying-text-models).

---

## 6. Handle failures

When status is `failed`, the agent's last few `status_info` lines almost always contain a human-readable explanation — read them first before anything else:

```bash
firectl session events <session-id> --api-key $FIREWORKS_API_KEY 2>/dev/null \
  | grep -E "status_info|error|failed|JOB_STATE" | tail -30
```

Show the relevant lines to the user clearly. **Do not retry automatically** — the right path depends on the error class. Offer three options:

1. **Retry** — create a new session with the same instruction. Best for transient infra errors (job-pool exhaustion, GCS timeouts).
2. **Modify and retry** — adjust the instruction based on the error and create a new session. Best for anything in your control: "model not supported", "dataset not found", "schema mismatch", instruction ambiguity.
3. **Abandon** — cancel the failed session and clean up. Best when the workload simply doesn't fit (e.g. context length above what any supported shape can handle).

### Known patterns

These are patterns reported in practice; treat them as starting hypotheses, not guarantees. Always confirm against the actual `status_info` and any `error:` lines in the stream.

| Symptom in events | Hypothesis | First action |
|---|---|---|
| `model not supported` / `not in the validated base model list` | You named a base model that isn't on the agent's supported list (see `readonly/skills/sft/references/base_models.md`, which the agent will print on request) | Re-prompt with a supported model — `qwen3-4b`, `qwen3-8b`, `qwen3p5-9b`, `llama-v3p3-70b-instruct`, etc. |
| `dataset not found` / `dataset access denied` | Wrong account or dataset name in the instruction | `firectl dataset list` to find the right resource name |
| `JOB_STATE_FAILED` early after `Launching SFT training job` | Schema problem in the dataset (column names, JSON shape) | Re-prompt with a different dataset, or inspect locally with `firectl dataset download` |
| `quota exceeded` / `429` | Account fine-tuning quota hit | Wait or contact Fireworks support |
| Long stall at `Launching SFT training job` then session times out | Training-pool queue full | Retry later; ephemeral |
| Session stays `running` for hours without a `status_info` update | Worker pool issue | Cancel + retry; if it keeps happening, contact Fireworks support with the session ID |

---

## 7. Cancel or delete

**Both commands need user confirmation before running.**

**Cancel** stops a running session but keeps the record (you can still read events):

```bash
firectl session cancel <session-id> --api-key $FIREWORKS_API_KEY
```

**Delete** removes the session record entirely — *irreversible*:

```bash
firectl session delete <session-id> --api-key $FIREWORKS_API_KEY
```

Both accept `--scope` (default `optimize`).

---

## Reference: list sessions

```bash
firectl session list --api-key $FIREWORKS_API_KEY
```

**Flags:**

| Flag | Purpose |
|---|---|
| `--filter` | AIP-160 filter expression, e.g. `status=waiting` |
| `--order-by` | Field to sort by; append ` desc` for descending |
| `--no-paginate` | Return all results without pagination |
| `--page-size` | Max results per page |
| `--page-token` | Specific page to fetch |
| `--scope` | Default `optimize` |
| Alias | `ls` |

> `-o / --output` is listed in the help but non-functional (see Phase 2 note). Output stays in text format.

Useful one-liner — find every session currently waiting on you:

```bash
firectl session list --api-key $FIREWORKS_API_KEY --filter "status=waiting"
```

---

## Full example session

Start to finish, single SFT run. Tested end-to-end against a 500-row chat dataset on `qwen3-4b`. Actual run: estimated $0.42, actual $0.37, ~36 min wall clock (planning + ~5 min training):

```bash
# 0. Prereqs (service-account key with Pilot scope)
source .env                    # FIREWORKS_API_KEY=fw_...
firectl session list --api-key $FIREWORKS_API_KEY    # smoke check; expect a list, not 403

# 1. Create — single natural-language instruction
firectl session create --api-key $FIREWORKS_API_KEY \
  -n "Run supervised fine-tuning on qwen3-4b using a small chat dataset from the account."
# → Session ID: 1779157376-41e1

# 2. Stream events (low-noise view)
firectl session events 1779157376-41e1 --api-key $FIREWORKS_API_KEY --wait \
  | grep -E "status_info|ask_user|\[done\]|error|failed|JOB_STATE"
# → ... [done] session status: waiting

# 3. Surface the agent's first question and respond
firectl session events 1779157376-41e1 --api-key $FIREWORKS_API_KEY 2>/dev/null \
  | awk '/\] status_info:/{buf=""} {buf=buf"\n"$0} /\[done\] session status: waiting/{print buf}' \
  | tail -100
# (read the question, decide an answer)
firectl session update 1779157376-41e1 --api-key $FIREWORKS_API_KEY \
  -n "Yes use qwen3-4b. Pick any small dataset from the account. Path A (validation loss only) is fine."

# 4. Stream again until the plan-approval gate
firectl session events 1779157376-41e1 --api-key $FIREWORKS_API_KEY --wait \
  | grep -E "status_info|ask_user|\[done\]"
# → ... [done] session status: waiting   (plan + cost table shown)

firectl session update 1779157376-41e1 --api-key $FIREWORKS_API_KEY \
  -n "Approved, proceed."

# 5. Stream until succeeded (or background-poll while you do other work)
firectl session events 1779157376-41e1 --api-key $FIREWORKS_API_KEY --wait \
  | grep -E "status_info|\[done\]|JOB_STATE"
# → ... [done] session status: succeeded
#   The final stream block contains the model ID, losses, metrics URL, and cost.

# 6. Call the fine-tuned model (Python SDK is the recommended path)
python - <<'PY'
import os
from fireworks import Fireworks

client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
resp = client.chat.completions.create(
    model="accounts/<acct>/models/<fine-tuned-model-id>",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.0, max_tokens=128,
)
print(resp.choices[0].message.content)
PY
```

---

## Cheat sheet

The four commands you'll use 95% of the time:

```bash
# Start
firectl session create  --api-key $FIREWORKS_API_KEY -n "<instruction>"

# Watch (low-noise)
firectl session events <id> --api-key $FIREWORKS_API_KEY --wait \
  | grep -E "status_info|ask_user|\[done\]|JOB_STATE|error|failed"

# Respond when waiting
firectl session update  <id> --api-key $FIREWORKS_API_KEY -n "<response>"

# Status check
firectl session get     <id> --api-key $FIREWORKS_API_KEY
```

Helper to print just the agent's last question:

```bash
firectl session events <id> --api-key $FIREWORKS_API_KEY 2>/dev/null \
  | awk '/\] status_info:/{buf=""} {buf=buf"\n"$0} /\[done\] session status: waiting/{print buf}' \
  | tail -100
```

---

## See also

- [Fireworks Fine-Tuning docs](https://docs.fireworks.ai/fine-tuning/fine-tuning-models) — high-level overview of the fine-tuning surface.
- [`firectl` reference](https://docs.fireworks.ai/tools-sdks/firectl/firectl) — full CLI surface, including dataset, model, and deployment commands.
- [`skills/dev/SKILL.md`](../dev/SKILL.md) — Training SDK power-user path (fork a recipe, run your own training loop, custom losses, RL, hotload).
