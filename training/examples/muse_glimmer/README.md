# Muse Glimmer mini cookbook

Fine-tune Muse Glimmer 30B for free on the example datasets, then on your own
data. "Muse Glimmer 30B SFT is free via UI and Serverless Training API until
10/31. No credit card required"
([source](https://docs.fireworks.ai/fine-tuning/models)). The promotion covers
LoRA SFT jobs started before `2026-11-01T00:00:00-07:00` (end of Oct 31 Pacific); full-parameter SFT, RL, deployments, inference,
DPO, and Dedicated trainers are not covered. The starter script refuses to launch after
the deadline unless you pass `--allow-paid`.

| | |
|---|---|
| Base model | `accounts/fireworks/models/muse-glimmer-30b` |
| Tokenizer | `meta-models/Muse-Glimmer-30B` |
| Renderer | `muse_glimmer` |
| Example data | `onboarding-prompt-router` (600 rows), `onboarding-cord-receipts` (300 rows), preloaded in eligible accounts |

## 0. Setup

```bash
git clone --depth 1 https://github.com/fw-ai/cookbook && cd cookbook/training
python3 -m venv .venv && source .venv/bin/activate && pip install -e .   # needs Python 3.11+: check python3 --version, else use python3.12
read -s FIREWORKS_API_KEY && export FIREWORKS_API_KEY   # never paste it into chat
python -m examples.sft.muse_glimmer_starter plan        # read-only: account, data, promotion, cost
```

Create the key at <https://app.fireworks.ai/settings/users/api-keys>, or put
`FIREWORKS_API_KEY=...` in a gitignored `training/.env`.

## 1. Pick a path

| You want | Path | Command |
|---|---|---|
| The quickest first result | Managed SFT | `launch --surface managed` |
| To see the loop and measure base vs tuned accuracy on your own data | Serverless Training API SFT | `launch --surface serverless --file data.jsonl` |
| Vision (receipt image → JSON) | Managed SFT | `launch --surface managed --dataset onboarding-cord-receipts --max-context-length 8192` (image tokens count toward context; 8192 leaves headroom) |
| RL on a verifiable task (paid, not covered) | Serverless Training API RL | DeepMath command in step 4 |

The Managed SFT path is also available in the UI at
<https://app.fireworks.ai/dashboard/fine-tuning>. Use Dedicated trainers only
for full-parameter training or long sustained runs; they are not free.

## 2. Text SFT on Managed SFT (about 10 minutes)

```bash
RUN=router-$(date +%Y%m%d-%H%M%S)   # a fresh ID per run; never reuse a deleted job's ID
python -m examples.sft.muse_glimmer_starter launch --surface managed --run-id $RUN
python -m examples.sft.muse_glimmer_starter watch --run-id $RUN
```

Rerunning `launch` with the same run ID reuses that job only if every setting
matches; otherwise it stops and lists the differences. `watch` prints the state
and status (percent and epoch) on every poll, a summary of the rendered
training samples once they are ready, and train and eval loss in the final
summary when the job finishes. It stops if a running job makes no progress for
30 minutes. Success looks like eval loss falling below its first value and the job
reaching `JOB_STATE_COMPLETED`. The output is the LoRA model
`accounts/<account>/models/muse-starter-$RUN`.

## 3. Your data on the Serverless Training API, with a behavioral eval

The Serverless Training path trains on a local chat JSONL file (format in
step 5), runs the loop in your process, and compares base and tuned on rows it
holds out.

```bash
RUN=tickets-$(date +%Y%m%d-%H%M%S)
python -m examples.sft.muse_glimmer_starter prepare --file data.jsonl --run-id $RUN
python -m examples.sft.muse_glimmer_starter launch --surface serverless --file data.jsonl --run-id $RUN
```

`prepare` is local and free. It validates the file (refusing malformed rows
and listing them), holds out 40 rows (`--n-eval`), writes `train.jsonl` and
`holdout.jsonl` to `muse_starter_logs/$RUN/`, and prints one rendered example
with its token count. Read the example before launching. `launch` refuses rows
longer than `--max-seq-len` (2,048), then samples the base and tuned adapters
on the holdout in the same session. It writes
`muse_starter_logs/$RUN/metrics.jsonl` (loss and gradient norm per step) and
`muse_starter_logs/$RUN/summary.json` (exact match, per-key accuracy when the
answers are JSON, and a paired test, base vs tuned). Sampling in-session needs no deployment (Muse public serverless
inference was retired on 2026-09-25), but it is billed per token separately
from the free training; `launch` ends with a cost line giving the sampled token
counts. Before the session closes, the tuned adapter is saved
as `accounts/<account>/models/muse-starter-sl-$RUN`; pass `--no-promote` to skip.

## 4. RL on DeepMath (Serverless Training API, paid)

RL is not covered by the promotion; price the run before launching it. The
DeepMath example defaults to a different model; pass the Muse settings and
keep the first run small:

```bash
python -m examples.rl.deepmath.train_deepmath_serverless \
  --base-model accounts/fireworks/models/muse-glimmer-30b \
  --tokenizer-model meta-models/Muse-Glimmer-30B --renderer-name muse_glimmer \
  --max-rows 32 --prompt-groups-per-step 8 --completions-per-prompt 4 \
  --max-completion-tokens 4096 --max-seq-len 8192 --lora-rank 8
```

Watch the verifier reward and the eval score every five steps; a rising
reward with stable gradient norms is the success signal.

## 5. Your own data

Managed SFT takes chat JSONL, one example per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Before uploading, quarantine rows with empty or missing assistant replies,
report them, and fix or exclude them only with the user's approval; never
silently rewrite or drop user data. Check that every remaining row parses.
Then validate and upload in one step, and run step 2 on it:

```bash
python -m examples.sft.muse_glimmer_starter upload --file data/train.clean.jsonl --dataset my-tickets-v1
RUN=tickets-$(date +%Y%m%d-%H%M%S)
python -m examples.sft.muse_glimmer_starter launch --surface managed --run-id $RUN --dataset my-tickets-v1
python -m examples.sft.muse_glimmer_starter watch --run-id $RUN
```

`upload` refuses a file with empty or malformed rows and lists them; it never
drops rows itself. A few hundred clean rows are
enough for a first signal.

## 6. Clean up

```bash
python -m examples.sft.muse_glimmer_starter cleanup --run-id $RUN
```

This deletes the run's Managed SFT job, its output model, and the saved
serverless model, if they exist. Serverless runs close their session when the
script exits. Nothing in steps 2–4 keeps billing after it finishes; step 4 (RL)
and deploying a tuned model are separate, paid steps.

## With a coding agent

Install the skills ([Getting started](../../../skills/GETTING-STARTED.md)) and ask *"Is there
something free I can run right now?"*. The agent runs these same commands for
you through the SDK, shows one plan to approve, and reports results.
