# Serverless non-agentic GRPO: Countdown task + reference loop

Train a single-turn math policy with **GRPO** on the Fireworks **serverless**
Training API — you write the task, a forkable reference loop runs it, and there
is no trainer job or inference deployment to provision. The runnable teaching notebook is
[`countdown_grpo.ipynb`](countdown_grpo.ipynb).

**Is this you?** You can grade answers automatically (right/wrong or a scored
check), but you do **not** have gold reasoning to imitate — and the task is a
**single-turn** completion, not a tool-calling agent. You want to write the
**task** (prompt, reward, rollout), smoke the GRPO loop cheaply on serverless,
then optionally run the **same** task code on dedicated GPUs. Classic cases: arithmetic puzzles, unit-testable code
snippets, extraction with a validator.

**Not this notebook** if you need:

| Need | Go here instead |
| --- | --- |
| Managed one-call RFT (no loop) | [`reasoning_rl`](../reasoning_rl/) (GSM8K) |
| Multi-turn tool-calling agent | [`agentic_rl_text2sql`](../agentic_rl_text2sql/) |
| Coding agents / sandboxes at scale | Harbor recipes under [`examples/harbor_rl/`](../../examples/harbor_rl/) |
| Preference pairs (chosen/rejected) | DPO case studies / `examples/dpo/` |

## The idea

1. **Write the task.** A rollout renders the prompt, samples one completion and
   scores it with `composite_reward` in
   [`countdown_rewards.py`](../../examples/archive/countdown/countdown_rewards.py)
   (partial credit for format, number usage, and hitting the target) — no GPU
   required to sanity-check it.
2. **Hand it to the loop.** The serverless async-RL recipe
   ([`recipes/experiment/async_rl_loop_serverless.py`](../../recipes/experiment/async_rl_loop_serverless.py))
   attaches to a pooled LoRA trainer, samples a group per prompt from the
   current snapshot, standardizes rewards within the group, skips zero-spread
   groups, then runs `forward_backward` + `optim_step` and publishes the next
   snapshot.
3. **Same task, dedicated compute.** §10 of the notebook runs the same rollout
   on [`recipes/async_rl_loop.py`](../../recipes/async_rl_loop.py) — a config
   swap, not a second notebook (the same serverless/dedicated side-by-side as
   [`sft_prompt_router`](../sft_prompt_router/)).

The task code is
[`examples/archive/countdown/countdown_rl.py`](../../examples/archive/countdown/countdown_rl.py).
This case study **imports** it; it does not paste GRPO math.

## Data

Rows are `{"messages": [...], "ground_truth": "{\"numbers\": [...], \"target\": N}"}`
from the [TinyZero Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
tasks — note `ground_truth` is a JSON-encoded **string** (that is what
[`prepare_data.py`](../../examples/archive/countdown/prepare_data.py) writes and
what the shipped sample rows contain; the reward's `parse_ground_truth` in
`countdown_rewards.py` also accepts a plain dict).
`countdown_train.sample.jsonl` (8 rows) ships so you can
see the schema. The notebook materializes a larger file via `prepare_dataset`
when you flip `PREPARE_FULL_DATASET`.

## Run

From the cookbook repo (with `training/.env` holding `FIREWORKS_API_KEY` and
`FIREWORKS_ACCOUNT_ID`):

```bash
cd training
uv sync                          # installs tinker + the local `training` package
source .venv/bin/activate
jupyter lab case-studies/grpo_countdown/countdown_grpo.ipynb
```

Kernel: **cookbook (3.12)** if you use the shared training venv. Leave
`RUN_LIVE = False` until you are ready to spend serverless tokens; teaching
cells (reward sanity, fake GRPO group) are free once deps are installed.
`group_relative_advantages` and the live run import the cookbook training
stack — they need the cookbook training install (`uv sync`), not just
`fireworks-ai`.

## Cost

Serverless bills per token. Smoke defaults in the notebook are intentionally
tiny (two steps of two prompts × four completions). Raise them when you want a
visible climb.
Serverless teardown is a no-op; the dedicated section shows trainer + deployment
cleanup for that path only.
