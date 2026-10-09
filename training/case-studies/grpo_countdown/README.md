# Serverless non-agentic GRPO: own the Countdown loop

Train a single-turn math policy with **GRPO** on the Fireworks **serverless**
Training API — the loop you write yourself, with no trainer job and no inference
deployment to provision. The runnable teaching notebook is
[`countdown_grpo.ipynb`](countdown_grpo.ipynb).

**Is this you?** You can grade answers automatically (right/wrong or a scored
check), but you do **not** have gold reasoning to imitate — and the task is a
**single-turn** completion, not a tool-calling agent. You want to **own the GRPO
loop** on serverless (smoke it cheaply), then optionally express the **same**
loop for dedicated GPUs. Classic cases: arithmetic puzzles, unit-testable code
snippets, extraction with a validator.

**Not this notebook** if you need:

| Need | Go here instead |
| --- | --- |
| Managed one-call RFT (no loop) | [`reasoning_rl`](../reasoning_rl/) (GSM8K) |
| Multi-turn tool-calling agent | [`agentic_rl_text2sql`](../agentic_rl_text2sql/) |
| Coding agents / sandboxes at scale | Harbor recipes under [`examples/rl/harbor/`](../../examples/rl/harbor/) |
| Preference pairs (chosen/rejected) | DPO case studies / `examples/serverless_dpo/` |

## The idea

1. **Connect once.** `FiretitanServiceClient(.../training/v1/serverless)` gives
   you both a LoRA training client and per-snapshot samplers.
2. **Score locally.** `composite_reward` in
   [`countdown_rewards.py`](../../examples/serverless_rl/countdown_rewards.py)
   gives partial credit for format, number usage, and hitting the target — no
   GPU required to sanity-check it.
3. **GRPO update.** Sample a group per prompt, standardize rewards within the
   group, drop zero-spread groups, then `forward_backward` + `optim_step`.
4. **Same algorithm, dedicated compute.** §10 of the notebook shows the
   dedicated expression via [`recipes/rl_loop.py`](../../recipes/rl_loop.py)
   — config swap, not a second notebook (the same serverless/dedicated
   side-by-side as [`sft_prompt_router`](../sft_prompt_router/)).

The production loop is already implemented as
[`examples/serverless_rl/countdown_rl.py`](../../examples/serverless_rl/countdown_rl.py).
This case study **imports** it; it does not paste GRPO math.

## Data

Rows are `{"messages": [...], "ground_truth": "{\"numbers\": [...], \"target\": N}"}`
from the [TinyZero Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
tasks — note `ground_truth` is a JSON-encoded **string** (that is what
`countdown_rl.prepare_dataset` writes, what the shipped sample rows contain, and
what the loop's eval path (`json.loads(row["ground_truth"])`) and the §10
dedicated snippet expect; the reward's `parse_ground_truth` in
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
`group_relative_advantages` and the live runner import
`examples/serverless_rl/countdown_rl.py` — they need the cookbook training
install (`uv sync`), not just `fireworks-ai`.

## Cost

Serverless bills per token. Smoke defaults in the notebook are intentionally
tiny (`STEPS=2`, `GROUP_SIZE=4`). Raise them when you want a visible climb.
Serverless teardown is a no-op; the dedicated section shows trainer + deployment
cleanup for that path only.
