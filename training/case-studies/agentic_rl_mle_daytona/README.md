# A tiny Karpathy-style autoresearch loop in a Daytona sandbox

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): give an AI agent a small but real
ML task and let it improve a `train.py` autonomously. The agent has a **`run_training(code)` tool** — it calls the
tool with a full `train.py` (structured argument, no code-block parsing), we run it **inside a
[Daytona](https://www.daytona.io) sandbox** under a fixed time budget, **keep it only if it improved** (the
"ratchet"), and feed the result back so it tries again.

```
spin up ONE Daytona sandbox (data baked in)
  loop:  agent calls run_training(code)  ->  run in sandbox (fixed budget)  ->  VAL_ACC
         keep if new best  ->  return result to agent  ->  agent tries again
```

**Super easy:** it's all in one notebook. No prepare step, no Kaggle, no GPU, no fine-tuning — a serverless
Fireworks model proposes code and a Daytona sandbox runs it.

## What it does

- Boots one stock `daytona-medium` sandbox and installs `datasets`/`pyarrow`.
- Downloads a **real** dataset into it: [`sealuzh/app_reviews`](https://huggingface.co/datasets/sealuzh/app_reviews)
  (app-store reviews, 1-5 `star`), subsamples a fixed slice to `train.parquet` / `val.parquet` (immutable across
  iterations so runs are comparable). Task: **predict `star` from the `review` text** (5-class).
- Writes **no starter code** — we only describe the contract (read the parquet files, print `VAL_ACC=<float>`);
  the agent writes `train.py` from scratch on its first tool call.
- Runs the agent loop: the model calls the `run_training(code)` tool; we write + run it under a fixed wall-clock
  budget, keep it only if it beats the current best, and return the result so the agent tries again.
- Prints the score history and the winning script.

## Files

- `mle_bench_daytona.ipynb` — the whole thing.

## Prerequisites

```bash
pip install daytona python-dotenv langchain-fireworks
```

`.env` (repo root): `FIREWORKS_API_KEY`, `DAYTONA_API_KEY`. Model defaults to serverless
`accounts/fireworks/models/glm-5p2` (change `MODEL` in the config cell).

## Run

Open `mle_bench_daytona.ipynb` and run top to bottom. Tune `N_ITERS` (loop iterations) and `RUN_BUDGET_S`
(per-experiment wall clock) in the config cell.

## Notes / caveats

- The agent is instructed to train only on the train split; `val.parquet` includes the labels (needed to score),
  so a misbehaving agent could peek. Fine for a demo — swap to a held-out grader if you want it airtight.
- Each experiment runs arbitrary model-written code; that's exactly why it runs in a Daytona sandbox, not locally.
- To point the loop at a different task, change the data step and baseline `train.py` in section 2 (keep the
  `VAL_ACC=<float>` contract).
