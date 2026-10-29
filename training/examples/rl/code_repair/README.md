# Code repair (serverless GRPO)

Single-function bug repair from [`archiki/UTGenDebug`](https://huggingface.co/datasets/archiki/UTGenDebug)
(HumanEval+ and MBPP+ functions with injected bugs). The prompt is a buggy
function and its docstring; the model replies with a fixed function, which is
run against hidden unit tests in a local sandbox. Reward is the share of test
cases passed. A reply with no code, a syntax error, a crash or a timeout scores
0, and so does a reply cut off at `--max-completion-tokens`.

> **macOS only as-is.** `sandbox.py` isolates model-written code with
> `/usr/bin/sandbox-exec`, which exists only on macOS. On Linux the grader
> fails; porting it needs another isolation layer (for example `bwrap` or a
> container) behind the same `grade()` interface.

## Files

| File | What it does |
| --- | --- |
| `prepare_data.py` | Downloads the dataset, checks that every canonical solution passes and every buggy version fails, writes `data/eval.jsonl` and `data/train_candidates.jsonl` |
| `build_dataset.py` | Keeps training rows where the base model's sampled rewards differ, writes `data/dataset.jsonl` (eval rows first) |
| `train_code_repair_serverless.py` | GRPO on the serverless pool via `recipes/experiment/async_rl_loop_serverless.py`; `--sampling-only` scores the base model instead |
| `sandbox.py`, `_worker.py`, `rewards.py`, `replies.py` | Grading: code extraction, sandboxed test run, reward |
| `eval_checkpoint.py` | Scores the base model or a sampler checkpoint on the eval rows with per-sample results |
| `compare_runs.py` | Compares two eval runs per problem, including only on problems both models finished |
| `compare_app/` | Local page that samples base and tuned side by side on one problem |

## Run it

From `training/`, with `FIREWORKS_API_KEY` set:

```bash
# 1. data
python -m training.examples.rl.code_repair.prepare_data \
    --splits mbpp_plus_fix mbpp_plus_fix_hard --eval-rows 78

# 2. score the training candidates with the base model, keep rows with reward spread
python -m training.examples.rl.code_repair.train_code_repair_serverless \
    --sampling-only --dataset training/examples/rl/code_repair/data/train_candidates.jsonl \
    --run-dir training/examples/rl/code_repair/runs/baseline_train
python -m training.examples.rl.code_repair.build_dataset \
    --sampled-rows training/examples/rl/code_repair/runs/baseline_train/sampled_rows.jsonl

# 3. train
python -m training.examples.rl.code_repair.train_code_repair_serverless \
    --eval-rows 78 --max-rows 402 --eval-samples-per-row 2 --evaluation-interval 4 \
    --max-concurrency-rollout-sample 128 --dcp-save-interval 4 \
    --run-dir training/examples/rl/code_repair/runs/full

# 4. evaluate and compare (checkpoint path is in runs/full/result.json)
python -m training.examples.rl.code_repair.eval_checkpoint \
    --checkpoint <final_sampler_snapshot> --run-dir training/examples/rl/code_repair/runs/eval_tuned_8k
python -m training.examples.rl.code_repair.eval_checkpoint \
    --max-tokens 32000 --run-dir training/examples/rl/code_repair/runs/eval_base_32k
python -m training.examples.rl.code_repair.compare_runs \
    training/examples/rl/code_repair/runs/eval_base_32k \
    training/examples/rl/code_repair/runs/eval_tuned_8k --names base32k tuned8k
```

## Reference result

`qwen3p8-27b`, 402 training rows, 9 optimizer steps (about 3 hours, about $65
at list price). Eval: 78 problems × 8 samples, temperature 1.0.

| Model, token cap | Fully fixed | Test pass rate | Truncated | Median reply tokens |
| --- | --- | --- | --- | --- |
| Base, 8k | 51.0% | 59.8% | 29.0% | 3,140 |
| Tuned, 8k | 59.3% | 79.3% | 0.2% | 603 |
| Base, 32k | 62.5% | 80.2% | 2.2% | 3,713 |

At the 8k training cap, a quarter of the base model's samples ran out of
tokens and scored 0, so the cheapest way to more reward was to reason less.
The tuned model answers in about a fifth of the tokens and no longer
truncates, which lifts fully fixed by 8 points at 8k. Its repair skill did not
improve: against the base model at 32k it is 3.8 points lower on the same
problems (95% CI −8.2 to +0.4). To train for repair skill rather than brevity,
raise `--max-completion-tokens` until the base model rarely truncates, and
measure the baseline at the same cap.
