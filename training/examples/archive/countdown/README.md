# Countdown — serverless GRPO

> **Deprecated (archived).** Covered by [`rl/deepmath/`](../../rl/deepmath/) (`train_deepmath_serverless.py`). See [`../README.md`](../README.md).

GRPO on Fireworks **serverless training**: no trainer job, no inference
deployment, no training shape or region. Each example is task code (dataset,
prompt, reward, rollout) handed to the serverless async-RL recipe,
[`recipes/experiment/async_rl_loop_serverless.py`](../../../recipes/experiment/async_rl_loop_serverless.py).

The recipe owns the loop:

- connects to a pooled LoRA trainer (`create_lora_training_client`) and, after
  every optimizer step, publishes a sampler snapshot
  (`save_weights_for_sampler`) that the next rollouts sample from;
- fans each dataset row out to `completions_per_prompt` rollouts, runs them
  asynchronously (`max_concurrency_rollout_sample` in flight), and assembles
  prompt groups with group-relative advantages;
- trains with the trainer's built-in GRPO loss (`"ppo"`), clipped against the
  rollout logprobs, and replays sampled expert routes (R3) for MoE models;
- logs step metrics to `metrics_file` and W&B, runs the example's held-out
  evaluation at the start, every `--evaluation-interval` steps and at the end,
  and saves resumable training state (`--dcp-save-interval`, plus a final save).

The same rollout code runs on a dedicated trainer + deployment through
[`recipes/async_rl_loop.py`](../../../recipes/async_rl_loop.py).

## Files

| File | What it is |
| --- | --- |
| `countdown_rl.py` | Countdown task (rollout, held-out eval) + `async_rl_loop_serverless.main`. Defaults target Kimi K3. |
| `countdown_rewards.py` | Countdown reward (`composite_reward`): format, number usage, correctness. |
| `prepare_data.py` | Downloads the TinyZero Countdown tasks to `data/countdown_3to4_train.jsonl`. |
| `visual_toolbench_rl.py` | VisualToolBench task package ([`../../rl/visual_toolbench/`](../../rl/visual_toolbench/)) + `async_rl_loop_serverless.main`. |
| `runs/run_countdown_k3_serverless.sh` | Countdown on Kimi K3. |
| `runs/run_countdown_dsv4_flash_serverless.sh` | Countdown on DeepSeek V4 Flash 0731 with DCP saves and resume. |
| `runs/run_{qwen3p6,kimi_k3,muse_glimmer}_serverless_2epoch.sh` | VisualToolBench two-epoch runs (see the [VisualToolBench README](../../rl/visual_toolbench/README.md)). |
| `data/countdown_train.jsonl` | 32-row Countdown sample for eyeballing the schema. |

## Countdown

Each rollout renders the row's chat `messages`, samples one completion and
scores it with `composite_reward` (partial credit for a parseable `<answer>`,
using exactly the given numbers, and hitting the target). Groups whose samples
all score the same carry no GRPO signal and are skipped. The first
`--eval-rows` rows of the dataset are held out and scored with
`completions_per_prompt` samples each; `eval/raw_reward` and `eval/accuracy`
are the learning curve.

Prepare the [TinyZero Countdown tasks](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
once (the launchers do this automatically):

```bash
python -m training.examples.archive.countdown.prepare_data   # -> data/countdown_3to4_train.jsonl
```

Rows are `{"messages": [...], "ground_truth": "{\"numbers\": [...], \"target\": N}"}`.
The file is shuffled when written, so training reads it in order.

Run on Kimi K3 (W&B is optional):

```bash
export FIREWORKS_API_KEY=fw_...
export WANDB_ENTITY=<your-wandb-entity> WANDB_API_KEY=...   # optional
bash training/examples/archive/launchers/run_countdown_k3_serverless.sh
```

Or invoke it directly with any serverless model. The renderer is resolved from
the tokenizer; pass `--renderer-name` only to override it:

```bash
export HF_TRUST_REMOTE_CODE=1             # Kimi K3 ships a custom tokenizer
python -m training.examples.archive.countdown.countdown_rl \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --max-rows 320 --prompt-groups-per-step 16 --completions-per-prompt 8 \
  --run-dir ./countdown-k3
```

`--max-rows` training prompts at `--prompt-groups-per-step` per step set the
number of optimizer steps (skipped groups can make a run end sooner).
`FIREWORKS_BASE_URL` overrides the API endpoint.

The run directory holds `metrics.jsonl`, `result.json` and `reward_curve.png`.
Set `COOKBOOK_TRACE_FILE` for a Perfetto trace of the client phases, or run
`python -m training.examples.tools.client_phase_trace_demo` for a bounded
one-step traced run.

### Checkpoints and resume

`--dcp-save-interval N` saves adapter + optimizer state every N steps, and the
recipe saves once more at the end. `result.json` records
`final_training_checkpoint`, `steps` and `resolved_rows`; resume a new process
from them with the same dataset and a larger `--max-rows`:

```bash
python -m training.examples.archive.countdown.countdown_rl \
  --init-from-checkpoint <final_training_checkpoint> \
  --step-offset <steps> --resolved-rows-offset <resolved_rows> \
  --max-rows <resolved_rows + new rows> --dcp-save-interval 5
```

The DSV4 launcher wraps this: set `COUNTDOWN_RESUME_FROM`,
`COUNTDOWN_STEP_OFFSET` and `COUNTDOWN_ROWS_OFFSET` from the prior
`result.json`, and `COUNTDOWN_STEPS` to the number of additional steps.

`result.json` also records `final_sampler_snapshot` and `training_session`.
Promotion is not part of this example; promote a session checkpoint with the
SDK's `FireworksClient.promote_session_checkpoint` while the session is still
available.

## VisualToolBench

`visual_toolbench_rl.py` trains a vision-language policy on multi-turn image
tool use (crop / zoom / rotate / adjust) graded by a rubric LLM judge. The task
code lives in [`../../rl/visual_toolbench/`](../../rl/visual_toolbench/README.md),
which also has the dedicated-trainer entry point. Data format, launchers and
model settings are documented there.

## Notes

- **Pool capacity.** If the pooled trainer for `base_model` is full, session
  creation fails with an out-of-capacity error; retry or use a dedicated
  trainer.
- **LoRA only.** Serverless training requires `lora_rank > 0`.
- **Matching tokenizer.** `--tokenizer-model` must match `--base-model`; the
  tokenizer renders prompts and decodes sampled tokens client-side.
- **Cost.** Serverless bills per token. Lower `--max-rows`,
  `--completions-per-prompt` and `--max-completion-tokens` for a cheap smoke
  run.

Other objectives (DAPO, GSPO, CISPO, …) live in
[`utils/rl/algorithm/`](../../../utils/rl/algorithm/); fork the recipe's
`train_chunk` to swap one in.
