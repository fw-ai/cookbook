#!/usr/bin/env bash
set -euo pipefail

# Run the Countdown serverless RL demo on DeepSeek V4 Flash (0731) without
# provisioning a trainer, deployment, shape, or region.
#
#   FIREWORKS_API_KEY   Fireworks key with serverless-training access (required)
#   WANDB_API_KEY       Weights & Biases key (optional; enables W&B tracking)
#   WANDB_ENTITY        W&B entity/team (optional; required for W&B logging)
#   COUNTDOWN_WANDB_RUN_NAME
#                       exact W&B run name (optional)
#   COUNTDOWN_RESUME_FROM / COUNTDOWN_STEP_OFFSET / COUNTDOWN_ROWS_OFFSET
#                       resume from a prior run's result.json:
#                       final_training_checkpoint / steps / resolved_rows (optional)
#   COUNTDOWN_DATASET   custom prepared JSONL path (optional; default auto-downloads)
#   COUNTDOWN_EVAL_PROMPT_GROUPS / COUNTDOWN_EVAL_INTERVAL
#                       fixed held-out evaluation controls (optional)

: "${FIREWORKS_API_KEY:?set a Fireworks key with serverless-training access}"

run_stamp="$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="${COUNTDOWN_RUN_DIR:-/tmp/countdown-dsv4-flash-$run_stamp}"
python_bin="${PYTHON_BIN:-python}"
steps="${COUNTDOWN_STEPS:-30}"
prompt_groups_per_step="${COUNTDOWN_PROMPT_GROUPS_PER_STEP:-16}"
group_size="${COUNTDOWN_GROUP_SIZE:-8}"
prompt_concurrency="${COUNTDOWN_PROMPT_CONCURRENCY:-8}"
dcp_save_interval="${COUNTDOWN_DCP_SAVE_INTERVAL:-5}"

cookbook_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$cookbook_root${PYTHONPATH:+:$PYTHONPATH}"
# Optional: prefer a local python-sdk checkout when developing against it.
if [[ -d "$cookbook_root/../python-sdk/src" ]]; then
  export PYTHONPATH="$cookbook_root/../python-sdk/src:$PYTHONPATH"
fi
export PYTHONUNBUFFERED=1
mkdir -p "$run_dir"

# Materialize the TinyZero countdown dataset on first run.
dataset="${COUNTDOWN_DATASET:-$cookbook_root/training/examples/archive/countdown/data/countdown_3to4_train.jsonl}"
if [[ ! -f "$dataset" ]]; then
  echo "dataset not found at $dataset -- preparing from Jiayi-Pan/Countdown-Tasks-3to4"
  "$python_bin" -m training.examples.archive.countdown.prepare_data --output "$dataset"
fi
sha256sum "$dataset" > "$run_dir/input_sha256.txt"

# The recipe derives its steps from the row count: COUNTDOWN_STEPS more steps
# of prompt_groups_per_step rows after any resumed rows. Groups whose rollouts
# all score the same are skipped, so a run can finish in fewer steps.
cmd=(
  "$python_bin" -u -m training.examples.archive.countdown.countdown_rl
  --base-model "${DSV4_BASE_MODEL:-accounts/fireworks/models/deepseek-v4-flash-0731}"
  --tokenizer-model "${DSV4_TOKENIZER:-deepseek-ai/DeepSeek-V4-Flash-0731}"
  --dataset "$dataset"
  --max-rows "$((${COUNTDOWN_ROWS_OFFSET:-0} + steps * prompt_groups_per_step))"
  --lora-rank "${COUNTDOWN_LORA_RANK:-32}"
  --lora-alpha "${COUNTDOWN_LORA_ALPHA:-32}"
  --max-seq-len "${COUNTDOWN_MAX_SEQ_LEN:-32768}"
  --prompt-groups-per-step "$prompt_groups_per_step"
  --completions-per-prompt "$group_size"
  --max-concurrency-rollout-sample "$((prompt_concurrency * group_size))"
  --max-completion-tokens "${COUNTDOWN_MAX_SAMPLE_TOKENS:-32500}"
  --eval-rows "${COUNTDOWN_EVAL_PROMPT_GROUPS:-16}"
  --evaluation-interval "${COUNTDOWN_EVAL_INTERVAL:-5}"
  --sample-timeout "${COUNTDOWN_SAMPLING_TIMEOUT_S:-3600}"
  --learning-rate "${COUNTDOWN_LEARNING_RATE:-1e-4}"
  --snapshot-prefix "countdown-dsv4-$run_stamp"
  --dcp-save-interval "$dcp_save_interval"
  --run-dir "$run_dir"
)

if [[ -n "${COUNTDOWN_RESUME_FROM:-}" ]]; then
  cmd+=(
    --init-from-checkpoint "$COUNTDOWN_RESUME_FROM"
    --step-offset "${COUNTDOWN_STEP_OFFSET:?set the steps value from the prior result.json}"
    --resolved-rows-offset "${COUNTDOWN_ROWS_OFFSET:?set the resolved_rows value from the prior result.json}"
  )
fi

# W&B: enabled when both the entity and key are present.
if [[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_API_KEY:-}" ]]; then
  cmd+=(
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "${WANDB_PROJECT:-serverless-rl-countdown}"
    --wandb-run-name "${COUNTDOWN_WANDB_RUN_NAME:-countdown-dsv4-flash-$run_stamp}"
  )
fi

printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
"${cmd[@]}" 2>&1 | tee "$run_dir/run.log"

echo "run_dir=$run_dir"
