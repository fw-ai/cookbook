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
#   COUNTDOWN_RESUME_FROM
#                       prior <account>/<run-id>/<training-checkpoint> (optional)
#   COUNTDOWN_OUTPUT_MODEL_ID
#                       promote the final sampler checkpoint to this model id
#                       (optional; empty means save without promotion)
#   COUNTDOWN_DATASET   custom prepared JSONL path (optional; default auto-downloads)

: "${FIREWORKS_API_KEY:?set a Fireworks key with serverless-training access}"

run_stamp="$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="${COUNTDOWN_RUN_DIR:-/tmp/countdown-dsv4-flash-$run_stamp}"
python_bin="${PYTHON_BIN:-python}"
steps="${COUNTDOWN_STEPS:-20}"
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
dataset="${COUNTDOWN_DATASET:-$cookbook_root/training/examples/serverless_rl/data/countdown_3to4_train.jsonl}"
if [[ ! -f "$dataset" ]]; then
  echo "dataset not found at $dataset -- preparing from Jiayi-Pan/Countdown-Tasks-3to4"
  "$python_bin" -m training.examples.serverless_rl.countdown_rl \
    --prepare-dataset --dataset "$dataset"
fi
sha256sum "$dataset" > "$run_dir/input_sha256.txt"

cmd=(
  "$python_bin" -u -m training.examples.serverless_rl.countdown_rl
  --base-model "${DSV4_BASE_MODEL:-accounts/fireworks/models/deepseek-v4-flash-0731}"
  --tokenizer-model "${DSV4_TOKENIZER:-deepseek-ai/DeepSeek-V4-Flash-0731}"
  --dataset "$dataset"
  --steps "$steps"
  --lora-rank "${COUNTDOWN_LORA_RANK:-32}"
  --lora-alpha "${COUNTDOWN_LORA_ALPHA:-64}"
  --max-seq-len "${COUNTDOWN_MAX_SEQ_LEN:-32768}"
  --prompt-groups-per-step "${COUNTDOWN_PROMPT_GROUPS_PER_STEP:-16}"
  --group-size "${COUNTDOWN_GROUP_SIZE:-8}"
  --prompt-concurrency "${COUNTDOWN_PROMPT_CONCURRENCY:-8}"
  --max-sample-tokens "${COUNTDOWN_MAX_SAMPLE_TOKENS:-4096}"
  --learning-rate "${COUNTDOWN_LEARNING_RATE:-1e-4}"
  --router-replay
  --checkpoint-name "cd-sample"
  --final-checkpoint-name "cd-final"
  --dcp-save-interval "$dcp_save_interval"
  --training-checkpoint-name "cd-state"
  --run-dir "$run_dir"
)

if [[ -n "${COUNTDOWN_RESUME_FROM:-}" ]]; then
  cmd+=(--resume-from "$COUNTDOWN_RESUME_FROM")
fi

if [[ -n "${COUNTDOWN_OUTPUT_MODEL_ID:-}" ]]; then
  cmd+=(--output-model-id "$COUNTDOWN_OUTPUT_MODEL_ID")
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
