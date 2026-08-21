#!/usr/bin/env bash
set -euo pipefail

# Run the Countdown serverless RL demo on Kimi K3 without provisioning a
# trainer, deployment, shape, or region.
#
#   FIREWORKS_API_KEY   Fireworks key with serverless-training access (required)
#   WANDB_API_KEY       Weights & Biases key (optional; enables W&B tracking)
#   WANDB_ENTITY        W&B entity/team (optional; required for W&B logging)

: "${FIREWORKS_API_KEY:?set a Fireworks key with serverless-training access}"

run_stamp="$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="${COUNTDOWN_RUN_DIR:-/tmp/countdown-k3-$run_stamp}"
python_bin="${PYTHON_BIN:-python}"
steps="${COUNTDOWN_STEPS:-20}"

cookbook_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$cookbook_root${PYTHONPATH:+:$PYTHONPATH}"
# Optional: prefer a local python-sdk checkout when developing against it.
if [[ -d "$cookbook_root/../python-sdk/src" ]]; then
  export PYTHONPATH="$cookbook_root/../python-sdk/src:$PYTHONPATH"
fi
export PYTHONUNBUFFERED=1
# The Kimi K3 tokenizer ships a custom TikTokenTokenizer (remote code).
export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"
mkdir -p "$run_dir"

# Materialize the TinyZero countdown dataset on first run.
dataset="$cookbook_root/training/examples/serverless_rl/data/countdown_3to4_train.jsonl"
if [[ ! -f "$dataset" ]]; then
  echo "dataset not found at $dataset -- preparing from Jiayi-Pan/Countdown-Tasks-3to4"
  "$python_bin" -m training.examples.serverless_rl.countdown_rl \
    --prepare-dataset --dataset "$dataset"
fi
sha256sum "$dataset" > "$run_dir/input_sha256.txt"

cmd=(
  "$python_bin" -u -m training.examples.serverless_rl.countdown_rl
  --base-model accounts/fireworks/models/kimi-k3
  --tokenizer-model "${KIMI_K3_TOKENIZER:-moonshotai/Kimi-K3}"
  --dataset "$dataset"
  --steps "$steps"
  --checkpoint-name "countdown-k3-$run_stamp"
  --final-checkpoint-name "countdown-k3-final-$run_stamp"
  --run-dir "$run_dir"
)

# W&B: enabled when both the entity and key are present.
if [[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_API_KEY:-}" ]]; then
  cmd+=(
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "${WANDB_PROJECT:-serverless-rl-countdown}"
    --wandb-run-name "countdown-k3-$run_stamp"
  )
fi

printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
"${cmd[@]}" 2>&1 | tee "$run_dir/run.log"

echo "run_dir=$run_dir"
