#!/usr/bin/env bash
set -euo pipefail

# Run the Muse Glimmer 30B pooled VTB example without provisioning a trainer,
# deployment, shape, or region.
: "${FIREWORKS_API_KEY:?set a Fireworks key with serverless-training access}"
: "${VTB_DATASET:?set VTB_DATASET to the 214-row aligned training JSONL}"
: "${VTB_EVAL_DATASET:?set VTB_EVAL_DATASET to the 50-row held-out JSONL}"

run_stamp="$(date -u +%Y%m%dt%H%M%Sz)-$$"
run_dir="${VTB_RUN_DIR:-/tmp/vtb-muse-glimmer-$run_stamp}"
tokenizer_model="${MUSE_GLIMMER_TOKENIZER:-meta-models/Muse-Glimmer-30B}"
python_bin="${PYTHON_BIN:-python}"

cookbook_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$cookbook_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$run_dir"

sha256sum "$VTB_DATASET" "$VTB_EVAL_DATASET" > "$run_dir/input_sha256.txt"

# 214 rows x 2 epochs = 428 prompt groups: 26 full 16-group steps and one
# final 12-group step. Each prompt group contains eight rollouts.
cmd=(
  "$python_bin" -u -m training.examples.serverless_rl.visual_toolbench_rl
  --dataset "$VTB_DATASET"
  --eval-dataset "$VTB_EVAL_DATASET"
  --base-model accounts/fireworks/models/muse-glimmer-30b
  --tokenizer-model "$tokenizer_model"
  --renderer-name muse_glimmer
  --steps 27
  --prompt-groups-per-step 16
  --group-size 8
  --rollout-concurrency 16
  --max-completion-tokens 32768
  --eval-max-completion-tokens 26666
  --no-require-complete-eval
  --max-turns 6
  --max-prompt-tokens 57344
  --max-workspace-images 6
  --tool-image-dim 1024
  --max-seq-len 131072
  --lora-rank 64
  --learning-rate 1e-4
  --adam-beta2 0.95
  --adam-eps 1e-12
  --adam-weight-decay 0
  --temperature 1.0
  --epochs 2
  --eval-interval 5
  --eval-upfront
  --eval-at-end
  --eval-group-size 1
  --eval-temperature 1.0
  --eval-top-p 0.95
  --eval-top-k 20
  --seed 0
  --shuffle
  --filter-constant-reward
  --filter-truncated-rollouts
  --judge-model accounts/fireworks/models/kimi-k3
  --judge-max-tokens 65536
  --judge-max-concurrency 32
  --judge-timeout-s 900
  --critical-reward-weight 0.2
  --sampling-timeout-s 1800
  --no-router-replay
  --no-router-replay-completion-only
  --grad-accumulation-normalization num_loss_tokens
  --require-tool-aligned-data
  --checkpoint-name "vtb-muse-$run_stamp"
  --final-checkpoint-name "vtb-muse-final-$run_stamp"
  --dcp-save-interval 2
  --run-dir "$run_dir"
  --plot-reward-curve
)

printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
"${cmd[@]}" 2>&1 | tee "$run_dir/run.log"

echo "run_dir=$run_dir"
