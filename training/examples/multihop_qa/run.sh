#!/usr/bin/env bash
set -euo pipefail

# Multi-hop QA async RL training (with optional IGPO turn-level scoring).
#
# Prerequisites:
#   Follow the setup instructions in ../../README.md.
#   export FIREWORKS_API_KEY=...
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Prepare dataset (downloads HotpotQA hard questions from HuggingFace)
python "$SCRIPT_DIR/prepare_data.py" --max-rows 2000 --difficulty hard
#
# Step 2: Train via async_rl_loop with IGPO turn-level scoring folded in
#   Replace TRAINING_SHAPE_ID with your training shape resource name.
#   Replace OUTPUT_MODEL_ID with your desired output model.

python "$SCRIPT_DIR/train.py" \
    --base-model "accounts/fireworks/models/qwen3-8b" \
    --tokenizer-model "Qwen/Qwen3-8B" \
    --dataset-path "$SCRIPT_DIR/dataset.jsonl" \
    --training-shape-id "${TRAINING_SHAPE_ID:?Set TRAINING_SHAPE_ID}" \
    --output-model-id "${OUTPUT_MODEL_ID:?Set OUTPUT_MODEL_ID}" \
    --max-rows 2000 \
    --epochs 3 \
    --completions-per-prompt 4 \
    --prompt-groups-per-step 4 \
    --learning-rate 1e-5 \
    --kl-beta 0.0 \
    --lora-rank 16 \
    --max-head-offpolicy-versions 1 \
    --ig-weight 1.0 \
    --scoring-workers 8 \
    --max-turns 8 \
    --search-top-k 2 \
    --temperature 1.0 \
    --skip-ig-last-turn \
    --filter-constant-reward \
    "$@"
