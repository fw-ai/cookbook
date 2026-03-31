#!/usr/bin/env bash
set -euo pipefail

# Multi-hop QA IGPO training example.
#
# Prerequisites:
#   pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook eval-protocol datasets
#   export FIREWORKS_API_KEY=...
#
# Step 1: Prepare dataset (downloads HotpotQA from HuggingFace)
python prepare_data.py --max-rows 500
#
# Step 2: Train with IGPO
#   Replace TRAINING_SHAPE with your training shape ID.
#   Replace OUTPUT_MODEL_ID with your desired output model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/train_multihop_qa_igpo.py" \
    --base-model "accounts/fireworks/models/qwen3-8b" \
    --tokenizer-model "Qwen/Qwen3-8B" \
    --dataset-path "$SCRIPT_DIR/dataset.jsonl" \
    --training-shape "${TRAINING_SHAPE:?Set TRAINING_SHAPE}" \
    --output-model-id "${OUTPUT_MODEL_ID:?Set OUTPUT_MODEL_ID}" \
    --max-rows 200 \
    --max-steps 8 \
    --epochs 3 \
    --completions-per-prompt 4 \
    --prompt-groups-per-step 4 \
    --learning-rate 1e-5 \
    --gamma 1.0 \
    --ig-weight 0.1 \
    --scoring-workers 8 \
    --search-top-k 2 \
    --temperature 1.0 \
    --skip-ig-last-turn \
    "$@"
