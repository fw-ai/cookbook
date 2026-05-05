#!/usr/bin/env bash
# Repro script for kevin-koehncke-vtvwg's silent train_loss=0.0000 + slowness.
#
# Mirrors the customer's hot config:
#   - LoRA rank 16
#   - grad_accum 16  (they recently bumped from 4 -> 16)
#   - lr 1e-4
#   - region US_VIRGINIA_1
#   - base model qwen3-vl-8b-du-0422 (full-FT'd by them on DU corpus)
#
# Replaces the customer's tinker_cookbook training script with the cookbook's
# standard sft_loop (forward_backward + cross-entropy on assistant tokens
# only). If we see non-zero loss => bug is in their custom training stack
# (loss fn, renderer, mask, or W&B logging). If we see loss=0 too => bug is
# trainer-side and we own the fix.
set -euo pipefail

HERE=$(dirname "$(realpath "$0")")
echo "$HERE"

export PYTHONPATH="${PYTHONPATH:-}:$HERE/../../../"
echo "$PYTHONPATH"

missing_vars=""
[ -z "${FIREWORKS_API_KEY:-}" ] && missing_vars="$missing_vars FIREWORKS_API_KEY"
if [ -n "$missing_vars" ]; then
    echo "Error: missing required env var(s):$missing_vars" >&2
    exit 1
fi

# 1. (Re)generate the synthetic DU-like VL dataset.
DATASET="$HERE/du_repro.jsonl"
if [ ! -f "$DATASET" ] || [ "${REGEN_DATASET:-0}" = "1" ]; then
    echo "Generating $DATASET ..."
    python3.12 "$HERE/make_du_repro_dataset.py" \
        --num-examples 64 \
        --output "$DATASET"
fi

# 2. Launch SFT with cookbook's standard loss path.
RUN_NAME="du-repro-$(date +%Y%m%d%H%M)"
echo "Launching $RUN_NAME ..."

python3.12 "$HERE/train_sft.py" \
    --base-model accounts/kevin-koehncke-vtvwg/models/qwen3-vl-8b-du-0422 \
    --tokenizer-model /tmp/qwen3-vl-8b-du-0422/hf \
    --renderer-name qwen3_vl_instruct \
    --dataset-path "$DATASET" \
    --region US_VIRGINIA_1 \
    --max-examples 64 \
    --lora-rank 16 \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 16 \
    --learning-rate 1e-4 \
    --grad-clip-norm 1.0 \
    --wandb-project du-repro \
    --wandb-run-name "$RUN_NAME" \
    --output-model-id "$RUN_NAME" \
    2>&1 | tee "$HERE/${RUN_NAME}.log"
