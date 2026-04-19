HERE=$(dirname $(realpath $0))
echo $HERE

export PYTHONPATH=$PYTHONPATH:$HERE/../../../
echo $PYTHONPATH

missing_vars=""
[ -z "${FIREWORKS_API_KEY:-}" ] && missing_vars="$missing_vars FIREWORKS_API_KEY"

if [ -n "$missing_vars" ]; then
    echo "Error: missing required env var(s):$missing_vars" >&2
    exit 1
fi

python3.12 train_sft.py \
    --base-model accounts/fireworks/models/qwen3-vl-8b-instruct \
    --tokenizer-model Qwen/Qwen3-VL-8B-Instruct \
    --dataset-path food_reasoning.jsonl \
    --region "" \
    --max-examples 50 \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --learning-rate 1e-5 \
    --output-model-id sft-vl-qwen3-8b-$(date +%Y%m%d%H%M)
