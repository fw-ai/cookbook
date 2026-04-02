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

python train_sft.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --tokenizer-model Qwen/Qwen3-8B \
    --dataset-path text2sql_dataset.jsonl \
    --region US_VIRGINIA_1 \
    --max-examples 100 \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --output-model-id sft-text-qwen3-8b-$(date +%Y%m%d%H%M)
