HERE=$(dirname $(realpath $0))
echo $HERE

export PYTHONPATH=$PYTHONPATH:$HERE/../../../
echo $PYTHONPATH

missing_vars=""
[ -z "${FIREWORKS_API_KEY:-}" ] && missing_vars="$missing_vars FIREWORKS_API_KEY"
[ -z "${FIREWORKS_ACCOUNT_ID:-}" ] && missing_vars="$missing_vars FIREWORKS_ACCOUNT_ID"

if [ -n "$missing_vars" ]; then
    echo "Error: missing required env var(s):$missing_vars" >&2
    exit 1
fi

python train_sft.py \
    --base-model accounts/fireworks/models/qwen3-32b \
    --tokenizer-model Qwen/Qwen3-32B \
    --dataset-path text2sql_dataset.jsonl \
    --training-shape accounts/fireworks/trainingShapes/qwen3-32b-65k-b200 \
    --region US_OHIO_1 \
    --max-examples 100 \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 4 \
    --learning-rate 1e-5 
