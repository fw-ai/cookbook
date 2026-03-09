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

python train_ifeval_orpo.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --tokenizer-model Qwen/Qwen3-8B \
    --dataset-path dataset.jsonl \
    --training-shape qwen3-8b-128k-h200 \
    --region US_VIRGINIA_1 \
    --max-pairs 200 \
    --epochs 1 \
    --grad-accum 4 \
    --learning-rate 1e-5 \
    --orpo-lambda 1.0
