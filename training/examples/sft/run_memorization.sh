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

python train_sft_memorization.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --tokenizer-model Qwen/Qwen3-8B \
    --region US_VIRGINIA_1 \
    --num-copies 16 \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --output-model-id sft-memorization-qwen3-8b-$(date +%Y%m%d%H%M)
