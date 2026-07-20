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

# output-mode: embedding | cos_similarity_matrix | contrastive_loss
python train_embedding.py \
    --base-model accounts/fireworks/models/qwen3-embedding-8b \
    --tokenizer-model Qwen/Qwen3-Embedding-8B \
    --dataset-path retrieval_pairs.jsonl \
    --output-mode embedding \
    --epochs 3 \
    --batch-size 8 \
    --temperature 0.02 \
    --learning-rate 1e-5
