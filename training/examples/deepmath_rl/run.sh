HERE=$(dirname $(realpath $0))
echo $HERE

export PYTHONPATH=$PYTHONPATH:$HERE/../../../../../
echo $PYTHONPATH
python train_deepmath.py \
    --base-model accounts/fireworks/models/qwen3-4b \
    --tokenizer-model Qwen/Qwen3-4b \
    --dataset-path dataset.jsonl \
    --training-shape acounts/fireworks/trainingShapes/qwen3-4b-minimum-h200 \
    --ref-training-shape accounts/fireworks/trainingShapes/qwen3-4b-minimum-h200-forward \
    --deployment-id deepmath-qwen3-4b-$(date +%s) \
    --region US_VIRGINIA_1 \
    --max-rows 500 \
    --epochs 3 \
    --completions-per-prompt 8 \
    --learning-rate 1e-5 \
    --kl-beta 0.001 \
    --output-model-id deepmath-qwen3-4b-$(date +%s)
