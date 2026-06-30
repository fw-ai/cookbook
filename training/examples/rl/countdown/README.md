# Countdown RL example

This example runs async GRPO on a small Countdown arithmetic task. It supports:

- `--variant text`: numbers are provided in the text prompt.
- `--variant vision`: numbers are rendered into a PNG and passed as an image.

The vision variant exercises the public multimodal RL path used for Qwen VL
fine tuning: renderer-built image prompts, `/v1/completions` image sampling with
token IDs and logprobs, hot-load weight sync, and trainer-side updates.

## Manual run

```bash
cd training
export FIREWORKS_API_KEY=...
python -m pip install -e '.[dev]'
python examples/rl/countdown/train.py \
  --variant vision \
  --base-model accounts/fireworks/models/qwen3-vl-8b-instruct \
  --tokenizer-model Qwen/Qwen3-VL-8B-Instruct \
  --training-shape accounts/fireworks/trainingShapes/qwen3-vl-8b-256k-h200-lora \
  --deployment-shape accounts/fireworks/deploymentShapes/rft-qwen3-vl-8b-instruct \
  --lora-rank 16 \
  --max-rows 16 \
  --completions-per-prompt 4 \
  --prompt-groups-per-step 2
```

For GitHub Actions, use the **RL Countdown** workflow. It is manual-only and
requires `FIREWORKS_API_KEY`; W&B is optional.
