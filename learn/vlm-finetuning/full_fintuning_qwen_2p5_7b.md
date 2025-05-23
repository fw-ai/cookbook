## Finetuning Qwen 2.5 VL 7B Instruct and Deploying to Fireworks

### Pre-requisites:
- [miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install)
  - Or anyway to create a python 3.10 environment
- 1 8xH100 node (1 8xA100 should should work too)

### Setup

```bash
# TODO(aidan) change to FW AI main
# Clone this repository E.g.,
git clone -b aidan-finetunine-guide-1 https://github.com/aidando73/cookbook.git
# Or
git clone -b aidan-finetunine-guide-1 git@github.com:aidando73/cookbook.git

cd cookbook/learn/vlm-finetuning

conda create --name vlm-finetune-env python=3.10 -y
conda activate vlm-finetune-env
pip install uv
uv pip install trl==0.17.0 pillow==10.4.0 torchvision==0.21.0 deepspeed==0.16.8
```

### Prepare dataset

Dataset should be in a .jsonl standard OpenAI chat format. E.g.,

```jsonl
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What's in these two images?"}, {"type": "image", "image": "data:image/jpeg;base64,..."}, {"type": "image", "image": "data:image/jpeg;base64,..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "There are two images of a cat and a dog."}]}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image", "image": "data:image/jpeg;base64,..."}]}, {"role": "assistant", "content": [{"type": "text", "text": "There is a cat in the image."}]}]}
```

Where `image` can be:
- A base64 encoded image: `data:image/jpeg;base64,...`/`data:image/png;base64,...`
- A relative path to an image file: `path/to/image.jpg` (relative to the dataset directory)
- An absolute path to an image file: `/path/to/image.jpg`

For this example, we'll use a synthetic dataset `train_sample.jsonl` file in the current directory. It contains 50 rows, of images of food (base64 encoded) and contains assistant responses that reason in `<think>...</think>` tags before classifying them. These responses were generated from Qwen 2.5 VL 32B Instruct.

### Running training
```bash
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    sft_vlm.py \
    --dataset_path train_sample.jsonl \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --output_dir sft-qwen2p5-vl-7b-instruct-$(date +%Y-%m-%d_%H-%M) \
    --bf16 \
    --torch_dtype bfloat16 \
    --num_epochs 2 \
    --gradient_checkpointing
```

This will output the model and checkpoints to `sft-qwen2p5-vl-7b-instruct-{date}`.

### Useful flags
- `--save_total_limit 5` will save only the last 5 checkpoints.
  - This is useful to avoid running out of disk space. By default trl will save every 500 steps.
  - You can set `--save_steps 2000` to save a checkpoint every 2000 steps.
  - Another option is `--save_strategy epoch` which will save a checkpoint every epoch.
- `--report_to wandb` will log metrics to wandb.
  - You will need to follow these instructions to login beforehand: https://docs.wandb.ai/quickstart
- `--logging_steps 100` will log metrics every 100 steps. (By default it's 500)
- `--gradient_accumulation_steps 2` will accumulate gradients every 2 steps. You can increase this to increase the effective batch size.
  - The current example is 1 batch per device with gradient accumulation at 2, so size 16 batches
    - (1 batch per device * 2 gradient accumulation steps * 8 devices).

See https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments for all accepted arguments.

### Deploying to fireworks

Pre-requisite:
- Install [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl)
- Login to fireworks: `firectl set-api-key YOUR_API_KEY`

```bash
# You can also create a model from a checkpoint
firectl create model sft-qwen2p5-vl-7b-instruct sft-qwen2p5-vl-7b-instruct-{REPLACE_WITH_CHECKPOINT_DATETIME}/checkpoint-500 --use-hf-apply-chat-template

# Create a 1 GPU deployment with the new model
firectl-admin create deployment accounts/{YOUR_ACCOUNT_ID}/models/sft-qwen2p5-vl-7b-instruct \
  --accelerator-type="NVIDIA_H100_80GB" \
  --min-replica-count 1 \
  --accelerator-count 1


firectl get model accounts/{YOUR_ACCOUNT_ID}/models/sft-qwen2p5-vl-7b-instruct
```

### Thanks for trying Fireworks

Please email me at aidan@fireworks.ai if you have any questions/feedback. Or file an issue and tag me @aidando73.
