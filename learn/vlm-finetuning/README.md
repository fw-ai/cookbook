# Full Finetuning Qwen 2.5 7B and deploying on Fireworks

### Pre-requisites:
- [miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install)
  - Or anyway to create a python 3.10 environment
- git
- 1 8xH100 node (1 8xA100 should should work too)

### Setup

```bash
git clone -b aidand-vlm-full-finetune git@github.com:aidando73/cookbook.git
# Or if using https:
git clone -b aidand-vlm-full-finetune https://github.com/aidando73/cookbook.git

cd cookbook/learn/vlm-finetuning

# Create environment
conda create --name axolotl-env python=3.10 -y
conda activate axolotl-env

# Install dependencies
pip install uv
uv pip install -U packaging==23.2 setuptools==75.8.0 wheel==0.45.1 ninja==1.11.1.4 requests==2.32.3 "huggingface-hub[cli]==0.31.0"
uv pip install --no-build-isolation "axolotl[flash-attn,deepspeed]==0.9.2"
```

We'll be using `axolotl` to finetune the model. I've tried finetuning with `trl`, but found that support for liger-kernel wasn't working and it adds a few breaking changes to config files.

To learn more about `axolotl`, see the [docs](https://docs.axolotl.ai/).

```bash
# Fetch deepspeed configs and examples
axolotl fetch deepspeed_configs 
axolotl fetch examples
```

### Formatting your dataset

Dataset should be in a .jsonl format similar to (but not exactly the same as) OpenAI chat format. E.g.,

```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "What's in these two images?"}, {"type": "image", "base64": "data:image/jpeg;base64,..."}, {"type": "image", "path": "path/to/image/relative/to/where/command/is/being/executed.jpg"}]}, {"role": "assistant", "content": [{"type": "text", "text": "There are two images of a cat and a dog."}]}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image", "url": "https://example.com/cat.jpg"}]}, {"role": "assistant", "content": [{"type": "text", "text": "There is a cat in the image."}]}]}
```

Reference the [axolotl multimodal docs](https://docs.axolotl.ai/docs/multimodal.html#dataset-format) for more details.

For this tutorial, we'll be using a sample synthetic dataset [sample_data/train.jsonl](sample_data/train.jsonl) dataset. It contains 50 rows, of images of food (specified by path) and contains assistant responses that reason in `<think>...</think>` tags before classifying them. These responses were generated from Qwen 2.5 VL 32B Instruct. Images were downloaded from https://huggingface.co/datasets/ethz/food101.

#### Common issues:

- Messages with `{"content": "Regular text here"}` are not supported. This should be instead `{"content": [{"type": "text", "text": "Regular text here"}]}`. Otherwise you will get error:
```
pyarrow.lib.ArrowInvalid: JSON parse error: Column(/messages/[]/content) changed from string to array in row 0
```
- If using relative image path, paths should be relative to where axolotl is called from. E.g., for path `image_dir/image.jpg` if you are in `vlm-finetuning` directory, then the path to the image should be `vlm-finetuning/image_dir/image.jpg`.

### Training

An already prepared axolotl config file is provided in [2p5-7b.yaml](2p5-7b.yaml).

To finetune, run:

```bash
axolotl train 2p5-7b.yaml
```

The final model will be saved in `outputs/out` and checkpoints will be saved in `outputs/out/checkpoint-<step>`.

### Deploying on Fireworks

**Pre-requisite:**
- Install [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl)

### Deployment

1. This requires a few variables to be set so we'll keep them in a file called `deploy_vars.sh` and load them when we run commands. First create this file:

```bash
cp deploy_vars.sh.example deploy_vars.sh
```

2. Next open `deploy_vars.sh` and add your account ID and API key.
3. Validate `ACCOUNT_ID` and `FIREWORKS_API_KEY` are correct.

```bash
source deploy_vars.sh && firectl -a $ACCOUNT_ID list models --api-key $FIREWORKS_API_KEY # You should see either an empty list or all your current models
ls $CHECKPOINT/config.json $CHECKPOINT/model-*-of-*.safetensors # You should see config.json and *.safetensors files
```

4. Next we create the model:

```bash
# For some reason axolotl checkpoints are missing a few config files
# So download them from the base model (we exclude weights)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir $CHECKPOINT --exclude "*.safetensors" "model.safetensors.index.json"

# Load variables before running firectl commands
source deploy_vars.sh && firectl -a $ACCOUNT_ID create model $MODEL_NAME $CHECKPOINT --api-key $FIREWORKS_API_KEY
```

Next we create the deployment:

```bash
source deploy_vars.sh && firectl -a $ACCOUNT_ID create deployment accounts/$ACCOUNT_ID/models/$MODEL_NAME \
  --accelerator-type="NVIDIA_H100_80GB" \
  --min-replica-count 1 \
  --accelerator-count 1 \
  --api-key $FIREWORKS_API_KEY \
  --deployment-id $MODEL_NAME # We set the deployment ID to the model name
```

Wait until the deployment is ready.

```bash
watch -c "firectl -a $ACCOUNT_ID list deployments --order-by='create_time desc' --api-key $FIREWORKS_API_KEY"
```

Then you can test the deployment:

```bash
source deploy_vars.sh && python fw_req.py --model accounts/$ACCOUNT_ID/models/$MODEL_NAME#accounts/$ACCOUNT_ID/deployments/$MODEL_NAME --api-key $FIREWORKS_API_KEY
```

### Thanks for trying Fireworks

Please email me at aidan@fireworks.ai if you have any questions/feedback. Or [drop something in my calendar](https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ2iKVtCNOXAOLoYRcGh4ppHL_ztUU-osdlrAeR8dyvoZY2V-pMMMu_ozOjvTVeLg65Erkuu0UET).