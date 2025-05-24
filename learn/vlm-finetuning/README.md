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

For this tutorial, we'll be using a sample synthetic dataset [sample_data/train.jsonl](sample_data/train.jsonl) dataset. It contains 50 rows, of images of food (specified by path) and contains assistant responses that reason in `<think>...</think>` tags before classifying them. These responses were generated from Qwen 2.5 VL 32B Instruct.

#### Common issues:

- Messages with `{"content": "Regular text here"}` are not supported. This should be instead `{"content": [{"type": "text", "text": "Regular text here"}]}`. Otherwise you will get error:
```
pyarrow.lib.ArrowInvalid: JSON parse error: Column(/messages/[]/content) changed from string to array in row 0
```
- If using relative image path, paths should be relative to where axolotl is called from. E.g., for path `image_dir/image.jpg` if you are in `vlm-finetuning` directory, then the path to the image should be `vlm-finetuning/image_dir/image.jpg`.

```bash
axolotl train 2p5-7b.yaml

# See here for the config files:
# https://docs.axolotl.ai/docs/config.html

# See here for dataset format:
# https://docs.axolotl.ai/docs/multimodal.html
```