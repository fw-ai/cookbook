# Instruction tuning with LoRA

## Setup

Install the required dependencies or build the [provided Docker image](https://github.com/fw-ai/cookbook/tree/main/recipes/docker/text).
```bash
# assuming that cookbook repo was checked out under /workspace
export PYTHONPATH=.:/workspace/cookbook
```

## Example command
```bash
# run with 8 GPUs - adjust based on your hardware setup
torchx run -s local_cwd dist.ddp -j 1x8 --script finetune.py -- \
  --config-name=summarize
```

The [`summarize`](https://github.com/fw-ai/cookbook/blob/main/recipes/tune/instruct_lora/conf/summarize.yaml)
config in the above command defines the tuning job. It controls different aspects of the
tuning process such as training data, base model, prompt format, and trainer flags.
Feel free to add a new config based on your custom needs.

## Running in Colab

Fine tuning can be done in a [Colab notebook](https://colab.research.google.com/). Some recipes
run in the free tier although keep in mind that any serious fine tuning of reasonably sized
models on real-world datasets may require an upgrade to a more powerful hardware to be
practical. [Here is a sample notebook](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/recipes/tune/instruct_lora/colabtune.ipynb) that you can play with.

## Running in local Jupyter

See the [following instructions](https://github.com/fw-ai/cookbook/tree/main/recipes/docker/text#jupyter-notebooks) explaining how to run the notebook inside a Docker container.
