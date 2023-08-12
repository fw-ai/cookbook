# Instruction tuning with LoRA

## Setup

Install the required dependencies or build the [provided Docker image](https://github.com/fw-ai-external/cookbook/tree/main/recipes/docker/text).
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

The [`summarize`](https://github.com/fw-ai-external/cookbook/blob/main/recipes/tune/instruct_lora/conf/summarize.yaml)
config in the above command defines the tuning job. It controls different aspects of the
tuning process such as training data, base model, prompt format, and trainer flags.
Feel free to add a new config based on your custom needs.
