# Model Fine-Tuning

## Introduction

Model fine-tuning offers an efficient approach to tailor a pre-existing model to a
specific use case. This strategy saves computational resources and time by utilizing a
pre-trained model as a starting point and continues the training process with user-supplied
data.

There exist numerous fine-tuning methodologies. The most straightforward method utilizes
the base model as a checkpoint and restarts the training process from this point. However,
this approach may be computationally expensive and might require model sharding
(distributing the model across multiple ranks or hosts), especially when the base model
is large, which is often the case for Language Model Learning (LLM).

A more pragmatic fine-tuning procedure employs a [Parameter Efficient Algorithm](https://github.com/huggingface/peft)
(PEFT). PEFT significantly reduces the computational and memory cost by updating only a minor
subset of the model weights. The most widely used PEFT tuning technique is based on the
[Low Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) approach, which incorporates
trainable decomposition matrices on top of the frozen pre-trained base model transformer layers.

## Code Structure

This repository hosts a variety of fine-tuning recipes for different scenarios, each located
in its individual subdirectory. These recipes utilize the [Hydra](https://github.com/facebookresearch/hydra)
framework to manage various configuration options. These configurations allow you to
choose the appropriate base model and tuning dataset according to your requirements.

Feel free to modify or add configurations to suit your needs. Please bear in mind
that different recipes or configurations may necessitate varying hardware capabilities.
For example, if you are experiencing a shortage of GPU VRAM, consider opting for one of
the `qlora` configurations, which implement aggressive quantization.

## Execution Guidelines

Prior to executing any of the recipes, ensure that you have installed all the necessary
dependencies on the host where you intend to execute the tuning job. To expedite this
process, we provide a [Docker container](https://github.com/fw-ai/cookbook/tree/main/recipes/docker/text)
equipped with all the required dependencies.

As fine-tuning can harness multiple devices, we recommend executing recipes via a distributed
training launcher such as [`torchx`](https://github.com/pytorch/torchx).
