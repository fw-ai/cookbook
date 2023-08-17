
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

Here is the config tree structure for a typical tuning recipe:
```
|-- conf
|   |-- data
|   |   |-- dataset
|   |   |   |-- dataset1.yaml
|   |   |   |-- dataset2.yaml
|   |   |   `-- ...
|   |   |-- data_collection1.yaml
|   |   |-- data_collection2.yaml
|   |   `-- ...
|   |-- model
|   |   |-- model1.yaml
|   |   |-- model2.yaml
|   |   `-- ...
|   |-- tuning_program1.yaml
|   |-- tuning_program2.yaml
|   `-- ...
|-- tuning_notebook.ipynb
|-- tuning_command.py
|-- README.md
`-- ...
```
In the above structure:
* **dataset/dataset.yaml** describes how to load and transform a single dataset,
* **data/data_collection.yaml** is a bundle of datasets used by a tuning program,
* **model/model.yaml** references a specific base model and defines the trainer parameters,
* **conf/tuning_program.yaml** bundles all configs defining a single fine tuning job - i.e., each tuning program references a single config,
* **tuning_[notebook|command]** are the program executors. Some recipes may come with different flavors of executors such as command line tools and notebooks.

Feel free to modify or add configurations to suit your needs. Please bear in mind
that different recipes or configurations may necessitate varying hardware capabilities.
For example, if you are experiencing a shortage of GPU VRAM, consider opting for one of
the `qlora` configurations, which implement aggressive quantization.

## Finetuning data

By default, the recipes pull the data from HuggingFace :hugs: hub. If you need to train on custom data,
you may want to [upload](https://huggingface.co/docs/datasets/upload_dataset) it to a (possibly private) HuggingFace
:hugs: dataset.
Very soon we will be adding functionality to load the data from a local drive so stay tuned.

## Execution Guidelines

Prior to executing any of the recipes, ensure that you have installed all the necessary
dependencies on the host where you intend to execute the tuning job. To expedite this
process, we provide a [Docker container](https://github.com/fw-ai/cookbook/tree/main/recipes/docker/text)
equipped with all the required dependencies.

As fine-tuning can harness multiple devices, we recommend executing recipes via a distributed
training launcher such as [`torchx`](https://github.com/pytorch/torchx).
