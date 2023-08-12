# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

from importlib import import_module

from datasets import DatasetDict, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from recipes.common.batch_transform import BatchTransform
from transformers import AutoTokenizer


def _create_transform(config: DictConfig,
                      tokenizer: AutoTokenizer) -> BatchTransform:
    """
    Creates a transform object from the provided config.

    Args:
        config: the config with transform parameters,
        tokenizer: the tokenizer that can be used to customize the
            transform.

    Returns:
        transform object.
    """
    clazz = config.get("class")
    module_name, class_name = clazz.rsplit(".", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    return cls(config, tokenizer)


def prepare_training_data(config: DictConfig,
                          tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Prepares training dataset by combining multiple HF datasets.

    Args:
        config: configuration parameters describing the dataset,
        tokenizer: the tokenizer used to generate EOS token.

    Returns:
        loaded dataset.
    """
    datasets = []
    for name, spec in config.data.dataset.items():
        split = spec.get("split", "train")
        data_files = spec.get("data_files")
        if spec.get("subset"):
            dataset = load_dataset(spec.huggingface_name,
                                   spec.subset,
                                   revision=spec.huggingface_revision,
                                   split=split,
                                   data_files=data_files)
        else:
            dataset = load_dataset(spec.huggingface_name,
                                   revision=spec.huggingface_revision,
                                   split=split)
        print(f"loaded dataset {name} of size {len(dataset)}")

        transform = _create_transform(spec.transform, tokenizer)
        dataset.set_transform(transform.__call__)
        key = transform.output_column

        max_length = config.model.cutoff_len
        print(f"using max length {max_length}")
        dataset = dataset.filter(
            lambda row: len(tokenizer(row[key])["input_ids"]) <= max_length,
            num_proc=16)
        dataset = dataset.shuffle(seed=1234)
        max_samples = spec.get("max_samples")
        if max_samples is not None and max_samples > 0:
            max_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples))
            print(f"truncated dataset {name} to size {len(dataset)}")

        for i in range(min(len(dataset), 3)):
            print(f"sample prompt {i} for dataset {name}:\n{dataset[i][key]}")

        columns_to_remove = dataset.column_names
        dataset = dataset.map(lambda row: tokenizer(
            row[key],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ),
                              remove_columns=columns_to_remove,
                              num_proc=16)

        dataset.reset_format()

        datasets.append(dataset)

    dataset = concatenate_datasets(datasets)

    print(f"using {len(dataset)} examples for training")

    return dataset.shuffle(seed=1234)
