# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

from functools import partial
from importlib import import_module
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from recipes.common.batch_transform import BatchTransform
from transformers import AutoTokenizer


def create_transform(config: DictConfig, tokenizer: AutoTokenizer) -> BatchTransform:
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


def _tokenize(
    tokenizer: AutoTokenizer,
    prompt_column: str,
    completion_column: str,
    max_length: int,
    mask_prompt: bool,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tokenizes a single row of data, optionally masking the prompt.

    Args:
        tokenizer: the tokenizer used for tokenization.
        prompt_column: column name containing the prompt text.
        completion_column: column name containing the completion text.
        max_length: maximum allowed token length after tokenization.
        mask_prompt: flag to determine if the prompt should be masked.
        row: the data row to tokenize.

    Returns:
        Dictionary containing tokenized data.
    """
    tokenized_row = tokenizer(
        row[prompt_column] + row[completion_column],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    if not mask_prompt:
        tokenized_row["labels"] = tokenized_row["input_ids"].copy()
        return tokenized_row
    tokenized_prompt = tokenizer(
        row[prompt_column],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    # subtract 1 to account for EOS token
    prompt_len = len(tokenized_prompt["input_ids"]) - 1
    tokenized_row["labels"] = [-100] * prompt_len + tokenized_row["input_ids"][
        prompt_len:
    ]
    return tokenized_row


def _pack(name: str, data: Dataset, max_length: int) -> Dataset:
    """
    Packs sequences from the dataset such that the total length of packed sequences
    does not exceed the specified maximum length. Sequences are concatenated until
    the max_length is reached or nearly reached.

    Args:
        name: name or description of the dataset, used for logging,
        data: dataset containing sequences to pack,
        max_length: maximum allowed length for the packed sequences.

    Returns:
        A new Dataset containing the packed sequences.
    """
    length = 0
    input_ids, attention_mask, labels = [], [], []
    result_input_ids, result_attention_mask, result_labels = [], [], []

    def merge(l: List[List[int]]) -> List[int]:
        return [item for sublist in l for item in sublist]

    for row in tqdm(data, desc=f"packing dataset {name}"):
        row_input_ids, row_attention_mask, row_labels = (
            row["input_ids"],
            row["attention_mask"],
            row["labels"],
        )
        if length + len(row_input_ids) <= max_length:
            input_ids.append(row_input_ids)
            attention_mask.append(row_attention_mask)
            labels.append(row_labels)
        else:
            result_input_ids.append(merge(input_ids))
            result_attention_mask.append(merge(attention_mask))
            result_labels.append(merge(labels))

            input_ids, attention_mask, labels = (
                [row_input_ids],
                [row_attention_mask],
                [row_labels],
            )
            length = 0

        length += len(row_input_ids)

    if input_ids:
        result_input_ids.append(merge(input_ids))
        result_attention_mask.append(merge(attention_mask))
        result_labels.append(merge(labels))

    return Dataset.from_dict(
        {
            "input_ids": result_input_ids,
            "attention_mask": result_attention_mask,
            "labels": result_labels,
        }
    )


def prepare_training_data(config: DictConfig, tokenizer: AutoTokenizer) -> DatasetDict:
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
        path = spec.get("huggingface_name", spec.get("format"))
        kwargs = {}
        if "data_files" in spec:
            kwargs["data_files"] = spec["data_files"]
        if "huggingface_revision" in spec:
            kwargs["revision"] = spec["huggingface_revision"]
        if spec.get("subset"):
            dataset = load_dataset(path, spec.subset, split=split, **kwargs)
        else:
            dataset = load_dataset(path, split=split, **kwargs)
        print(f"loaded dataset {name} of size {len(dataset)}")

        transform = create_transform(spec.transform, tokenizer)
        dataset.set_transform(transform.__call__)
        prompt_column = transform.prompt_column
        completion_column = transform.completion_column

        max_length = config.model.cutoff_len
        print(f"using max length {max_length}")
        dataset = dataset.filter(
            lambda row: len(
                tokenizer(row[prompt_column] + row[completion_column])["input_ids"]
            )
            <= max_length,
            num_proc=16,
        )
        dataset = dataset.shuffle(seed=1234)
        max_samples = spec.get("max_samples")
        if max_samples is not None and max_samples > 0:
            max_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples))
            print(f"truncated dataset {name} to size {len(dataset)}")

        for i in range(min(len(dataset), 3)):
            row = dataset[i]
            print(
                f"sample prompt {i} for dataset {name}:\n{row[prompt_column] + row[completion_column]}"
            )

        columns_to_remove = dataset.column_names
        tokenize = partial(
            _tokenize,
            tokenizer,
            prompt_column,
            completion_column,
            max_length,
            config.data.mask_prompt,
        )
        dataset = dataset.map(tokenize, remove_columns=columns_to_remove, num_proc=16)

        dataset.reset_format()

        if spec.get("pack", False):
            dataset = _pack(name, dataset, max_length)

        datasets.append(dataset)

    dataset = concatenate_datasets(datasets)

    print(f"using {len(dataset)} examples for training")

    return dataset.shuffle(seed=1234)
