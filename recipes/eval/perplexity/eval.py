# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as: python eval.py

import collections
import random
from typing import Dict, List

import hydra
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
import torch
from recipes.common.peft import load_inference_model
from recipes.common.tokenizer import load_tokenizer
from recipes.common.env import init_env
from recipes.eval.perplexity_rank.transform import DatasetTransform
from tqdm import tqdm
from transformers import AutoTokenizer


def _prepare_data(config: DictConfig) -> Dataset:
    """
    Prepares evaluation dataset.

    Args:
        config: configuration parameters describing the dataset.

    Returns:
        loaded dataset.
    """
    path = config.get("path", config.get("path", config.get("huggingface_name")))
    dataset = load_dataset(
        path,
        revision=config.get("huggingface_revision"),
        split=config.get("split", "train"),
        data_files=config.get("data_files"),
    )
    print(f"loaded dataset {path} of size {len(dataset)}")

    transform = DatasetTransform.create(config.transform)
    dataset = transform(dataset)

    return dataset


def _patch(config: DictConfig) -> None:
    """
    Applies module patches.

    Args:
        config: the config describing patching behavior.
    """
    if config.model.get("flash_attention"):
        # flash attention may not have been installed
        from recipes.common.llama_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()


def _evaluate(
    config: DictConfig,
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """
    Evaluates the data using perplexity scoring.

    Args:
        config: the configuration describing the evaluation program,
        texts: the dataset to use for evaluation,
        model: the model to evaluate.
        tokenizer: the tokenizer associated with the model.

    Returns:
        computed evaluation metrics.
    """
    stride = config.stride
    num_tokens = config.num_tokens
    perplexity = collections.defaultdict(int)
    device = torch.cuda.current_device()
    total_input_tokens = 0
    for text in tqdm(texts, desc="computing perplexity"):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        num_input_tokens = input_ids.size(1)
        total_input_tokens += num_input_tokens
        if num_input_tokens < num_tokens:
            raise ValueError(
                f"expected at least {num_tokens}, found {num_input_tokens}"
            )
        input_ids = input_ids[:, :num_tokens].to(device)
        torch.cuda.reset_peak_memory_stats()
        for n in range(stride, num_tokens + 1, stride):
            local_input_ids = input_ids[:, :n]
            with torch.no_grad():
                loss = model(input_ids=local_input_ids, labels=local_input_ids).loss
            perplexity[n] += torch.exp(loss).item()
    perplexity = {k: v / len(texts) for k, v in perplexity.items()}
    return {
        "avg_input_tokens": total_input_tokens / len(texts),
        "perplexity": perplexity,
        "peak_memory_usage (GB)": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_memory_cached (GB)": torch.cuda.max_memory_cached(device) / 1024**3,
    }


@hydra.main(
    version_base=None, config_path="conf", config_name="codellama-7b-the-stack-smol"
)
def _app(config: DictConfig) -> None:
    """
    Runs the evaluation program.

    Args:
        config_path: the directory sting config files,
        config_name: selected config to run.
    """
    random.seed(1234)
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")
    _patch(config)
    init_env()
    dataset = _prepare_data(config.dataset)
    tokenizer = load_tokenizer(config.model)
    model = load_inference_model(config)
    stats = _evaluate(config, dataset["text"], model, tokenizer)
    print(f"eval stats: {stats}")


if __name__ == "__main__":
    _app()
