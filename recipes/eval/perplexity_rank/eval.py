# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as: python eval.py

import random
from collections import defaultdict
import re
from typing import Any, Dict, List, Tuple

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from peft.peft_model import PeftModel
from recipes.common.env import init_env
from recipes.common.peft import load_inference_model
from recipes.common.tokenizer import load_tokenizer
from recipes.eval.perplexity_rank.transform import DatasetTransform
from tqdm import tqdm

from transformers import AutoTokenizer

_global_stats = {
    "matched_completions": 0,
    "mismatched_completions": 0,
}


def _prepare_data(config: DictConfig, tokenizer: AutoTokenizer) -> Dataset:
    """
    Prepares evaluation dataset.

    Args:
        config: configuration parameters describing the dataset,
        tokenizer: the tokenizer used to generate EOS token.

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

    transform = DatasetTransform.create(config.transform, tokenizer)
    dataset = transform(dataset)

    return dataset


def _patch(config: DictConfig) -> None:
    """
    Applies module patches.

    Args:
        config: the config describing patching behavior.
    """
    if config.model.flash_attention:
        # flash attention may not have been installed
        from recipes.common.llama_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()


def _perplexity(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    model: PeftModel,
    query: str,
    document: str,
    device: torch.device,
) -> float:
    """
    Calculates perplexity of the generated completion.

    Args:
        config: config describing the evaluation task,
        tokenizer: the tokenizer to use for encoding and decoding,
        model: the sequence generation model,
        query: the query to use in the prompt,
        document: the document to include in the prompt,
        device: the device where to run the inference.

    Returns:
        perplexity calculated on the completion.
    """
    prompt = config.prompt_template.format(document=document, query=query)
    num_document_tokens = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    completion = config.completion_template.format(document=document, query=query)
    text = prompt + completion
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    num_target_tokens = len(input_ids[0]) - num_document_tokens
    target_ids = input_ids.clone()
    target_ids[:, :-num_target_tokens] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    _global_stats["matched_completions"] += 1

    return neg_log_likelihood


def _parse_completion(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    model: PeftModel,
    query: str,
    document: str,
    device: torch.device,
) -> float:
    """
    Scores completions w.r.t. a provided template.

    Args:
        config: config describing the evaluation task,
        tokenizer: the tokenizer to use for encoding and decoding,
        model: the sequence generation model,
        query: the query to use in the prompt,
        document: the document to include in the prompt,
        device: the device where to run the inference.

    Returns:
        0.0 if the completion matches the template, 1.0 otherwise.
    """
    prompt = config.prompt_template.format(document=document, query=query)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
    )
    outputs = outputs.squeeze(0)
    try:
        index = outputs.tolist().index(tokenizer.eos_token_id)
        outputs = outputs[:index]
    except ValueError:
        ...
    outputs = outputs.unsqueeze(0)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    start = decoded.find(prompt) + len(prompt)
    completion = decoded[start:].strip().lower()
    positive_match = config.completion_positive_marker.format(
        document=document, query=query
    ).lower()
    pattern = re.compile(config.completion_pattern)
    if pattern.match(completion):
        _global_stats["matched_completions"] += 1
    else:
        _global_stats["mismatched_completions"] += 1
    score = 0.0 if positive_match in completion else 1.0
    return score


def _shuffle_aligned_lists(
    list1: List[Any], list2: List[Any]
) -> Tuple[List[Any], List[Any]]:
    """
    Shuffles two lists while ensuring that the elements at corresponding
    indices remain aligned.

    Args:
        list1: the first list to be shuffled,
        list2: the second list to be shuffled. Must have the same length
            as list1.

    Returns:
        tuple containing two lists. The first element is the shuffled version
            of list1, and the second element is the shuffled version of list2.
    """

    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    combined = list(zip(list1, list2))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2 = zip(*combined)

    return list(shuffled_list1), list(shuffled_list2)


def _recall(
    predictions: List[float], labels: List[int], recall_limits: List[int]
) -> Tuple[List[float], List[float]]:
    """
    Computes recalls from predictions and ground truth labels.

    Args:
        predictions: the predictions generated by the model (lower is better),
        labels: ground truth labels from the dataset (greater is better),
        recall_limits: the boundaries of recall computation - i.e., the
            values of k in recall@k.

    Returns:
        tuple of lists of the same length equal to len(recall_limits):
            - the recall values for each limit,
            - the baseline recall that corresponds to random ordering of
                data samples.
    """
    # remove position bias
    predictions, labels = _shuffle_aligned_lists(predictions, labels)
    top_score = max(labels)
    top_indices = [i for i in range(len(labels)) if labels[i] == top_score]
    labels = set(top_indices)

    prediction_ids = [(prediction, i) for i, prediction in enumerate(predictions)]
    sorted_prediction_ids = sorted(prediction_ids)
    sorted_predictions = [x[1] for x in sorted_prediction_ids]

    recalls = []
    baseline_recalls = []
    for recall_limit in recall_limits:
        if len(sorted_predictions) < recall_limit:
            recalls.append(-1.0)
            baseline_recalls.append(-1)
        else:
            intersection = set(sorted_predictions[:recall_limit]).intersection(labels)
            denominator = min(len(labels), recall_limit)
            recalls.append(float(len(intersection)) / denominator)
            n = float(len(predictions))
            l = float(len(labels))
            baseline_recalls.append((recall_limit / (n / l)) / l)

    return (recalls, baseline_recalls)


def _evaluate(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    model: PeftModel,
    dataset: Dataset,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates the data using perplexity scoring.

    Args:
        config: the configuration describing the evaluation program,
        tokenizer: the tokenizer to use for encoding of the instruction and
            decoding of the results,
        model: the model generating responses,
        dataset: the dataset to use for evaluation,
        device: the default device where the eval should run.

    Returns:
        computed evaluation metrics.
    """
    _NAME_TO_SCORING_FN = {
        "perplexity": _perplexity,
        "parse_completion": _parse_completion,
    }
    scoring_fn = _NAME_TO_SCORING_FN[config.scoring]
    i = 0
    sum_recall = defaultdict(float)
    num_recall = defaultdict(int)
    sum_baseline_recall = defaultdict(float)
    pbar = tqdm(total=len(dataset), desc="evaluating the model")
    while i < len(dataset):
        query = dataset[i]["query"]
        documents = []
        scores = []
        while i < len(dataset) and dataset[i]["query"] == query:
            document = dataset[i]["document"]
            documents.append(document)
            scores.append(dataset[i]["score"])
            i += 1
            pbar.update(1)

        perplexities = []
        for document in documents:
            perplexity = scoring_fn(config, tokenizer, model, query, document, device)
            perplexities.append(perplexity)

        recalls, baseline_recalls = _recall(perplexities, scores, config.recall_limits)
        for recall_limit, recall, baseline_recall in zip(
            config.recall_limits, recalls, baseline_recalls
        ):
            if recall >= 0:
                sum_recall[recall_limit] += recall
                num_recall[recall_limit] += 1
                sum_baseline_recall[recall_limit] += baseline_recall

    result = {
        "recall": {},
        "baseline_recall": {},
    }
    for recall_limit in sum_recall:
        result["recall"][recall_limit] = (
            float(sum_recall[recall_limit]) / num_recall[recall_limit]
        )
        result["baseline_recall"][recall_limit] = (
            float(sum_baseline_recall[recall_limit]) / num_recall[recall_limit]
        )

    return result


@hydra.main(version_base=None, config_path="conf", config_name="msmarco_rank_p_q_d")
def _app(config: DictConfig) -> None:
    """
    Runs the evaluation program.

    Args:
        config_path: the directory sting config files,
        config_name: selected config to run.
    """
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")
    _patch(config)
    env = init_env()
    tokenizer = load_tokenizer(config)
    dataset = _prepare_data(config.dataset, tokenizer)
    model = load_inference_model(config, tokenizer, env.device)
    stats = _evaluate(config, tokenizer, model, dataset, env.device)
    print(f"eval stats: {stats}")
    print(f"global stats: {_global_stats}")


if __name__ == "__main__":
    _app()
