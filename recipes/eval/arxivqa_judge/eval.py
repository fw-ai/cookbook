# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as:
#   export PYTHONPATH=.:<cookbook_root>
#   python eval.py

import asyncio
import base64
import json
import os
import random
from datetime import datetime
from typing import Any, Callable

import hydra
import pymupdf
import transformers
from anthropic import AsyncAnthropic
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from recipes.eval.common.hf_data import load_data
from recipes.eval.common.llm_client import achat_completion_create_retrying
from recipes.eval.common.util import exists, read_file
from tqdm import tqdm
from transformers import AutoTokenizer


def _load_data(config: OmegaConf) -> Dataset:
    """
    Loads and transforms input data.

    Args:
        config: Data config.

    Result:
        The transformed dataset.
    """
    dataset = load_data(config)

    max_samples = config.get("max_samples", -1)
    if max_samples > 0 and config.get("overfetch_factor", 0) > 0:
        effective_max_samples = min(len(dataset), max_samples * config.overfetch_factor)
        dataset = dataset.select(range(effective_max_samples))
        print(f"truncated initial dataset to size {len(dataset)}")

    transform = hydra.utils.get_class(config.transform.get("class"))(
        config.transform, AutoTokenizer.from_pretrained(config.tokenizer)
    )
    dataset = dataset.map(
        transform,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=32,
        num_proc=16,
        desc="Transforming data",
    )

    def filter_too_long(example: dict[str, Any]) -> bool:
        if exists(example["_path"]):
            content = read_file(example["_path"])
        else:
            print(f"file not found: {example['_path']}")
            return False
        encoded = base64.b64encode(content).decode("utf-8")
        if len(encoded) > config.max_size:
            return False
        doc = pymupdf.open(stream=content, filetype="pdf")
        return doc.page_count <= config.max_pages

    dataset = dataset.filter(
        filter_too_long,
        batch_size=10,
        num_proc=32,
        desc="Filtering out too long documents",
    )
    dataset = dataset.shuffle(seed=1234)

    if max_samples > 0:
        effective_max_samples = min(len(dataset), max_samples)
        dataset = dataset.select(range(effective_max_samples))
        print(f"truncated dataset to size {len(dataset)}")

    dataset.reset_format()

    print(f"dataset: {dataset}")

    return dataset


def _load_document(config: DictConfig, content: bytes) -> list[str]:
    """
    Loads document.

    Args:
        config: Document config.
        content: The document content.

    Result:
        The document URLs.
    """
    if config.get("transform"):
        encoded_url = base64.b64encode(content).decode("utf-8")
        encoded_url = f"data:application/pdf;base64,{encoded_url}"
        encoded_url = f"{encoded_url}#transform={config.transform}"
        return [encoded_url]

    doc = pymupdf.open(stream=content, filetype="pdf")
    result = []

    # Iterate over each page in the PDF
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)  # Load the page
        # pix = page.get_pixmap(dpi=200)  # Create a Pixmap (image)
        pix = page.get_pixmap(dpi=100)  # Create a Pixmap (image)
        encoded_url = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
        encoded_url = f"data:image/jpeg;base64,{encoded_url}"
        result.append(encoded_url)

    return result


def _prepare_conversation(
    config: DictConfig, row: dict[str, Any], content: bytes
) -> list[dict[str, Any]]:
    """
    Prepares conversation.

    Args:
        config: Conversation config.
        row: The row.
        content: The document content.

    Result:
        The conversation.
    """
    messages = []
    if config.get("system_message") is not None:
        messages.append({"role": "system", "content": config.system_message})
    user_message = config.get("user_message")
    user_message = user_message.format(**row)

    document_urls = [
        {
            "type": "image_url",
            "image_url": {
                "url": document_url,
                **({"detail": config.detail} if config.get("detail") else {}),
            },
        }
        for document_url in _load_document(config, content)
    ]

    messages.append(
        {
            "role": "user",
            "content": document_urls + [{"type": "text", "text": user_message}],
        }
    )
    return messages


def _prepare_judge_conversation(
    config: DictConfig,
    row: dict[str, Any],
    content: bytes,
    answer_a: str,
    answer_b: str,
) -> list[dict[str, Any]]:
    """
    Prepares judge conversation.

    Args:
        config: Judge config.
        row: The row.
        content: The document content.
        answer_a: The answer A.
        answer_b: The answer B.

    Result:
        The judge conversation.

    """
    user_message = config.user_message.format(
        **row, answer_a=answer_a, answer_b=answer_b
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.b64encode(content).decode("utf-8"),
                    },
                },
                {"type": "text", "text": user_message},
            ],
        }
    ]


async def _generate(
    config: DictConfig,
    row: dict[str, Any],
    content: bytes,
    achat_completion_create: Callable,
) -> tuple[str, int]:
    """
    Generates answer.

    Args:
        config: Generate config.
        row: The row.
        content: The document content.
        achat_completion_create: The chat completion create function.

    Result:
        The answer and number of prompt tokens.
    """
    messages = _prepare_conversation(config, row, content)
    response = await achat_completion_create(
        model=config.model, messages=messages, temperature=0
    )
    # print(f"DEBUG: response: {response}")
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    # prompt_tokens = response.usage.prompt_tokens
    try:
        return response.choices[0].message.content.strip(), prompt_tokens
    except Exception as _:
        print(f"failed to parse response: {response}")
        return response.model_dump_json(), 0


async def _process_row(
    config: DictConfig,
    data_pbar: tqdm,
    request_pbar: tqdm,
    row: dict[str, Any],
    metrics: dict[str, int],
    output_path: str,
    achat_completion_create_a: Callable,
    achat_completion_create_b: Callable,
    achat_completion_create_judge: Callable,
) -> None:
    """
    Processes row.

    Args:
        config: Config.
        data_pbar: The data progress bar.
        request_pbar: The request progress bar.
        row: The row.
        metrics: The metrics.
        output_path: The output path.
        achat_completion_create_a: The chat completion create function for A.
        achat_completion_create_b: The chat completion create function for B.
        achat_completion_create_judge: The chat completion create function for judge.
    """
    if not exists(row["_path"]):
        print(f"file not found: {row['_path']}")
        data_pbar.update()
        request_pbar.update()
        return
    content = read_file(row["_path"])
    data_pbar.update()
    metrics["total"] += 1
    answer_a, prompt_tokens_a = await _generate(
        config.generate_a, row, content, achat_completion_create_a
    )
    answer_b, prompt_tokens_b = await _generate(
        config.generate_b, row, content, achat_completion_create_b
    )

    if random.random() < 0.5:
        messages = _prepare_judge_conversation(
            config.judge, row, content, answer_a, answer_b
        )
        reverse = False
    else:
        messages = _prepare_judge_conversation(
            config.judge, row, content, answer_b, answer_a
        )
        reverse = True

    if prompt_tokens_a == 0 or prompt_tokens_b == 0:
        # there was an error in generating the answers
        result = "invalid"
    else:
        response = await achat_completion_create_judge(
            model=config.judge.model,
            messages=messages,
            temperature=0,
            max_tokens=config.judge.max_tokens,
        )
        result = response.content[0].text.lower().strip()

    if reverse:
        result = (
            "b>a"
            if result.startswith("a>b")
            else "a>b" if result.startswith("b>a") else result
        )
    if result.startswith("a>b"):
        metrics["a>b"] += 1
    elif result.startswith("b>a"):
        metrics["b>a"] += 1
    elif result.startswith("a=b"):
        metrics["a=b"] += 1
    else:
        metrics["invalid"] += 1
        print(f"question: {row['question']} invalid result: {result}")
    print(
        f"DEBUG: prompt_tokens_a: {prompt_tokens_a}, prompt_tokens_b: {prompt_tokens_b}"
    )
    metrics["prompt_tokens_a"] += prompt_tokens_a
    metrics["prompt_tokens_b"] += prompt_tokens_b
    request_pbar.update()

    result_data = {
        "question": row["question"],
        "a": config.generate_a.model,
        "b": config.generate_b.model,
        "judge": config.judge.model,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "judge_result": result,
        "document_path": row["_path"],
        "prompt_tokens_a": prompt_tokens_a,
        "prompt_tokens_b": prompt_tokens_b,
    }
    with open(output_path, "a") as f:
        json.dump(result_data, f)
        f.write("\n")
    print(f"metrics: {json.dumps(metrics, indent=2)}")

    return result


async def _acall(config: DictConfig, data: Dataset) -> list[str]:
    """
    Calls the main function.

    Args:
        config: Config.
        data: The data.

    Result:
        The result.
    """
    # init
    metrics = {
        "a": config.generate_a.model,
        "b": config.generate_b.model,
        "judge": config.judge.model,
        "total": 0,
        "a>b": 0,
        "b>a": 0,
        "a=b": 0,
        "invalid": 0,
        "prompt_tokens_a": 0,
        "prompt_tokens_b": 0,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(config.working_dir, f"results_{timestamp}.jsonl")
    stats_path = os.path.join(config.working_dir, f"stats_{timestamp}.json")

    # generate setup
    semaphore_a = asyncio.Semaphore(config.generate_a.concurrency)
    client_a = AsyncOpenAI(
        api_key=config.generate_a.api_key,
        base_url=config.generate_a.base_url,
        default_headers=(
            {
                "x-fireworks-account-id": config.generate_a.account,
            }
            if config.generate_a.get("account")
            else {}
        ),
    )
    achat_completion_create_a = achat_completion_create_retrying(client_a, semaphore_a)

    semaphore_b = asyncio.Semaphore(config.generate_b.concurrency)
    client_b = AsyncOpenAI(
        api_key=config.generate_b.api_key,
        base_url=config.generate_b.base_url,
        default_headers=(
            {
                "x-fireworks-account-id": config.generate_b.account,
            }
            if config.generate_b.get("account")
            else {}
        ),
    )
    achat_completion_create_b = achat_completion_create_retrying(client_b, semaphore_b)

    # judge setup
    judge_semaphore = asyncio.Semaphore(config.judge.concurrency)
    judge_client = AsyncAnthropic(api_key=config.judge.api_key)
    achat_completion_create_judge = achat_completion_create_retrying(
        judge_client, judge_semaphore
    )

    # create tasks
    tasks = []
    data_pbar = tqdm(total=len(data), desc="preparing data")
    request_pbar = tqdm(total=len(data), desc="generating responses")
    for row in data:
        task = asyncio.create_task(
            _process_row(
                config,
                data_pbar,
                request_pbar,
                row,
                metrics,
                output_path,
                achat_completion_create_a,
                achat_completion_create_b,
                achat_completion_create_judge,
            )
        )
        tasks.append(task)

    # process results
    result = await asyncio.gather(*tasks)

    stats = {
        "config": OmegaConf.to_object(config),
        "metrics": metrics,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"written output to {output_path}")
    print(f"written stats to {stats_path}")

    return result


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def _app(config: DictConfig) -> None:
    """
    Computes stats from data.

    Args:
        config: Config.
    """
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")

    transformers.set_seed(123)
    random.seed(123)

    dataset = _load_data(config.dataset)
    asyncio.run(_acall(config, dataset))


if __name__ == "__main__":
    _app()
