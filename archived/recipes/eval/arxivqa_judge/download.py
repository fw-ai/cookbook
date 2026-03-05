# Copyright (c) Fireworks AI, Inc. and affiliates.
#
# All Rights Reserved.

# Run as:
#   export PYTHONPATH=.:<cookbook_root>
#   python download.py

import asyncio
import os

import aiohttp
import backoff
import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from recipes.eval.common.util import exists, write_file
from tqdm import tqdm


@backoff.on_exception(
    backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3, jitter=None
)
async def _fetch_content(
    config: DictConfig, session: aiohttp.ClientSession, url: str
) -> bytes:
    async with session.get(
        url, timeout=aiohttp.ClientTimeout(total=config.request_timeout)
    ) as response:
        if response.status == 404:
            print(f"{url} not found")
            # Return None for 404 to skip this document
            return None
        response.raise_for_status()
        return await response.read()


async def _process_document(
    config: DictConfig,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    document_id: str,
) -> bool:
    url = config.url_template.format(document_id=document_id)
    path = os.path.join(config.output_path, f"{document_id}.pdf")

    if exists(path):
        # Already exists, consider it successfully processed.
        print(f"{path} already exists")
        return True

    async with semaphore:
        content = await _fetch_content(config, session, url)
        if content is None:
            # This was a 404
            return False
        write_file(content, path)
        return True


async def _process(config: DictConfig) -> None:
    # Create output directory if it's a local path
    if "://" not in config.output_path:
        os.makedirs(config.output_path, exist_ok=True)

    semaphore = asyncio.Semaphore(config.concurrency)

    # Load data
    dataset = load_dataset(config.huggingface_dataset)
    dataset = dataset[config.split]
    document_ids = list({x[config.document_id_column] for x in dataset})

    total_docs = len(document_ids)
    num_404s = 0

    async with aiohttp.ClientSession() as session:
        with tqdm(total=total_docs, desc="Downloading documents") as pbar:
            for i in range(0, total_docs, config.batch_size):
                batch = document_ids[i : i + config.batch_size]

                tasks = [
                    asyncio.create_task(
                        _process_document(config, semaphore, session, doc_id)
                    )
                    for doc_id in batch
                ]

                for fut in asyncio.as_completed(tasks):
                    success = await fut
                    if not success:
                        num_404s += 1
                        if num_404s / (i + config.batch_size) > config.tolerance:
                            raise RuntimeError(
                                f"Too many not found documents: {num_404s}/{i + config.batch_size}"
                            )
                    pbar.update(1)

                print(f"Not found count: {num_404s}/{i + config.batch_size}")


@hydra.main(version_base=None, config_path="conf", config_name="download")
def _app(config: DictConfig) -> None:
    print(f"config: {OmegaConf.to_yaml(config, resolve=True)}")
    asyncio.run(_process(config))


if __name__ == "__main__":
    _app()
