#!/usr/bin/env python3
"""Reproduce missing raw_output in streaming responses.

Fires many requests and logs every case where raw_output is missing
from the assembled streaming response, including the full response
for debugging.
"""

from __future__ import annotations

import os
import sys
import asyncio
import json
import logging
import time

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import transformers
from fireworks.training.sdk.deployment import (
    DeploymentConfig,
    DeploymentManager,
    DeploymentSampler,
    ServerMetrics,
    _SSEDecoder,
)
from fireworks.training.sdk.errors import async_request_with_retries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FIREWORKS_API_KEY", "fw_58efLjimG74e2zwAf69iqS")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = "e2e-repro-rawoutput-chengxili-v1"
DEPLOYMENT_SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-30b-a3b-throughput/versions/ai5r0aoy"
BASE_MODEL = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"

NUM_REQUESTS = 500
CONCURRENCY = 32
MAX_TOKENS = 512


async def send_one(
    client,
    url: str,
    headers: dict,
    payload: dict,
    idx: int,
) -> dict:
    """Send one streaming request and return diagnosis info."""
    import json as _json

    t0 = time.time()
    resp = await async_request_with_retries(
        client.post, url, headers=headers, json=payload, timeout=120,
    )

    if resp.status_code != 200:
        return {"idx": idx, "status": resp.status_code, "error": "non-200", "elapsed": time.time() - t0}

    # Parse SSE manually to capture everything
    accumulated_text = ""
    raw_output = None
    finish_reason = None
    all_chunk_keys = []
    chunk_count = 0
    perf_metrics = None

    decoder = _SSEDecoder()
    async for sse in decoder.aiter_events(resp):
        if sse.data.startswith("[DONE]"):
            break
        try:
            chunk = _json.loads(sse.data)
        except (ValueError, TypeError):
            continue

        chunk_count += 1
        for choice in chunk.get("choices", []):
            choice_keys = sorted(choice.keys())
            all_chunk_keys.append(choice_keys)

            text_delta = choice.get("text", "")
            if text_delta:
                accumulated_text += text_delta

            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            ro = choice.get("raw_output")
            if ro:
                raw_output = ro

        pm = chunk.get("perf_metrics")
        if pm:
            perf_metrics = pm

    elapsed = time.time() - t0
    has_token_ids = raw_output is not None and raw_output.get("completion_token_ids") is not None

    result = {
        "idx": idx,
        "status": 200,
        "has_raw_output": raw_output is not None,
        "has_token_ids": has_token_ids,
        "finish_reason": finish_reason,
        "text_len": len(accumulated_text),
        "chunk_count": chunk_count,
        "elapsed": round(elapsed, 2),
    }

    if not has_token_ids:
        # Capture full diagnosis for missing raw_output
        result["all_chunk_keys"] = all_chunk_keys
        result["raw_output_value"] = raw_output
        result["text_preview"] = accumulated_text[:200]
        result["has_perf_metrics"] = perf_metrics is not None
        logger.warning("MISSING raw_output on request %d: chunks=%d, finish=%s, text_len=%d, keys=%s",
                        idx, chunk_count, finish_reason, len(accumulated_text), all_chunk_keys[-3:] if all_chunk_keys else [])

    return result


async def run(inference_url: str, inference_model: str, tokenizer):
    import httpx

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
         {"role": "user", "content": "What is the probability that a fair coin flipped 10 times gives exactly 5 heads?"}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )

    client = httpx.AsyncClient(
        verify=True,
        timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=0),
    )

    url = f"{inference_url}/inference/v1/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "X-Api-Key": API_KEY}
    payload = {
        "model": inference_model,
        "prompt": prompt_ids,
        "n": 1,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True,
        "raw_output": True,
        "logprobs": True,
        "perf_metrics_in_response": True,
    }

    sem = asyncio.Semaphore(CONCURRENCY)
    missing = []
    ok = 0

    async def _guarded(idx):
        nonlocal ok
        async with sem:
            result = await send_one(client, url, headers, payload, idx)
            if result.get("has_token_ids"):
                ok += 1
            else:
                missing.append(result)
            if (idx + 1) % 50 == 0:
                logger.info("Progress: %d/%d sent, %d missing so far", idx + 1, NUM_REQUESTS, len(missing))

    tasks = [_guarded(i) for i in range(NUM_REQUESTS)]
    await asyncio.gather(*tasks)

    await client.aclose()

    logger.info("=" * 60)
    logger.info("RESULTS: %d/%d OK, %d missing raw_output", ok, NUM_REQUESTS, len(missing))
    if missing:
        logger.info("Missing cases:")
        for m in missing:
            logger.info("  %s", json.dumps(m, default=str))
    else:
        logger.info("No missing raw_output cases found!")
    logger.info("=" * 60)

    return missing


def main():
    mgr = DeploymentManager(api_key=API_KEY, base_url=BASE_URL)

    logger.info("Creating fresh deployment %s...", DEPLOYMENT_ID)
    config = DeploymentConfig(
        deployment_id=DEPLOYMENT_ID,
        base_model=BASE_MODEL,
        deployment_shape=DEPLOYMENT_SHAPE,
        min_replica_count=2,
        max_replica_count=2,
        hot_load_bucket_type="FW_HOSTED",
    )
    mgr.create_or_get(config)
    mgr.wait_for_ready(DEPLOYMENT_ID, timeout_s=600)
    logger.info("READY")

    inference_model = f"accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)

    # Warmup
    logger.info("Warming up...")
    warmup_sampler = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
    )
    for attempt in range(10):
        try:
            result = asyncio.run(warmup_sampler.sample_with_tokens(
                messages=[{"role": "user", "content": "Say hi."}],
                n=1, max_tokens=8, temperature=0.0,
            ))
            if result:
                logger.info("Warmup done")
                break
        except Exception:
            time.sleep(5)
    warmup_sampler.close()

    missing = asyncio.run(run(BASE_URL, inference_model, tokenizer))

    # Cleanup
    logger.info("Cleaning up...")
    try:
        mgr.scale_to_zero(DEPLOYMENT_ID)
        mgr._delete(f"/v1/accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}?ignoreChecks=true&hard=true")
        logger.info("Deleted")
    except Exception as e:
        logger.warning("Cleanup: %s", e)
    mgr.close()

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
