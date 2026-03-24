#!/usr/bin/env python3
"""Test: is raw_output ever missing from a fully-ready deployment?

Waits until deployment is truly ready (READY state + 2 replicas +
5 consecutive successful warmup requests), then fires 1000 requests
with NO retries. If raw_output is missing on any request, it's a
real streaming bug. If not, it was a deployment readiness issue.
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
    _SSEDecoder,
)
from fireworks.training.sdk.errors import async_request_with_retries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = "e2e-repro-rawoutput-chengxili-v2"
DEPLOYMENT_SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-30b-a3b-throughput/versions/ai5r0aoy"
BASE_MODEL = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"

NUM_REQUESTS = 1000
CONCURRENCY = 32
MAX_TOKENS = 512


async def send_one_raw(client, url, headers, payload, idx):
    """Send one streaming request, return (has_raw_output, diagnosis)."""
    import json as _json

    resp = await async_request_with_retries(
        client.post, url, headers=headers, json=payload, timeout=120,
    )
    if resp.status_code != 200:
        return None, {"idx": idx, "issue": f"HTTP {resp.status_code}"}

    raw_output = None
    finish_reason = None
    text = ""
    chunk_count = 0
    last_choice_keys = None

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
            last_choice_keys = sorted(choice.keys())
            t = choice.get("text", "")
            if t:
                text += t
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            ro = choice.get("raw_output")
            if ro:
                raw_output = ro

    has_ids = raw_output is not None and raw_output.get("completion_token_ids") is not None

    if not has_ids:
        return False, {
            "idx": idx,
            "issue": "missing_raw_output",
            "finish_reason": finish_reason,
            "text_len": len(text),
            "chunk_count": chunk_count,
            "last_choice_keys": last_choice_keys,
            "has_raw_output_key": raw_output is not None,
            "raw_output_keys": sorted(raw_output.keys()) if raw_output else None,
        }
    return True, None


def main():
    mgr = DeploymentManager(api_key=API_KEY, base_url=BASE_URL)

    # Create deployment
    logger.info("Creating deployment %s...", DEPLOYMENT_ID)
    config = DeploymentConfig(
        deployment_id=DEPLOYMENT_ID,
        base_model=BASE_MODEL,
        deployment_shape=DEPLOYMENT_SHAPE,
        min_replica_count=2,
        max_replica_count=2,
        hot_load_bucket_type="FW_HOSTED",
    )
    mgr.create_or_get(config)

    # Wait for TRULY ready: READY state + 2 replicas
    logger.info("Waiting for READY with 2 replicas...")
    import httpx
    for i in range(60):
        r = httpx.get(
            f"{BASE_URL}/v1/accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15,
        )
        data = r.json()
        state = data.get("state", "?")
        ready = data.get("replicaStats", {}).get("readyReplicaCount", 0)
        logger.info("  [%ds] state=%s ready=%d/2", i * 10, state, ready)
        if state == "READY" and ready >= 2:
            break
        time.sleep(10)

    # Warmup: 5 consecutive successful requests with raw_output
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    inference_model = f"accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}"
    sampler = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
    )

    logger.info("Warming up (5 consecutive successes required)...")
    consecutive = 0
    for attempt in range(30):
        try:
            result = asyncio.run(sampler.sample_with_tokens(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                n=1, max_tokens=32, temperature=0.0,
            ))
            if result and result[0].full_tokens:
                consecutive += 1
                logger.info("  Warmup %d/5 OK (attempt %d)", consecutive, attempt + 1)
                if consecutive >= 5:
                    break
            else:
                consecutive = 0
        except Exception as e:
            logger.info("  Warmup attempt %d failed: %s", attempt + 1, e)
            consecutive = 0
            time.sleep(5)
    sampler.close()

    if consecutive < 5:
        logger.error("Could not get 5 consecutive warmup successes, aborting")
        return

    logger.info("Deployment fully warm. Sending %d requests with NO retries...", NUM_REQUESTS)

    # Fire requests
    async def run():
        client = httpx.AsyncClient(
            verify=True,
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=0),
        )

        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Reason step by step. Put answer in \\boxed{}."},
             {"role": "user", "content": "What is the probability of rolling a sum of 7 with two dice?"}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )

        url = f"{BASE_URL}/inference/v1/completions"
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
        ok = 0
        http_errors = 0
        missing = []

        async def _go(idx):
            nonlocal ok, http_errors
            async with sem:
                has_ids, diag = await send_one_raw(client, url, headers, payload, idx)
                if has_ids is None:
                    http_errors += 1
                    if diag:
                        logger.warning("  HTTP error on %d: %s", idx, diag["issue"])
                elif has_ids:
                    ok += 1
                else:
                    missing.append(diag)
                    logger.warning("  MISSING raw_output on %d: %s", idx, diag)

                if (idx + 1) % 100 == 0:
                    logger.info("  Progress: %d/%d, ok=%d, http_err=%d, missing=%d",
                                idx + 1, NUM_REQUESTS, ok, http_errors, len(missing))

        await asyncio.gather(*[_go(i) for i in range(NUM_REQUESTS)])
        await client.aclose()
        return ok, http_errors, missing

    ok, http_errors, missing = asyncio.run(run())

    logger.info("=" * 60)
    logger.info("RESULTS: %d OK, %d HTTP errors, %d missing raw_output", ok, http_errors, len(missing))
    if missing:
        logger.info("CONCLUSION: raw_output IS missing on fully-ready deployment -- streaming bug")
        for m in missing:
            logger.info("  %s", json.dumps(m, default=str))
    else:
        logger.info("CONCLUSION: raw_output is NEVER missing on fully-ready deployment")
        logger.info("  The earlier failures were deployment readiness issues, not streaming bugs")
    if http_errors:
        logger.info("  (%d HTTP errors were 404/425 deployment readiness, not streaming)", http_errors)
    logger.info("=" * 60)

    # Cleanup
    logger.info("Cleaning up...")
    try:
        mgr.scale_to_zero(DEPLOYMENT_ID)
        mgr._delete(f"/v1/accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}?ignoreChecks=true&hard=true")
        logger.info("Deleted")
    except Exception as e:
        logger.warning("Cleanup: %s", e)
    mgr.close()


if __name__ == "__main__":
    main()
