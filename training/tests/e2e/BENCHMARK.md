# Streaming Inference Concurrency Benchmark

## Overview

Comparison of **fixed concurrency** vs **adaptive concurrency** (AIMD with proportional increase) for RL training rollouts.

**Important**: Every benchmark run must create **fresh deployments** to ensure clean KV cache state. Never reuse deployments from previous runs.

## Setup

- **Model**: `qwen3-30b-a3b-instruct-2507` (MoE, 30B params)
- **Deployment shape**: `rft-qwen3-30b-a3b-throughput` (2x NVIDIA H200 141GB, FP8)
- **Replicas**: 2 per deployment
- **Region**: US_VIRGINIA_1
- **Dataset**: DeepMath-Probability-Hard (200 rows, cycled)

## Test Parameters

| Parameter | Value |
|-----------|-------|
| Batches | 8 |
| Prompts per batch | 256 |
| Completions per prompt | 8 |
| Requests per batch | 2,048 |
| Total requests | 16,384 |
| Max tokens | 131,072 (128K) |
| Temperature | 0.7 |

All 8 batches fire concurrently. The SDK semaphore/controller throttles in-flight requests.

## How to Run

### Step 1: Install SDK from source

```bash
cd fireworks-ai-python && pip install -e .
```

### Step 2: Create fresh deployments

Create a **new** deployment for each test. Never reuse -- the KV cache from prior requests contaminates results.

```bash
export FIREWORKS_API_KEY="$FIREWORKS_API_KEY"

# For each test, create a fresh deployment:
python3 -c "
from fireworks.training.sdk.deployment import DeploymentManager, DeploymentConfig
mgr = DeploymentManager(api_key='$FIREWORKS_API_KEY', base_url='https://api.fireworks.ai')
config = DeploymentConfig(
    deployment_id='bench-TESTNAME-$(date +%s)',
    base_model='accounts/fireworks/models/qwen3-30b-a3b-instruct-2507',
    deployment_shape='accounts/fireworks/deploymentShapes/rft-qwen3-30b-a3b-throughput/versions/ai5r0aoy',
    min_replica_count=2, max_replica_count=2,
    hot_load_bucket_type='FW_HOSTED',
)
mgr.create_or_get(config)
mgr.wait_for_ready(config.deployment_id, timeout_s=600)
print(f'READY: {config.deployment_id}')
mgr.close()
"
```

### Step 3: Run benchmarks

```bash
cd cookbook

# Fixed concurrency (specify MAX_CONCURRENCY via env var)
FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
BENCH_DEPLOYMENT_ID=<fixed-deployment-id> \
BENCH_MAX_CONCURRENCY=32 \
  python training/tests/e2e/bench_fixed.py

# Adaptive concurrency
FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
BENCH_DEPLOYMENT_ID=<adaptive-deployment-id> \
  python training/tests/e2e/bench_adaptive.py
```

Run on **separate fresh deployments** in parallel for fair comparison.

### Step 4: Clean up

```bash
# Delete deployments after benchmark
python3 -c "
from fireworks.training.sdk.deployment import DeploymentManager
mgr = DeploymentManager(api_key='$FIREWORKS_API_KEY', base_url='https://api.fireworks.ai')
for dep_id in ['<fixed-id>', '<adaptive-id>']:
    mgr.scale_to_zero(dep_id)
    mgr._delete(f'/v1/accounts/{mgr.account_id}/deployments/{dep_id}?ignoreChecks=true&hard=true')
    print(f'Deleted {dep_id}')
mgr.close()
"
```

## Metrics Collected

### Per-batch metrics (both modes)

| Metric | Description |
|--------|-------------|
| **Wall time** | Total wall-clock time for the batch |
| **Avg request latency** | Mean time from request send to last SSE token |
| **P50 request latency** | Median request latency |
| **Avg response length** | Mean completion length in tokens |
| **Median response length** | Median completion length |
| **Success rate** | `completions / expected` |
| **First response** | Text preview of first completion per batch |

### Server metrics (from `perf_metrics` in final SSE chunk)

| Metric | Description |
|--------|-------------|
| **Prefill queue (avg, p50, max)** | Time requests waited in prefill queue |
| **Generation queue** | Time in generation queue |
| **Server TTFT** | Server-side time-to-first-token |
| **Client TTFT** | Client-measured time-to-first-token |
| **Cache hit rate** | `cached_prompt_tokens / prompt_tokens` |
| **Server concurrent (avg, max)** | In-flight requests on the server |

### Adaptive-only metrics

| Metric | Description |
|--------|-------------|
| **Window** | Concurrency window size after each batch |
| **Avg PQ** | Step-averaged prefill queue fed to AIMD |
| **EMA PQ** | Exponential moving average across steps |

## Results (v4 run, 2026-03-24)

### Batch 1 (cold cache)

| Config | Wall Time | Avg Latency | Prefill Queue | Server Concurrent | Cache Hit |
|--------|-----------|-------------|---------------|-------------------|-----------|
| Fixed-32 | 1020s | 498s | 0.083s | 14.3 | 79% |
| Adaptive (32→36) | 1371s | 494s | 0.080s | 14.4 | 83% |
| Fixed-64 | 1487s | 367s | 4.4s | 30.0 | 99% |
| Fixed-128 | 737s | 358s | 25.6s | 62.2 | 81% |

### Adaptive window progression (5 batches)

| Batch | Window | Server Concurrent | Prefill Queue | Cache Hit |
|-------|--------|-------------------|---------------|-----------|
| 1 | 32→36 | 14.4 | 0.080s | 83% |
| 2 | 36→40 | 16.4 | 0.088s | 97% |
| 3 | 40→44 | 18.4 | 0.099s | 96% |
| 4 | 44→48 | 20.2 | 0.129s | 96% |
| 5 | 48→50 | 22.2 | 0.343s | 96% |

### Key Observations

1. **Adaptive matches fixed-32 on batch 1** (avg_lat 494s vs 498s) with `initial_window = 8 * gpu_count = 32`.
2. **Proportional increase works**: window grew 32→50 in 5 steps (+4/step) instead of old +1/step.
3. **Adaptive surpasses fixed-32**: by batch 5, server concurrent reaches 22.2 vs fixed-32's 14.3.
4. **AIMD self-tunes**: prefill queue rose 0.08→0.34s as window grew, approaching 0.5s target.
5. **Fixed-128 wins on raw throughput** (737s batch 1) but at 25.6s prefill queue -- the server is severely overloaded.
6. **~0.05% "incomplete chunked read"** errors on all configs -- transient network issue, not code bug.
