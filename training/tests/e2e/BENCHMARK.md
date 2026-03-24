# Streaming Inference Concurrency Benchmark

## Overview

Comparison of **fixed concurrency** (old approach: `asyncio.Semaphore(32)`) vs **adaptive concurrency** (new: `AdaptiveConcurrencyController` with batch-level AIMD) for RL training rollouts.

## Setup

- **Model**: `qwen3-30b-a3b-instruct-2507` (MoE, 30B params)
- **Deployment shape**: `rft-qwen3-30b-a3b-throughput` (2x NVIDIA H200 141GB, FP8)
- **Replicas**: 2 per deployment (separate deployments for each test)
- **Dataset**: DeepMath-Probability-Hard (200 rows, cycled)
- **Region**: US_VIRGINIA_1

## Test Parameters

| Parameter | Value |
|-----------|-------|
| Batches | 32 |
| Prompts per batch | 32 |
| Completions per prompt | 8 |
| Requests per batch | 256 |
| Total requests | 8,192 |
| Max tokens | 131,072 (128K) |
| Temperature | 0.7 |

## How to Run

Both tests run in parallel on separate deployments to avoid interference.

### Prerequisites

```bash
# Install SDK from source
cd fireworks-ai-python && pip install -e .

# Create deployments
python -c "
from fireworks.training.sdk.deployment import DeploymentManager, DeploymentConfig
mgr = DeploymentManager(api_key='YOUR_KEY', base_url='https://api.fireworks.ai')
for name in ['bench-fixed', 'bench-adaptive']:
    config = DeploymentConfig(
        deployment_id=name,
        base_model='accounts/fireworks/models/qwen3-30b-a3b-instruct-2507',
        deployment_shape='accounts/fireworks/deploymentShapes/rft-qwen3-30b-a3b-throughput/versions/ai5r0aoy',
        min_replica_count=2, max_replica_count=2,
        hot_load_bucket_type='FW_HOSTED',
    )
    mgr.create_or_get(config)
    mgr.wait_for_ready(name, timeout_s=600)
    print(f'{name} READY')
mgr.close()
"
```

### Run Benchmarks in Parallel

```bash
cd cookbook

# Terminal 1: Fixed concurrency
FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
BENCH_DEPLOYMENT_ID=e2e-bench-fixed-chengxili-v1 \
  python training/tests/e2e/bench_fixed.py

# Terminal 2: Adaptive concurrency
FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
BENCH_DEPLOYMENT_ID=e2e-bench-adaptive-chengxili-v1 \
  python training/tests/e2e/bench_adaptive.py
```

Results are written to `bench_fixed_results.json` and `bench_adaptive_results.json`.

## Metrics Collected

| Metric | Description |
|--------|-------------|
| **Success rate** | `completions / expected * 100` |
| **Avg batch time** | Mean wall-clock time per batch (256 requests) |
| **Median batch time** | Median wall-clock time per batch |
| **Avg response length** | Mean completion length in tokens across all batches |
| **Median response length** | Median completion length per batch |
| **Throughput** | Total completions / total time |
| **First response per batch** | Text preview of the first completion in each batch |
| **Window (adaptive only)** | Concurrency window size after each batch |
| **Avg prefill queue (adaptive only)** | Mean `prefill-queue-duration` per batch |
| **Cache hit rate (adaptive only)** | `cached-prompt-tokens / prompt-tokens` per batch |

## Results

> Results will be filled in after running the benchmark.

### Fixed Concurrency (max_concurrency=32)

| Metric | Value |
|--------|-------|
| Total completions | |
| Success rate | |
| Total time | |
| Avg batch time | |
| Median batch time | |
| Avg response length | |
| Throughput | |

### Adaptive Concurrency (initial_window=16)

| Metric | Value |
|--------|-------|
| Total completions | |
| Success rate | |
| Total time | |
| Avg batch time | |
| Median batch time | |
| Avg response length | |
| Throughput | |
| Final window | |
| Final EMA prefill queue | |
| Window range | |

### Key Differences

| Aspect | Fixed | Adaptive |
|--------|-------|----------|
| **Concurrency control** | Static semaphore (32) | AIMD based on `prefill_queue_duration` |
| **Inference mode** | Streaming (SSE) | Streaming (SSE) |
| **Server metrics** | Not used | `perf_metrics` from final SSE chunk |
| **Window adjustment** | Never | Between batches via `step_completed()` |
| **Observability** | None | `avg_pq`, `cache_hit_rate`, `window` logged per batch |
