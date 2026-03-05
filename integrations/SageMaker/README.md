## Fireworks AI on Amazon SageMaker

This directory contains example scripts to deploy Fireworks-powered LLM inference on Amazon SageMaker and test the endpoints.

For a broader integration guide, see the Fireworks documentation here: [Fireworks × SageMaker Integration](https://fireworks.ai/docs/ecosystem/integrations/sagemaker).

### Directory layout

```text
integrations/SageMaker/
  ├─ README.md                      # This document
  ├─ env_setup.sh                   # One-click local environment setup script
  ├─ deployment_scripts/
  │  ├─ deploy_multi_gpu_replicated.py # n replicas × 1 GPU each (fixed)
  │  ├─ deploy_multi_gpu_sharded.py    # n replicas × k GPUs each (model sharded)
  │  └─ deploy_spec_decode.py          # sharded with optional speculative decoding (using draft model)
  │
  └─ testing_scripts/
     └─ test_endpoint.py      # Uses AWS runtime client directly
```

### What’s included

- Deployment automation using the SageMaker SDK with sensible defaults and early validations:
  - Parses the AWS region from your ECR image URI (you can override with `--region`).
  - Verifies your S3 model bucket is in the same region as the deployment.
  - Sets CUDA forward-compatibility env vars to ease NVIDIA driver mismatches.

- Two deployment modes:
  - Multi-replica (replicated): multiple isolated replicas, each fixed at 1 GPU.
  - Multi-GPU (sharded): multiple replicas, each with k GPUs to shard large models.

- A lightweight test script:
  - `test_endpoint.py` can be used to test your endpoint is working correctly.

### Prerequisites

- Fireworks AI Docker image and metering key; please see [this guide](https://fireworks.ai/docs/ecosystem/integrations/sagemaker) for more information.
- An AWS account with permissions/quota to create SageMaker models/endpoints.
- An IAM role ARN for SageMaker (e.g., `arn:aws:iam::[YOUR_AWS_ACCOUNT_ID]:role/[ROLE_NAME]`).
- An ECR image containing the Fireworks inference container.
- An S3 URI to your model files.

### Quick environment setup

Use the helper script to prepare a local Python environment for running the deploy/test scripts:

```bash
cd integrations/SageMaker
bash env_setup.sh
# then activate when needed
source .venv/bin/activate
```

What the script does:
- Installs `uv` if not already available (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Creates a virtual environment `.venv` with Python 3.12 in `integrations/SageMaker/`.
- Activates the venv and installs `sagemaker` and `boto3`.

### Usage (deploy)

Multi-GPU, replicated (fixed 1 GPU per replica):

```bash
uv run integrations/SageMaker/deployment_scripts/deploy_multi_gpu_replicated.py \
  --s3-model-path s3://[BUCKET_NAME]/[PATH] \
  --ecr-image-uri [YOUR_AWS_ACCOUNT_ID].dkr.ecr.[YOUR_REGION].amazonaws.com/[IMAGE]:[TAG] \
  --sagemaker-role-arn arn:aws:iam::[YOUR_AWS_ACCOUNT_ID]:role/[ROLE_NAME] \
  --num-replicas 8
# Optional: --endpoint-name, --instance-type, --num-cpus-per-replica, --memory-per-replica, --max-batch-size, --region
```

Multi-GPU, sharded (k GPUs per replica):

```bash
uv run integrations/SageMaker/deployment_scripts/deploy_multi_gpu_sharded.py \
  --s3-model-path s3://[BUCKET_NAME]/[PATH] \
  --ecr-image-uri [YOUR_AWS_ACCOUNT_ID].dkr.ecr.[YOUR_REGION].amazonaws.com/[IMAGE]:[TAG] \
  --sagemaker-role-arn arn:aws:iam::[YOUR_AWS_ACCOUNT_ID]:role/[ROLE_NAME] \
  --num-replicas 2 \
  --num-gpus-per-replica 4
# Optional: --endpoint-name, --instance-type, --num-cpus-per-replica, --memory-per-replica, --max-batch-size, --region
```

Multi-GPU, sharded with speculative decoding (using draft model):

```bash
uv run integrations/SageMaker/deployment_scripts/deploy_spec_decode.py \
  --s3-model-path s3://[BUCKET_NAME]/[PATH] \
  --ecr-image-uri [YOUR_AWS_ACCOUNT_ID].dkr.ecr.[YOUR_REGION].amazonaws.com/[IMAGE]:[TAG] \
  --sagemaker-role-arn arn:aws:iam::[YOUR_AWS_ACCOUNT_ID]:role/[ROLE_NAME] \
  --num-replicas 2 \
  --num-gpus-per-replica 2 \
  --enable-speculation \
  --num-draft-tokens 3
# Optional: --endpoint-name, --instance-type, --num-cpus-per-replica, --memory-per-replica, --max-batch-size, --region, --draft-local-path
```

Notes:
- The scripts fail fast on cross‑region mismatches (e.g., ECR image in `us-west-2` but attempting to deploy in `us-east-1`).
- By default, the region is parsed from the ECR image URI; you can explicitly override with `--region`.
- The Fireworks inference engine can keep processing incoming requests asynchronously by default
- General heuristics for settings:
-- To improve performance for high-concurrency, high-throughput workloads, try increasing `--max-batch-size` and/or the number of replicas
-- Speculative decoding tends to have larger speedups on larger models (≈8B+)
-- Tune `--num-draft-tokens` (start with 2–4) to balance speedup and quality

### Usage (test)

```bash
uv run integrations/SageMaker/testing_scripts/test_endpoint.py \
  --endpoint-name [YOUR_ENDPOINT_NAME] \
  --region [YOUR_REGION]
```

### Learn more

For end‑to‑end guidance, visit the Fireworks documentation: [Fireworks × SageMaker Integration](https://fireworks.ai/docs/ecosystem/integrations/sagemaker).


