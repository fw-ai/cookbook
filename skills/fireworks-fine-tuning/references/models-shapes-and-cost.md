# Models, training shapes & cost

*Source of truth: [Training shapes catalog](https://docs.fireworks.ai/fine-tuning/training-api/training-shapes.md) · [pricing](https://fireworks.ai/pricing) — these are generated live; always defer to them over anything written here.*

How to choose a base model + training shape, and reason about cost. **Always confirm against the live catalog and pricing page — model lists and prices change.**

## Training shapes

A **training shape** is a pre-configured GPU + model profile bundled into one reference ID, so you don't hand-author hardware. Picking a shape ID is usually the only shape-specific value you set.

Locked by the shape (not user-overridable): `acceleratorType`, `acceleratorCount`, `nodeCount`, `maxSupportedContextLength`, trainer image, linked deployment shape. You still set: `base_model`, `lora_rank`, `learning_rate`, `display_name`, replica counts.

**Pick one:** find your base model in the catalog → pick the shape matching your method (SFT/DPO/RFT) and approach (LoRA vs full-param); the per-model **Training method support** matrix shows combinations + total GPU + surfaces. Pass the full path, e.g. `accounts/fireworks/trainingShapes/qwen3p5-9b-256k`. Catalog (live): https://docs.fireworks.ai/fine-tuning/training-api/training-shapes.md

```python
service = FiretitanServiceClient.from_firetitan_config(
    api_key=api_key,
    base_model="accounts/fireworks/models/qwen3p5-9b",
    training_shape_id="accounts/fireworks/trainingShapes/qwen3p5-9b-256k",
    lora_rank=0,   # 0 = full-parameter; positive int (16, 64…) = LoRA
)
```

## Supported base models

Most major open families — **DeepSeek, Qwen, Kimi, GLM, Gemma, Llama, Nemotron, MiniMax** and more. Once supported, every managed method (SFT/DPO/RFT) works. The set is generated from the live registry — **don't hardcode it**. Live: [tunable text](https://app.fireworks.ai/models?filter=LLM&tunable=true) · [vision](https://app.fireworks.ai/models?filter=vision&tunable=true) · [context table](https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro.md#supported-base-models)

## Context length & LoRA rank

**We support the model's full context.** Each shape exposes `max_supported_context_length` (up to 256K / 262,144 for Gemma-4 / Kimi / Qwen3.5 families; smaller models less). Per-job: set `max_context_length` (or inherit from the shape). `lora_rank=0` → full-param; positive int → LoRA.

## GPU classes

The platform maps GPU class + count to the model — **let it**. Larger/MoE → more or larger accelerators + multi-node; small dense → fewer. Types across training/serving: **H200, B200, B300, GB300** (H100/A100 for some). The shape owns this.

## Pricing model

**Per training token** — SFT & DPO, all sizes; tiered by base-model size (per 1M training tokens). **Read current rates from the live [pricing page](https://fireworks.ai/pricing)** — don't trust a hardcoded table. Shape of it: smaller models cost less; full-param > LoRA; DPO > SFT. RFT is **free under 16B**.

**Per GPU-hour** — full-param/RFT trainer time + dedicated deployments (on-demand, GPU-class rates). Again, [pricing page](https://fireworks.ai/pricing) is the source of truth.

**Inference:** serverless per-token; dedicated per GPU-hour.

### GPU-hour vs per-token + utilization
Naive per-token billing grows badly for multi-turn (every turn re-prefills). Fireworks dedicated uses session-affinity routing to reuse KV cache across turns. GPU-hour bills wall-clock GPU time regardless of throughput, so **utilization is the lever** — saturate the GPU. See [Multi-turn cost comparison](https://docs.fireworks.ai/fine-tuning/multi-turn-cost-comparison.md).

## Estimating cost before a run

Estimate the spend surface before protected work; label uncertainty rather than implying false precision, and link the [pricing page](https://fireworks.ai/pricing) instead of hardcoding rates:
- **Training** scales with base model, billing mode, train tokens, and epochs, times the number of candidate runs (each hyperparameter-grid cell is a separate run). LoRA SFT is token-billed.
- **Inference / eval** scales with model, sample count, average input + output tokens, and number of passes.
- **Deployment** scales with the shape/accelerator, replica count, and uptime; on-demand bills per GPU-second until torn down (include any validation-deployment uptime).
- If a required rate or shape is unknown, label that line unknown rather than guessing.

## Critical rules

- **Let the platform map GPU → model** (don't override accelerator/count/node).
- **Use the full shape path** and let the SDK pin the version.
- **Use the model's full context** — read `max_supported_context_length`, don't assume a stale cap.
- **Always check the live catalog + pricing** — generated from the live registry; they change.
- **Billing-aware:** SFT/DPO = per training token; full-param/RFT + deployments = per GPU-hour; saturate the GPU.
