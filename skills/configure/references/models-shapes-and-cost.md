# Models, training shapes & cost

*Source of truth: [Training shapes catalog](https://docs.fireworks.ai/fine-tuning/models.md) · [pricing](https://fireworks.ai/pricing) — these are generated live; always defer to them over anything written here.*

**Configure cost workflow** (when to estimate, manifest fields, progressive
disclosure): [`cost-estimation.md`](cost-estimation.md). **This file** owns
model/shape selection and the cost **formulas** below — keep rates live-linked,
not duplicated in the skill tree.

How to choose a base model + training shape, and reason about cost. **Always confirm against the live catalog and pricing page — model lists and prices change.**

## Training shapes

A **training shape** is a pre-configured GPU + model profile bundled into one reference ID, so you don't hand-author hardware. Picking a shape ID is usually the only shape-specific value you set.

Locked by the shape (not user-overridable): `acceleratorType`, `acceleratorCount`, `nodeCount`, `maxSupportedContextLength`, trainer image, linked deployment shape. You still set: `base_model`, `lora_rank`, `learning_rate`, `display_name`, replica counts.

**Pick one:** find your base model in the catalog → pick the shape matching your method (SFT/DPO/RFT) and approach (LoRA vs full-param); the per-model **Training method support** matrix shows combinations + total GPU + surfaces. Pass the full path, e.g. `accounts/fireworks/trainingShapes/qwen3p5-9b-256k`. Catalog (live): https://docs.fireworks.ai/fine-tuning/models.md

```python
service = FiretitanServiceClient.from_firetitan_config(
    api_key=api_key,
    base_model="accounts/fireworks/models/qwen3p5-9b",
    training_shape_id="accounts/fireworks/trainingShapes/qwen3p5-9b-256k",
    lora_rank=0,   # 0 = full-parameter; positive int (16, 64…) = LoRA
)
```

## Supported base models

Most major open families — **DeepSeek, Qwen, Kimi, GLM, Gemma, Llama, Nemotron, MiniMax** and more. Eligibility is specific to method, tuning mode, and training shape. Confirm SFT, DPO, ORPO, or RFT support in the live Training Shapes matrix before launch. The set is generated from the live registry — **don't hardcode it**. Live: [tunable text](https://app.fireworks.ai/models?filter=LLM&tunable=true) · [vision](https://app.fireworks.ai/models?filter=vision&tunable=true) · [method-support matrix](https://docs.fireworks.ai/fine-tuning/models.md)

## Context length & LoRA rank

Each shape exposes `max_supported_context_length`. A catalog-level model maximum may come from a shape that does not support the selected method or tuning mode. For a job, use the maximum context length of a compatible shape. Set `max_context_length` or inherit it from that shape. `lora_rank=0` means full-parameter; a positive integer means LoRA.

## GPU classes

The platform maps GPU class + count to the model — **let it**. Larger/MoE → more or larger accelerators + multi-node; small dense → fewer. Types across training/serving: **H200, B200, B300, GB300** (H100/A100 for some). The shape owns this.

## Pricing model

**Per training token** — managed SFT and DPO where listed. **Read current rates from the live [pricing page](https://fireworks.ai/pricing)** rather than trusting a hardcoded table.

**Runtime-priced resources** — managed RFT may be free for eligible small models (confirm on the live pricing page; do not assume a fixed size threshold). Other managed RFT and dedicated Training API resources follow current pricing. Dedicated trainers and deployments use GPU-hour rates and are metered by runtime.

**Inference:** serverless per-token; dedicated per GPU-hour.

### GPU-hour vs per-token + utilization
Naive per-token billing grows badly for multi-turn (every turn re-prefills). Fireworks dedicated uses session-affinity routing to reuse KV cache across turns. GPU-hour bills wall-clock GPU time regardless of throughput, so **utilization is the lever** — saturate the GPU. See [Multi-turn cost comparison](https://docs.fireworks.ai/fine-tuning/multi-turn-cost-comparison.md).

## Estimating cost before a run

Follow [`cost-estimation.md`](cost-estimation.md) for the full workflow, route
table, and output contract. This section covers shape/catalog context only.

- **Managed and Serverless training** — token-billed; formulas and meters live in
  `cost-estimation.md`. Read live [pricing](https://fireworks.ai/pricing).
- **Dedicated SFT or DPO** — **not calculated in the skill.** Send users to the
  [Training cost estimator](https://docs.fireworks.ai/fine-tuning/cost-estimator).
  Do not quote Dedicated $/M rates, GPU-hour training cost, or Dedicated vs
  Tinker comparisons from private throughput.
- **Managed RFT** — excluded from the estimator workflow.
- **Training API RL** — contact the [Training team](https://fireworks.ai/contact-training).
- **Eval inference and post-training deployment** — separate ancillary lines when
  present in the approved plan; use live inference rates. Deployment uptime can
  dominate small managed LoRA runs — recommend scale-to-zero/teardown.

## Critical rules

- **Let the platform map GPU → model** (don't override accelerator/count/node).
- **Use the full shape path** and let the SDK pin the version.
- **Use a compatible shape's context** — read the selected method and tuning mode's `max_supported_context_length`, not the model-level maximum.
- **Always check the live catalog + pricing** — generated from the live registry; they change.
- **Billing-aware:** use the current pricing page for the selected managed method or Training API infrastructure; label unknown rates rather than generalizing.
