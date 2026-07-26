# 5. GPUs and distributed systems

You do not need to write CUDA to work on this stack, but you do need a mental
model of why some things are fast and others are not. Nearly every performance
question reduces to one of two facts: *matrix multiplies are cheap, memory
traffic is not*, and *communication is the tax on parallelism*.

## 5.1 What a GPU is

A CPU is optimized for latency on one thread. A GPU is optimized for throughput
across tens of thousands.

- **SM (streaming multiprocessor)** — the unit of scheduling. An H100 has 132.
  Each runs many **warps** (groups of 32 threads executing in lockstep).
- **CUDA cores** — general FP32/INT arithmetic.
- **Tensor cores** — dedicated matrix-multiply-accumulate units. They are where
  essentially all of a transformer's advertised FLOPs live; anything that is not
  a matmul is running at a small fraction of peak.
- **Memory hierarchy** — the important part:

| Level | Size | Bandwidth | Latency |
|---|---|---|---|
| Registers | ~256 KB/SM | ~20 TB/s | ~1 cycle |
| Shared memory / L1 | ~228 KB/SM | ~10 TB/s | ~30 cycles |
| L2 | ~50 MB | ~5 TB/s | ~200 cycles |
| HBM (global) | 80–192 GB | 3–8 TB/s | ~400–800 cycles |
| Host RAM over PCIe | TB | ~64 GB/s | very slow |

Two orders of magnitude separate registers from HBM. Kernel optimization is
mostly the art of touching HBM as few times as possible — which is precisely
what FlashAttention (§1.3) does, and what kernel *fusion* does in general:
combining `x → norm → matmul → activation` into one kernel so intermediates never
round-trip to HBM.

## 5.2 The roofline model

Everything above collapses into one picture. For a kernel doing $F$ FLOPs while
moving $B$ bytes, arithmetic intensity is $I = F/B$. Achievable performance is

$$\text{FLOP/s} = \min\big(\text{peak FLOP/s},\; I \times \text{bandwidth}\big)$$

The crossover — machine balance — is $\text{peak} / \text{bandwidth}$. For an
H100 that is roughly $990\,\text{TFLOP/s} \div 3.35\,\text{TB/s} \approx 300$
FLOP/byte. Below that intensity you are memory-bound; above it, compute-bound.

Where common operations land:

| Operation | Intensity | Verdict |
|---|---|---|
| Large GEMM (prefill, training) | ~1000s | Compute-bound — good |
| Decode at batch 1 | ~1 | Hopelessly memory-bound |
| Decode at batch 256 | ~256 | Approaching balance |
| RMSNorm, softmax, residual add | <10 | Memory-bound → must be fused |
| All-reduce | 0 | Pure communication |

This one table explains continuous batching, speculative decoding, kernel fusion,
and disaggregation all at once. Reread §2.1 with it in mind.

## 5.3 Precision formats

| Format | Bits (E/M) | Range | Where used |
|---|---|---|---|
| FP32 | 8/23 | huge | Master weights, optimizer moments, reductions |
| TF32 | 8/10 | FP32 range | Transparent tensor-core FP32 replacement |
| **BF16** | 8/7 | FP32 range | The training default |
| FP16 | 5/10 | narrow | Legacy; needs loss scaling |
| FP8 E4M3 | 4/3 | small | Forward activations/weights on Hopper+ |
| FP8 E5M2 | 5/2 | larger | Gradients (need range more than precision) |
| FP4 (NVFP4/MXFP4) | — | tiny | Blackwell inference; relies on block scales |

Rule of thumb: **tensor-core throughput roughly doubles each time you halve the
bit width.** That is the entire economic argument for low precision, and it is
also the source of the train–inference numerical gap in §4.8 — the inference
engine takes the FP8/FP4 deal, the trainer usually does not.

Two things stay in FP32 regardless: accumulation inside a matmul, and the
optimizer state (§3.4).

## 5.4 Accelerators you will see in this stack

Approximate figures — always check the live catalog and pricing rather than
trusting numbers written down anywhere, including here.

| GPU | HBM | Bandwidth | Notes |
|---|---|---|---|
| A100 | 40/80 GB | ~2.0 TB/s | Ampere; no FP8 |
| H100 | 80 GB | ~3.35 TB/s | Hopper; FP8; ~990 TFLOP/s BF16 dense |
| H200 | 141 GB | ~4.8 TB/s | Same compute as H100, much more memory → bigger KV cache |
| B200 | 192 GB | ~8 TB/s | Blackwell; FP4; roughly 2× H100 BF16 |
| B300 | ~288 GB | higher | Blackwell Ultra |
| GB300 | rack-scale | NVL72 domain | Grace CPU + Blackwell, 72 GPUs in one NVLink domain |

The H100 → H200 jump is instructive: **identical compute, 76% more memory
bandwidth and capacity.** For decode that is nearly a 1.4× throughput win purely
from memory, and it enables much longer context at the same batch size. Memory,
not FLOPs, is usually the binding constraint for serving.

Interconnect matters as much as the chips:

- **NVLink / NVSwitch** — intra-node GPU-to-GPU, ~900 GB/s per GPU on H100.
- **InfiniBand / RoCE** — inter-node, ~400 Gb/s (50 GB/s) per NIC.

That is an order of magnitude difference, and it dictates parallelism placement
(§5.7): chatty parallelism strategies must stay inside a node.

## 5.5 Collectives

Distributed training is built on a few collective primitives: `all-reduce` (sum
across ranks, everyone gets the result), `all-gather`, `reduce-scatter`,
`all-to-all` (the MoE expert-routing pattern).

Ring all-reduce of $S$ bytes over $P$ ranks moves

$$2\,\frac{P-1}{P}\,S \ \text{bytes per rank} \;\approx\; 2S$$

so its time is essentially $2S/\text{bandwidth}$, independent of $P$ — which is
why data parallelism scales well. The practical trick is **overlap**: start
reducing layer $\ell$'s gradients while layer $\ell-1$ is still computing its
backward, hiding communication behind compute.

## 5.6 Sharding the training state: ZeRO/FSDP

From §3.5, full-parameter AdamW costs ~16 bytes per parameter. ZeRO shards it
across $P$ data-parallel ranks in stages:

| Stage | Sharded | Bytes/param/GPU |
|---|---|---|
| 0 (plain DDP) | nothing | 16 |
| 1 | optimizer state | $4 + 12/P$ |
| 2 | + gradients | $2 + 14/P$ |
| **3 / FSDP** | + parameters | $16/P$ |

Stage 3 (PyTorch's FSDP) keeps only a shard of each layer resident and
`all-gather`s the full layer just before using it, then frees it. Memory becomes
$O(1/P)$; the cost is an extra all-gather per layer per forward and per backward.

**HSDP (hybrid sharded data parallel)** is the practical compromise: shard
*within* a node (over fast NVLink) and *replicate* across nodes (so inter-node
traffic is one all-reduce per step instead of per-layer all-gathers). This is
what `TrainerConfig.replica_count` refers to in the cookbook — the number of
data-parallel replicas of the sharded group.

## 5.7 The parallelism zoo

| Strategy | Splits | Communication | Placement |
|---|---|---|---|
| **Data (DP)** | the batch | all-reduce of gradients per step | Across nodes |
| **Tensor (TP)** | individual matrices | all-reduce **twice per layer** | Inside a node only |
| **Pipeline (PP)** | layer groups | point-to-point activations | Across nodes; needs micro-batches to fill the bubble |
| **Expert (EP)** | MoE experts | all-to-all per MoE layer | Depends on expert count |
| **Context/Sequence (CP/SP)** | the sequence | ring attention passes | Long-context training |

Real jobs compose these ("4D parallelism"). The placement rule follows from
§5.4: TP is the chattiest, so it goes on NVLink inside a node; PP and DP tolerate
slower links and go across nodes.

PP has a distinctive failure mode, the **bubble**: with $S$ stages and $M$
micro-batches, the fraction of time stages sit idle is $(S-1)/(M+S-1)$. You need
$M \gg S$ to amortize it, which is one reason pipeline parallelism interacts
badly with small batches.

## 5.8 Why "training shapes" exist

Choosing accelerator type, count, node count, parallelism degrees, and context
limits for a given model is expert work, and getting it wrong wastes money in
ways that are hard to notice. Fireworks packages the answer as a **training
shape** — one ID that pins the hardware and topology:

```16:21:skills/fireworks-training/references/models-shapes-and-cost.md
service = FiretitanServiceClient.from_firetitan_config(
    api_key=api_key,
    base_model="accounts/fireworks/models/qwen3p5-9b",
    training_shape_id="accounts/fireworks/trainingShapes/qwen3p5-9b-256k",
    lora_rank=0,   # 0 = full-parameter; positive int (16, 64…) = LoRA
)
```

The shape owns `acceleratorType`, `acceleratorCount`, `nodeCount`,
`maxSupportedContextLength`, the trainer image, and the linked deployment shape.
You own `base_model`, `lora_rank`, `learning_rate`, and replica counts. The
cookbook actively deprecates manual overrides:

```154:157:training/utils/config.py
    accelerator_type: str | None = None
    """Deprecated and ignored. Trainer accelerator type is owned by the
    training shape; setting this emits a ``DeprecationWarning``. Use
    ``replica_count`` for data-parallel scaling."""
```

The same philosophy applies to region: leave it unset and let the backend place
the trainer and its deployment together, rather than pinning a region and
accidentally splitting a colocated pair across a network boundary
([`CLAUDE.md`](../CLAUDE.md)).

## 5.9 Back-of-envelope cost

Two formulas cover most planning questions.

**Training compute** (from §1.9):

$$\text{GPU-hours} \approx \frac{6 N \cdot T_\text{tokens}}{\text{peak FLOP/s} \times \text{MFU} \times 3600}$$

with MFU (model FLOPs utilization) realistically 0.3–0.5 for a well-tuned
full-parameter run.

**Serving throughput** (from §2.1–2.2):

$$\text{tokens/s} \approx \frac{B}{\text{decode step time}}, \qquad \text{decode step time} \gtrsim \frac{\text{model bytes} + B \cdot \text{KV bytes read}}{\text{HBM bandwidth}}$$

and $B$ itself is capped by KV cache capacity:
$B_\text{max} \approx (\text{HBM} - \text{weights}) / (\text{KV bytes/token} \times n)$.

The operational punchline from
[`models-shapes-and-cost.md`](../skills/fireworks-training/references/models-shapes-and-cost.md):
**for dedicated deployments the dominant cost is usually uptime, not tokens.** A
small LoRA SFT job costs cents; a GPU left running for a week costs hundreds to
thousands of dollars. Scale to zero, and tear down deployments you are not using.

---

**Next:** [6. The Fireworks stack map](fireworks-stack-map.md) — where all of this
lives in the code.
