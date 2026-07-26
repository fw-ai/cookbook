# 2. Inference: KV caches, batching, speculation, disaggregation

Serving an LLM is mostly a **memory-movement** problem. The math from part 1 is
cheap; moving weights and cached state through the memory hierarchy is not. Every
technique in this file exists to move fewer bytes, or to get more useful work out
of the bytes you already moved.

## 2.1 Two phases with opposite bottlenecks

Generating a response has two distinct stages:

**Prefill** — process the whole prompt ($n$ tokens) in one forward pass. All
positions are known, so this is a batch of big matrix multiplies. It is
**compute-bound**: you do $2N \cdot n$ FLOPs while reading the weights once.
Latency here is **TTFT** (time to first token).

**Decode** — generate tokens one at a time. Each step processes exactly one new
token but must read *every weight in the model* to do it. It is
**memory-bandwidth-bound**: one token of work for gigabytes of weight traffic.
Latency here is **TPOT/ITL** (time per output token / inter-token latency).

A single formula makes the asymmetry obvious. Define **arithmetic intensity** as
FLOPs performed per byte read. For decode at batch size $B$ with an $N$-parameter
model in 2-byte precision:

$$\text{FLOPs} \approx 2NB, \qquad \text{bytes} \approx 2N \ (\text{weights, read once regardless of } B)$$
$$\text{intensity} \approx B$$

Modern accelerators need intensity in the hundreds to saturate their tensor
cores. At $B = 1$ you are using well under 1% of the FLOPs. **This single fact
motivates continuous batching, speculative decoding, and disaggregation.**

Concretely: an 8B model in bf16 is 16 GB. An H200 has ~4.8 TB/s of HBM
bandwidth, so the floor on one decode step is $16/4800 \approx 3.3$ ms → about
300 tokens/s, *no matter how fast the GPU's math units are*. The only way to do
better per GPU is to serve more sequences with those same weight reads, or to
produce more than one token per weight read.

## 2.2 The KV cache

During decode at step $t$, attention for the new token needs $k_j$ and $v_j$ for
all $j < t$. Because of causal masking, those tensors **do not change** when new
tokens are appended — position 5's key is the same whether the sequence is 6 or
6000 tokens long. So you compute them once and keep them.

Without a cache, generating $n$ tokens re-runs the full prefix each step:
$O(n^2)$ total work. With a cache, each step is $O(1)$ model work plus an
$O(t)$ attention read. The KV cache is not an optimization detail; it is what
makes autoregressive generation tractable.

### How big is it?

Per token, you store one key and one value vector per KV head per layer:

$$\boxed{\text{bytes/token} = 2 \times L \times g \times d_h \times \text{bytes\_per\_element}}$$

where $g$ is the number of **KV** heads (§1.5). Worked examples in bf16:

| Model | $L$ | $g$ | $d_h$ | Bytes/token | 32k context | 32k × 64 concurrent |
|---|---|---|---|---|---|---|
| 8B-class (GQA) | 32 | 8 | 128 | 128 KB | 4.2 GB | 268 GB |
| 70B-class (GQA) | 80 | 8 | 128 | 320 KB | 10.5 GB | 671 GB |
| 70B with MHA (hypothetical) | 80 | 64 | 128 | 2.5 MB | 84 GB | — |

Read that table again: at long context and real concurrency, **the KV cache is
larger than the model weights**. It, not the parameter count, sets your maximum
batch size, and maximum batch size sets your throughput and therefore your cost
per token. Everything follows from here:

- **GQA/MLA** (§1.5) shrink $g$ or replace it with a latent.
- **KV quantization** to FP8 or INT8 halves it again, at some accuracy risk.
- **Paged allocation** stops you wasting it.
- **Prefix reuse** stops you rebuilding it.

### PagedAttention

Naively you reserve a contiguous buffer per request sized to `max_tokens`. Most
requests finish early, so you waste most of it, and the leftover holes are
unusable (external fragmentation). **PagedAttention** borrows virtual memory:
split the cache into fixed-size blocks (e.g. 16 tokens), keep a per-sequence
block table, and let a sequence's blocks live anywhere in HBM. Waste drops to
under one block per sequence, and blocks become *shareable* — several beams or
several requests with the same prefix can point at the same physical block with a
reference count.

### Prefix caching and session affinity

If two requests share a prefix — a long system prompt, a few-shot preamble, or
the earlier turns of a conversation — the KV blocks for that prefix are
identical, and the second request can skip prefilling them entirely.

This is a big deal for **multi-turn** workloads (agents, tool use, RL rollouts).
Turn $k$ of a conversation contains all of turns $1..k-1$, so naive per-token
billing re-prefills the whole history every turn: cost grows quadratically in the
number of turns. Fireworks dedicated deployments use **session-affinity
routing** — route a conversation's turns back to the replica that already holds
its KV blocks — so the history is reused rather than recomputed. This is called
out in
[`skills/fireworks-training/references/models-shapes-and-cost.md`](../skills/fireworks-training/references/models-shapes-and-cost.md)
as one of the main reasons dedicated GPU-hour pricing can beat per-token pricing
for agentic workloads.

## 2.3 Continuous batching

Static batching — collect $B$ requests, run them together, wait for all to finish
— wastes enormous capacity, because generation lengths vary by an order of
magnitude and the whole batch runs until its slowest member is done.

**Continuous batching** (a.k.a. iteration-level scheduling) makes the scheduling
decision at every decode step instead: when a sequence emits its EOS token, evict
it and admit a waiting request into that slot immediately. Combined with paged KV
blocks, the batch composition is fluid and the GPU stays full.

**Chunked prefill** completes the picture. A 20k-token prefill occupies the GPU
long enough to stall every decode in flight, producing visible stutter. Split the
prefill into chunks and interleave them with decode steps: TTFT for the new
request degrades slightly, TPOT for everyone else stays smooth.

The knob you will meet in this repo is on the *client* side of that queue.
`ConcurrencyConfig.prefill_queue_target` (`training/utils/config.py`) drives an
AIMD (additive-increase/multiplicative-decrease) controller —
`AdaptiveConcurrencyController` in the SDK — that raises in-flight request count
while the server's reported `prefill_queue_duration` stays under target and backs
off when it does not. It is congestion control, TCP-style, for rollout traffic.

## 2.4 Speculative decoding

Decode is memory-bound (§2.1), so a forward pass over 1 token and a forward pass
over 5 tokens cost *almost the same wall-clock time* — both read all the weights,
and the extra math is nearly free. Speculative decoding converts that slack into
tokens.

**The algorithm.** Let $p$ be the target (real) model and $q$ a cheap **draft**.

1. Draft $\gamma$ tokens autoregressively with $q$ (cheap: small model, or no
   model at all).
2. Run the target **once** over all $\gamma$ proposed positions in parallel,
   obtaining $p(\cdot \mid \text{prefix} + \text{first } j \text{ drafts})$ for
   every $j$.
3. Accept or reject each draft token in order by rejection sampling.

**Why it is exact.** For draft token $x$ drawn from $q$, accept it with
probability

$$\min\!\left(1, \frac{p(x)}{q(x)}\right)$$

If rejected, stop and sample the replacement from the normalized residual
distribution

$$p'(x) = \frac{\max\big(0,\; p(x) - q(x)\big)}{\sum_{x'} \max\big(0,\; p(x') - q(x')\big)}$$

The composition of these two steps is provably distributed exactly as $p$. So
speculative decoding is a **pure latency optimization** — with a correct
implementation the output distribution is unchanged, no matter how bad the draft
model is. A bad draft just means low acceptance and little speedup.

**How much it wins.** With per-token acceptance probability $\alpha$, the expected
number of tokens produced per verification pass is

$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

At $\alpha = 0.8, \gamma = 4$: $\approx 3.4$ tokens per target pass — roughly a 2–3×
latency win after draft overhead. Note the diminishing return in $\gamma$: each
extra speculated token only pays off if all previous ones were accepted.

**Flavors**, cheapest to most accurate:

| Method | Draft source | Notes |
|---|---|---|
| Prompt lookup / n-gram | Copy from context | Free; excellent for summarization, code edits, RAG (lots of verbatim reuse) |
| Draft model | A small model of the same family | Classic; needs a matched tokenizer |
| Medusa | Extra heads on the target predicting $t+2, t+3, \dots$ | No separate model; tree-structured candidates |
| EAGLE / EAGLE-2/3 | Autoregress on the target's *hidden features*, not tokens | Highest acceptance rates today; the usual default |

**When to turn it off.** Speculation trades FLOPs for latency. If the server is
already saturated — large batches, throughput-oriented offline workloads — those
FLOPs are not free anymore, and rejected drafts are pure waste; throughput can go
*down*. That is why this repo exposes it as a deployment flag rather than
enabling it unconditionally:

```250:251:training/utils/config.py
    disable_speculative_decoding: bool = False
    """When true, disable the base model's default draft/EAGLE speculation."""
```

For RL rollouts there is a second reason to care. Rollouts need per-token
logprobs from the sampler, and different speculative implementations can report
those slightly differently from a plain decode path. Any such discrepancy shows
up downstream as train–inference mismatch, which part 4 handles explicitly with
TIS (§4.8) and observability metrics that compare the trainer's logprobs against
the raw inference logprobs (`training/utils/rl/observability.py`). If those
metrics look pathological, toggling `disable_speculative_decoding` is a
legitimate bisection step.

## 2.5 Disaggregated prefill/decode

Prefill and decode want opposite things from the hardware (§2.1) and, on a shared
replica, they fight:

- A long prefill monopolizes the SMs and stalls every in-flight decode → ugly
  inter-token-latency spikes.
- Decode's tiny batches leave the tensor cores idle → wasted compute.
- They also want different parallelism: prefill likes tensor parallelism to cut
  TTFT; decode often prefers less TP and more replicas, since it is bandwidth-
  bound and TP adds all-reduce latency per step.

**Disaggregation** ("disagg", also P/D disaggregation) runs them as separate
pools of workers. A request is prefilled on a prefill worker, its KV cache is
transferred over the interconnect (NVLink within a node, RDMA/InfiniBand across
nodes) to a decode worker, and generation proceeds there.

The trade:

- **Win:** each pool is scheduled and scaled for one bottleneck; TTFT and TPOT
  can be tuned independently and both SLOs met; no head-of-line blocking.
- **Cost:** you must ship the whole KV cache for the prompt across the network —
  tens to hundreds of MB per request (see the table in §2.2) — so it only pays
  off with fast interconnect and reasonably long prompts. Below some prompt
  length, chunked prefill on a colocated replica wins.

Disaggregation is a serving-side architecture decision made by the platform, not
a cookbook config. What surfaces to you is its *effect*: the
`prefill_queue_duration` signal that the adaptive concurrency controller in §2.3
reacts to.

> Terminology warning: this repo also uses the word "disaggregate" for something
> completely different. `training/renderer/_disaggregate_mixin.py` splits a
> multi-turn conversation into one supervised example per assistant turn, so
> thinking-model traces are trained with the right visibility. Same word,
> unrelated concept.

## 2.6 Quantization and why training cares

Serving in lower precision reduces both weight bytes (→ faster decode, §2.1) and
KV bytes (→ bigger batches, §2.2):

| Format | Bits | Typical use |
|---|---|---|
| BF16 | 16 | Training default; the numerical reference point |
| FP8 (E4M3) | 8 | Weights + activations on Hopper/Blackwell; near-lossless with per-tensor or per-block scales |
| INT8 / INT4 | 8 / 4 | Weight-only quantization (AWQ, GPTQ); good for memory-bound decode |
| FP4 (NVFP4/MXFP4) | 4 | Blackwell-class; block scaling factors do the heavy lifting |

The key idea is **per-block scaling**: store a low-precision mantissa plus a
higher-precision scale for each small group of values, so outliers do not destroy
the whole tensor's dynamic range.

For this repo, the important downstream consequence is that **the inference
engine and the trainer are not numerically the same function.** They differ in
precision, in kernels, in reduction order, in parallelism layout, possibly in
MoE routing (§1.7), and possibly in speculation (§2.4). Feed the same tokens
through both and you get different logprobs. During supervised training this is
harmless. During RL it silently biases your gradient, which is exactly what TIS
(§4.8) exists to correct.

## 2.7 Serving metrics vocabulary

| Metric | Meaning | Driven by |
|---|---|---|
| TTFT | Time to first token | Prefill compute, queueing, prefix cache hit rate |
| TPOT / ITL | Time per output token | HBM bandwidth, batch size, TP degree |
| Throughput | Total tokens/s across all requests | Batch size, therefore KV cache capacity |
| Goodput | Throughput that actually met its SLO | Scheduling quality |
| MFU | Model FLOPs utilization vs peak | High in prefill/training, low in decode |

The recurring tension: batching raises throughput and lowers cost per token, but
raises per-request latency. Speculation lowers latency but spends FLOPs.
Disaggregation lets you stop making that trade globally and make it per phase.

---

**Next:** [3. Training](training.md) — forward, backward, optimizer step, and LoRA.
