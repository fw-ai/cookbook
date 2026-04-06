# KV-Cache Compression: Reproducible Comparison

This package is a self-contained, reproducible comparison of three
approaches to KV-cache key compression:

- **Conventional scalar quantization**
- **TurboQuant**
- **RotorQuant**

The goal is to compare them under matched storage budgets, with the
same workload, on the same GPU, using the same measurement harness.

**Links:** [RotorQuant repo](https://github.com/scrya-com/rotorquant) ·
[TurboQuant paper](https://arxiv.org/abs/2504.19874) ·
[QJL paper](https://arxiv.org/abs/2406.03482)

## TL;DR

1. **Conventional scalar quantization degrades significantly below
   8-bit.** At 4-bit it is already around `0.915` score cosine, and at
   3-bit it is about `0.80`.

2. **TurboQuant has the best quality at matched storage budgets.**
   Across the tested contexts and bit widths, it is consistently at or
   above RotorQuant on score cosine.

3. **TurboQuant has the faster decode score path.** With matched fused
   Triton kernels, its per-query score pipeline is faster than
   RotorQuant's.

4. **RotorQuant's build path is competitive.** RotorQuant build is
   `0.159 ms` at 2k context and `0.652 ms` at 32k (8-bit), essentially
   matching TurboQuant at 32k.

5. **Total time depends on context length.** At 2k and 4k, RotorQuant's
   lower build cost wins overall. At 32k, TurboQuant is slightly lower
   total because the score path is paid on every query.

## Scope

This repository is a kernel-level comparison of three ways to compress
attention keys under a matched measurement setup:

- matched effective storage budgets
- the same synthetic decode workload
- the same measurement harness
- the same hardware target

The repository includes both the code used for the comparison and the
reference outputs.

## The three methods

- **Conventional**: independently round each key dimension to a small
  number of bits, then expand it back before computing attention. Simple
  and fast, but quality degrades quickly.

- **TurboQuant**: rotate keys with one global orthogonal matrix, quantize
  the rotated values, then store a 1-bit residual sketch so the score
  estimate remains unbiased.

- **RotorQuant**: use many small local 3D Clifford rotations instead of
  one global rotation, then apply the same two-part estimator (quantized
  main term + 1-bit sketch correction).

## Evaluation setup

All three implementations start from the public
[RotorQuant repository](https://github.com/scrya-com/rotorquant),
which includes both TurboQuant and RotorQuant code. Two substantive
implementation changes were made for this comparison:

1. **RotorQuant's memory layout was trimmed to matched effective
   bits/dim.** The original layout stores the full Clifford multivector
   (4 components per 3D group, including an unused trivector component),
   plus padding for the last partial group. This evaluation stores only
   the 3 vector-grade components used by reconstruction and omits the
   padding, so that RotorQuant and TurboQuant have the same effective
   storage at each nominal bit width.

2. **Fused Triton scoring kernels were written for both methods.** Each
   kernel computes the full two-term estimator (MSE gather-dot + QJL
   sign-dot) in one pass. The original RotorQuant Triton kernel computes
   only the first term; TurboQuant had no Triton scoring kernel. Using
   the same kernel structure for both methods makes the decode-side speed
   comparison direct.

These changes affect storage layout and the scoring implementation, not
the underlying estimators. The quality results (score cosine, top-1
match) reflect the same mathematical estimators as the public code at
corrected bit budgets.

| Design choice | Why it matters |
|---|---|
| **RotorQuant and TurboQuant are compared at matched effective bits/dim** | The nominal labels correspond to the same real storage budget |
| **Both methods use the full two-term estimator** | Reported scores include both the quantized main term and the QJL correction |
| **Both methods use fused Triton score kernels** | The decode-side comparison uses the same high-level kernel structure |
| **Steady-state timing is reported separately from cold setup** | One-time initialization is separated from recurring build and score cost |
| **RotorQuant build uses a fused Triton path** | Build timing reflects the optimized cache-construction path included in this package |

## Claims

| Claim | Status | Evidence |
|---|---|---|
| Conventional scalar quantization degrades sharply below 8-bit | **Supported** | Score cosine falls from `0.9997` at 8-bit to about `0.915` at 4-bit and about `0.80` at 3-bit |
| TurboQuant and RotorQuant both preserve much more quality below 8-bit | **Supported** | Both stay above `0.97` at 4-bit and around `0.92` at 3-bit |
| TurboQuant has the best quality at matched storage budgets | **Supported** | TurboQuant score cosine is at or above RotorQuant across the tested grid |
| TurboQuant has the faster decode score path | **Supported** | At 8-bit: `0.081 ms` vs `0.130 ms` at 2k, and `0.104 ms` vs `0.157 ms` at 32k |
| RotorQuant has lower total time at shorter contexts | **Supported** | At 8-bit: `0.289 ms` vs `0.497 ms` at 2k, and `0.294 ms` vs `0.572 ms` at 4k |
| TurboQuant has lower total time at 32k | **Supported** | At 8-bit: `0.768 ms` vs `0.809 ms` |
| These rankings transfer directly to real model activations | **Unknown** | This benchmark uses random unit-norm vectors, not full model traces |

## How to read the results

**Bits per dimension.** TurboQuant and RotorQuant store a small per-key
residual norm (16 bits total) on top of their MSE indices and QJL signs.
Across 128 dimensions this adds 0.125 bits/dim, so "8-bit" actually
stores 8.125 bits/dim. Conventional has no such overhead.

**Score cosine.** How well compressed attention scores match the
uncompressed reference. 1.0 = perfect. Above ~0.99 the compression is
nearly invisible. Around 0.80 most fine structure is lost.

**Top-1 match.** Percentage of query heads where the compressed method
agrees with the reference about which key matters most.

**Score e2e.** Time to compute attention scores for one query against
all cached keys. This cost is paid on every decode step. For TurboQuant
and RotorQuant it includes the full pipeline: pre-rotate query +
pre-sketch query + fused Triton kernel.

**Build.** Time to compress **all** cached keys (the full context) in
one batch, averaged over 50 timed iterations. The tables show the
steady-state build: the quantizer is pre-built (codebook, rotation
matrix, QJL projection matrix all allocated) and `torch.compile` is
warmed. What remains is the per-batch compression work — rotating,
quantizing, and projecting the keys.

In a real serving system, keys arrive one at a time as tokens are
generated. The build cost per token would be much smaller than these
batch numbers. The batch build is the relevant cost when filling the
cache from a prompt.

**Total.** Build + score e2e. This represents the combined cost in the
batch-fill-then-score scenario measured by the harness.

In the raw CSV files, these columns are named `score_cosine`,
`top1_match_pct`, `e2e_est_ms`, `build_ms`, and `total_ms`.

## Results

Both TurboQuant and RotorQuant use fused Triton scoring kernels with
the same structure (MSE gather-dot + QJL sign-dot), both in steady state
(pre-built quantizer, `torch.compile` warmed), scored at three context
lengths. Build cost is for compressing the full context in one batch;
score cost is for one query against the compressed cache. All timing
values are averages over 50 iterations after 10 warmup iterations.
Conventional is included as a baseline.

### 2048 cached keys

| method | bits | eff bits/dim | score cos | top-1 % | score e2e (ms) | build (ms) | total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| conventional | 16 | 16.000 | 1.0000 | 100.0 | 0.035 | — | 0.035 |
| conventional | 8 | 8.000 | 0.9997 | 87.5 | 0.064 | 0.039 | 0.103 |
| conventional | 4 | 4.000 | 0.9149 | 25.0 | 0.061 | 0.037 | 0.098 |
| conventional | 3 | 3.000 | 0.7989 | 0.0 | 0.061 | 0.036 | 0.098 |
| TurboQuant | 8 | 8.125 | 0.9998 | 100.0 | 0.081 | 0.417 | 0.497 |
| TurboQuant | 4 | 4.125 | 0.9755 | 50.0 | 0.081 | 0.416 | 0.497 |
| TurboQuant | 3 | 3.125 | 0.9218 | 37.5 | 0.104 | 0.476 | 0.581 |
| TurboQuant | 2 | 2.125 | 0.8045 | 0.0 | 0.080 | 0.420 | 0.499 |
| RotorQuant | 8 | 8.125 | 0.9936 | 87.5 | 0.130 | 0.159 | 0.289 |
| RotorQuant | 4 | 4.125 | 0.9709 | 37.5 | 0.130 | 0.159 | 0.289 |
| RotorQuant | 3 | 3.125 | 0.9193 | 25.0 | 0.131 | 0.159 | 0.289 |
| RotorQuant | 2 | 2.125 | 0.8031 | 12.5 | 0.128 | 0.159 | 0.288 |

### 4096 cached keys

| method | bits | eff bits/dim | score cos | top-1 % | score e2e (ms) | build (ms) | total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| conventional | 16 | 16.000 | 1.0000 | 100.0 | 0.034 | — | 0.034 |
| conventional | 8 | 8.000 | 0.9997 | 100.0 | 0.063 | 0.037 | 0.100 |
| conventional | 4 | 4.000 | 0.9168 | 37.5 | 0.063 | 0.037 | 0.099 |
| conventional | 3 | 3.000 | 0.8047 | 25.0 | 0.063 | 0.038 | 0.100 |
| TurboQuant | 8 | 8.125 | 0.9998 | 100.0 | 0.083 | 0.489 | 0.572 |
| TurboQuant | 4 | 4.125 | 0.9741 | 37.5 | 0.080 | 0.439 | 0.519 |
| TurboQuant | 3 | 3.125 | 0.9186 | 37.5 | 0.081 | 0.437 | 0.517 |
| TurboQuant | 2 | 2.125 | 0.7940 | 25.0 | 0.081 | 0.439 | 0.520 |
| RotorQuant | 8 | 8.125 | 0.9942 | 87.5 | 0.132 | 0.162 | 0.294 |
| RotorQuant | 4 | 4.125 | 0.9689 | 87.5 | 0.133 | 0.163 | 0.296 |
| RotorQuant | 3 | 3.125 | 0.9157 | 37.5 | 0.131 | 0.161 | 0.292 |
| RotorQuant | 2 | 2.125 | 0.7925 | 25.0 | 0.134 | 0.162 | 0.296 |

### 32768 cached keys

| method | bits | eff bits/dim | score cos | top-1 % | score e2e (ms) | build (ms) | total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| conventional | 16 | 16.000 | 1.0000 | 100.0 | 0.087 | — | 0.087 |
| conventional | 8 | 8.000 | 0.9997 | 100.0 | 0.179 | 0.156 | 0.334 |
| conventional | 4 | 4.000 | 0.9170 | 37.5 | 0.179 | 0.156 | 0.335 |
| conventional | 3 | 3.000 | 0.8031 | 0.0 | 0.179 | 0.156 | 0.334 |
| TurboQuant | 8 | 8.125 | 0.9998 | 87.5 | 0.104 | 0.663 | 0.768 |
| TurboQuant | 4 | 4.125 | 0.9756 | 50.0 | 0.100 | 0.586 | 0.686 |
| TurboQuant | 3 | 3.125 | 0.9230 | 12.5 | 0.100 | 0.566 | 0.667 |
| TurboQuant | 2 | 2.125 | 0.8032 | 12.5 | 0.101 | 0.551 | 0.652 |
| RotorQuant | 8 | 8.125 | 0.9998 | 87.5 | 0.157 | 0.652 | 0.809 |
| RotorQuant | 4 | 4.125 | 0.9755 | 50.0 | 0.153 | 0.577 | 0.730 |
| RotorQuant | 3 | 3.125 | 0.9230 | 12.5 | 0.152 | 0.563 | 0.715 |
| RotorQuant | 2 | 2.125 | 0.8028 | 12.5 | 0.152 | 0.551 | 0.703 |

**Notes on measurement:**

- **Steady state** means the quantizer object is pre-built (codebook,
  rotation matrix, QJL projection matrix all allocated and compiled) and
  `torch.compile` is warmed. What the build column measures is the
  per-batch key compression: rotating, quantizing, and projecting all
  `kv_len` keys in one call. This is not a one-time setup cost — it is
  paid every time a new batch of keys is compressed.
- Each timing value is the mean of 50 timed iterations after 10 warmup
  iterations, using `torch.cuda.Event` elapsed time. Variance is low
  (sub-1% RSD) on the B200 for these kernel sizes.
- **Score e2e** is per-query: how long to score one query against the
  full compressed cache. In decoding, this cost is paid on every token.
- Quality is identical between cold and steady-state for both methods
  (the compression math is the same; only timing differs).
- At 32k, quality converges: both methods reach 0.9998 cosine at 8-bit.
  The fidelity difference between methods appears mainly at shorter
  contexts where per-head statistics are noisier.

## Summary: 8-bit steady-state comparison

| Context | TurboQuant total (ms) | RotorQuant total (ms) | Faster |
|---|---:|---:|---|
| 2048 | 0.497 | 0.289 | RotorQuant |
| 4096 | 0.572 | 0.294 | RotorQuant |
| 32768 | 0.768 | 0.809 | TurboQuant |

- At shorter contexts, RotorQuant's cheaper build dominates total time.
- At 32k, build cost is effectively tied, so TurboQuant's faster decode
  score path becomes the deciding factor.

## Measurement setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA B200 (1 of 8) |
| batch size | 1 |
| query heads | 8 |
| key-value heads | 4 (GQA ratio 2) |
| tokens per query | 1 (single-token decode) |
| head dimension | 128 |
| data type | bfloat16 |
| cached key lengths | 2048, 4096, and 32768 |
| warmup iterations | 10 (discarded, not timed) |
| timed iterations | 50 (mean reported in all tables) |
| timer | `torch.cuda.Event` start/end with `synchronize()` |
| key vectors | random, unit length |

The workload is synthetic. Real model activations have statistical
structure (correlations across heads, varying magnitudes per layer) that
this benchmark does not capture. Methods may rank differently on real
data.

## What is in this repo

- `scripts/run_unified_table.py`: the measurement harness
- `turboquant/`: the library code used by the harness
- `configs/kernel_table_phase1.json`: workload shape and timing config
- `results/unified/unified_table-20260401-193204.{csv,md}`: reference run

## Reproduce from this repository

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the full benchmark

```bash
python scripts/run_unified_table.py --gpu 0
```

This writes a timestamped CSV and Markdown table into `results/unified/`.

### 3. Compare against the reference run

The authoritative reference outputs in this repository are:

- `results/unified/unified_table-20260401-193204.csv`
- `results/unified/unified_table-20260401-193204.md`

If you want the exact same workload shape as the reference run, use the
default config in `configs/kernel_table_phase1.json`, which includes:

- contexts: `2048`, `4096`, `32768`
- bits: `16`, `8`, `4`, `3`, `2`
- warmup: `10`
- timed iterations: `50`

### 4. Reproduce one slice quickly

For a faster local sanity check:

```bash
python scripts/run_unified_table.py --gpu 0 --context 2048 --methods turboquant rotorquant
```

For the large-context comparison only:

```bash
python scripts/run_unified_table.py --gpu 0 --context 32768
```

## Caveats

- This is a **synthetic kernel benchmark**, not an end-to-end serving
  benchmark.
- Absolute timings are hardware-dependent.
- Results are for **head_dim = 128**, **batch size = 1**,
  **single-token decode**, and a **B200**.
- Real model activations may shift the relative ranking.
