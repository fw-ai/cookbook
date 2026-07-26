# LLM systems from first principles

An onboarding primer for the technology underneath the Fireworks Training API and
this cookbook. It starts from "what is a token" and builds up to the exact loss
functions and scheduling knobs implemented in `training/`.

Nothing here is a substitute for the live [Fireworks docs](https://docs.fireworks.ai)
or the skill references in [`skills/fireworks-training/`](../skills/fireworks-training/SKILL.md).
Those are operational; this is conceptual. Where a concept lands in real code,
the file path is cited so you can read the implementation next.

## Reading order

| # | File | What you learn |
|---|---|---|
| 1 | [transformers-and-attention.md](transformers-and-attention.md) | Tokens, embeddings, Q/K/V, softmax attention, multi-head, RoPE, MLP/SwiGLU, MoE, parameter and FLOP counting |
| 2 | [inference.md](inference.md) | Prefill vs decode, KV cache math, continuous batching, speculative decoding, disaggregated serving, quantization |
| 3 | [training.md](training.md) | Forward/backward/optimizer step, autodiff, AdamW, gradient accumulation, memory budgets, LoRA, SFT, DPO, ORPO, distillation |
| 4 | [reinforcement-learning.md](reinforcement-learning.md) | Policy gradients, REINFORCE, PPO clipping, GRPO, KL estimators, on-policy vs off-policy, sync vs async RL, train–inference mismatch and TIS |
| 5 | [gpus-and-systems.md](gpus-and-systems.md) | SMs, HBM, roofline, precision formats, collectives, FSDP/HSDP, TP/PP/EP, what a "training shape" hides |
| 6 | [fireworks-stack-map.md](fireworks-stack-map.md) | How every concept above maps to the Tinker protocol, the SDK, and files in `training/` |

Read 1 → 2 → 3 → 4 in order; each depends on the previous. Files 5 and 6 can be
read at any point and are useful as references.

## The one-paragraph version

A transformer turns a sequence of tokens into a probability distribution over the
next token, using attention (a learned, content-addressed lookup between
positions) and MLPs, stacked in residual blocks. Serving it fast is a memory
problem, not a math problem: you cache per-token key/value tensors so generation
does not re-read history, batch many requests to amortize weight loads, and use
tricks like speculative decoding to buy more tokens per pass. Training it is
three primitives repeated: forward (compute a loss), backward (compute
gradients), optimizer step (change the weights). Fine-tuning shrinks the cost by
only training a low-rank correction (LoRA). Reinforcement learning replaces the
fixed label with a reward you compute yourself, which forces you to run
inference and training in the same loop — and that coupling is where most of the
interesting systems work in this repo lives.

## Notation used throughout

| Symbol | Meaning |
|---|---|
| $t_1 \dots t_n$ | token IDs in a sequence |
| $n$ | sequence length (context) |
| $d$ or $d_\text{model}$ | model/residual-stream width |
| $L$ | number of transformer layers |
| $h$ | number of attention heads, $d_h = d/h$ the per-head width |
| $V$ | vocabulary size |
| $N$ | total parameter count |
| $\pi_\theta$ | the policy (the model being trained), parameters $\theta$ |
| $\pi_\text{ref}$ | frozen reference model |
| $G$ | group size in GRPO (completions per prompt) |
