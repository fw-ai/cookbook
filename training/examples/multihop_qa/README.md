# Multi-Hop QA with IGPO

Information Gain-based Policy Optimization (IGPO) for multi-turn agentic reinforcement learning on the HotpotQA dataset.

> **Reference**: Wang et al., *"Information Gain-based Policy Optimization"* ([arXiv:2510.14967](https://arxiv.org/abs/2510.14967), ICLR 2026).

## What is IGPO?

Standard RL for LLMs (GRPO, PPO) assigns a single reward at the end of a trajectory.
In multi-turn agentic settings — where the model issues tool calls across several turns — intermediate turns receive zero reward, making credit assignment difficult.

IGPO adds a **turn-level intrinsic reward** based on **information gain**: how much the model's belief in the correct answer changes after observing new evidence from a tool call.
Concretely, after each turn *t* with prefix *p_t*, the reward is:

```
r_t = r_env(t) + ig_weight * [log P(answer | p_t) - log P(answer | p_{t-1})]
```

This gives intermediate turns meaningful signal — a `search()` call that retrieves relevant paragraphs increases `log P(answer)` and earns positive IG reward, while an unhelpful query earns negative IG reward.

## Architecture

IG scoring runs through the **inference deployment** (completions API with `echo=True, logprobs=1`), not through the trainer.
This keeps the trainer exclusively for `forward_backward` and avoids GPU batching conflicts.

```
┌──────────────┐    rollout    ┌──────────────┐
│   Inference   │◀────────────▶│  Orchestrator │
│  Deployment   │              │  (this script)│
│  (sampling +  │   IG score   │              │
│   IG scoring) │◀─────────────│              │
└──────────────┘              └──────┬───────┘
                                     │ forward_backward
                                     ▼
                              ┌──────────────┐
                              │   Trainer     │
                              │  (tinker)     │
                              └──────────────┘
```

## Quick start

### 1. Install dependencies

```bash
pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook eval-protocol datasets httpx
```

### 2. Prepare dataset

```bash
# Hard-only HotpotQA (recommended — matches paper difficulty)
python prepare_data.py --max-rows 2000 --difficulty hard

# Or combine with harder datasets (MuSiQue + 2WikiMultiHopQA)
python prepare_data.py --dataset all --max-rows 3000 --difficulty hard
```

This downloads multi-hop QA data and writes `dataset.jsonl`. The `--difficulty hard`
filter keeps only hard-level HotpotQA questions where intermediate credit assignment
matters most for IGPO.

### 3. Run training

```bash
export FIREWORKS_API_KEY="your-api-key"
export TRAINING_SHAPE="your-training-shape-id"
export OUTPUT_MODEL_ID="your-output-model-id"

# IGPO (with information gain reward)
python train_multihop_qa_igpo.py \
    --training-shape "$TRAINING_SHAPE" \
    --output-model-id "$OUTPUT_MODEL_ID" \
    --ig-weight 0.1

# GRPO baseline (no IG, environment reward only)
python train_multihop_qa_igpo.py \
    --training-shape "$TRAINING_SHAPE" \
    --output-model-id "$OUTPUT_MODEL_ID" \
    --ig-weight 0.0
```

Or use the convenience script:

```bash
TRAINING_SHAPE=... OUTPUT_MODEL_ID=... bash run.sh
```

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--ig-weight` | `0.1` | Weight for IG intrinsic reward. **Start small (0.01–0.2)**. Values >= 0.5 tend to destabilize training because IG rewards (log-probability differences) can dominate environment rewards. Set to `0` for pure GRPO. |
| `--gamma` | `0.95` | Discount factor for turn-level return accumulation (paper uses 0.95). |
| `--completions-per-prompt` | `4` | Group size for GRPO-style advantage normalization. |
| `--prompt-groups-per-step` | `4` | Number of prompts per optimization step (effective batch = this × completions_per_prompt). |
| `--learning-rate` | `1e-5` | Adam learning rate. |
| `--kl-beta` | `0.001` | KL penalty coefficient against the reference model. |
| `--eps-clip` | `0.2` | PPO-style clipping range for the surrogate loss. |
| `--max-steps` | `8` | Maximum search turns per question. |
| `--search-top-k` | `2` | Number of paragraphs returned per search query. |
| `--skip-ig-last-turn` | `True` | Skip IG reward on the final turn (use env reward only). Helps when the last turn is `submit_answer` and environment reward is already meaningful. |
| `--epochs` | `3` | Number of passes over the dataset. |

### Reward normalization

Following the paper, IG rewards and outcome rewards are **z-normalized independently**
within each prompt group before being combined. This prevents either signal from
dominating regardless of their raw scales. The combined signal then undergoes
backward discounted return accumulation (γ=0.95) to propagate future-turn
information to earlier turns.

### Tuning `ig_weight`

With separate normalization, `ig_weight` controls the IG scoring computation
but both streams contribute equally after normalization. Set to `0` to disable
IG scoring entirely (pure GRPO).

**Recommended starting points:**
- `0.0` — pure GRPO baseline (no IG scoring overhead)
- `0.1` — default; IG scoring active with separate normalization
- Values >= 0.5 emit a warning (IG scoring is expensive; the normalization
  already balances the signals)

## Files

| File | Description |
|---|---|
| `train_multihop_qa_igpo.py` | Main training script with sampling loop, IG scoring, and weight sync. |
| `multihop_qa_rollout.py` | Rollout processor: manages multi-turn tool-call conversations with the model. |
| `search_env.py` | Local TF-IDF search environment over HotpotQA paragraph pools. |
| `prepare_data.py` | Dataset preparation: downloads HotpotQA and writes `dataset.jsonl`. |
| `run.sh` | Convenience wrapper script. |

## Model compatibility

This example uses Qwen3 models by default. The rollout processor includes `_strip_think_block()` to handle Qwen3's `<think>...</think>` reasoning tokens, which consume token budget and can cause context overflow in multi-turn conversations.

For models that do not produce thinking tokens, `_strip_think_block` is a no-op (it only matches `<think>` tags). For models with different thinking token formats, adjust the regex patterns in `multihop_qa_rollout.py`.
