# Multi-Hop QA with async RL (+ optional IGPO turn-level scoring)

Multi-turn agentic reinforcement learning on HotpotQA.  Runs on top of
`training/recipes/async_rl_loop.py` (gate-native rollout/train overlap,
PPO inner minibatches, weight-sync hotload), with optional turn-level
**Information Gain** scoring folded into the trajectory's scalar reward.

> **Reference**: Wang et al., *"Information Gain-based Policy Optimization"* ([arXiv:2510.14967](https://arxiv.org/abs/2510.14967), ICLR 2026).

## What is IGPO (in this example)?

Standard RL for LLMs (GRPO, PPO) assigns a single reward at the end of a trajectory.
In multi-turn agentic settings — where the model issues tool calls across several turns — intermediate turns receive zero reward, making credit assignment difficult.

IGPO adds a **turn-level intrinsic reward** based on **information gain**: how much the model's belief in the correct answer changes after observing new evidence from a tool call:

```
ig_t = log P(answer | prefix_t) - log P(answer | prefix_{t-1})
```

This example folds the per-turn IG into the trajectory's scalar reward used by the async RL loop:

```
reward = outcome_last + ig_weight * sum(ig_per_turn)
```

The async loop then computes GRPO-style advantages by z-normalising this
augmented reward across the prompt group.  IG-rich trajectories (where
search calls genuinely raised `log P(answer)`) earn higher advantage even
when the final answer is wrong.

## Architecture

IG scoring runs through the **inference deployment** (completions API with
`echo=True, logprobs=1`), in parallel with sampling, via a thread-pool of
scoring workers.  The trainer is reserved for `forward_backward` only.

```
   ┌──────────────┐    rollout (multi-turn)   ┌──────────────┐
   │   Inference   │◀──────────────────────▶│  rollout_fn   │
   │  Deployment   │                          │  (rollout.py) │
   │  (sampling +  │   IG score (per turn)    │               │
   │   IG scoring) │◀─────────────────────────│               │
   └──────────────┘                          └──────┬───────┘
                                                    │ RolloutSample
                                                    ▼
                                            ┌──────────────┐
                                            │ async_rl_loop │
                                            │  (recipe)     │
                                            └──────┬───────┘
                                                   │ forward_backward
                                                   ▼
                                            ┌──────────────┐
                                            │   Trainer     │
                                            │  (tinker)     │
                                            └──────────────┘
```

## Quick start

### 1. Set up environment

Follow the setup instructions in [../../README.md](../../README.md).

### 2. Prepare dataset

```bash
# Hard-only HotpotQA (recommended — matches paper difficulty)
python examples/multihop_qa/prepare_data.py --max-rows 2000 --difficulty hard

# Or combine with harder datasets (MuSiQue + 2WikiMultiHopQA)
python examples/multihop_qa/prepare_data.py --dataset all --max-rows 3000 --difficulty hard
```

This downloads multi-hop QA data and writes `dataset.jsonl`.

### 3. Run training

```bash
export FIREWORKS_API_KEY="your-api-key"
export TRAINING_SHAPE_ID="accounts/<acct>/trainingShapes/<shape-id>"
export OUTPUT_MODEL_ID="accounts/<acct>/models/<short-id>"

# IGPO (with information gain reward folded in)
python examples/multihop_qa/train.py \
    --training-shape-id "$TRAINING_SHAPE_ID" \
    --output-model-id "$OUTPUT_MODEL_ID" \
    --ig-weight 1.0

# GRPO baseline (no IG, environment reward only)
python examples/multihop_qa/train.py \
    --training-shape-id "$TRAINING_SHAPE_ID" \
    --output-model-id "$OUTPUT_MODEL_ID" \
    --ig-weight 0.0
```

Or use the convenience script:

```bash
TRAINING_SHAPE_ID=... OUTPUT_MODEL_ID=... bash run.sh
```

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--ig-weight` | `1.0` | Folds turn-level IG into the scalar reward as `outcome + ig_weight * sum(ig_per_turn)`. Set to `0` for pure GRPO baseline. |
| `--skip-ig-last-turn` | `True` | Zero IG on the final `submit_answer` turn (env reward already covers it). |
| `--scoring-workers` | `4` | Threadpool size for IG scoring. |
| `--completions-per-prompt` | `8` | Group size for GRPO advantage normalisation. |
| `--prompt-groups-per-step` | `8` | Prompt rows per outer rollout batch. |
| `--max-head-offpolicy-versions` | `1` | Off-policy staleness budget (in weight-sync versions). `0` = strict on-policy. |
| `--ppo-n-minibatches` | `1` | Inner PPO minibatches per rollout batch (`1` = legacy 1:1). |
| `--max-concurrency-rollout-sample` | unbounded | Cap in-flight rollout invocations (LLM-call units, matches `deployment.max_batch_size`). |
| `--learning-rate` | `1e-5` | Adam learning rate. |
| `--kl-beta` | `0.001` | KL penalty against the reference model. **Set to `0` for LoRA runs** so no reference trainer is provisioned. |
| `--max-turns` | `8` | Max search/submit turns per trajectory. |
| `--search-top-k` | `2` | Paragraphs returned per search. |
| `--epochs` | `1` | Passes over the dataset. |

### Reward folding vs. the paper

The reference paper z-normalises IG and outcome rewards **separately within each prompt group** and combines them with a backward discounted-return accumulation per turn.  This example collapses the per-turn signal into a single trajectory-level scalar so it fits the `RolloutSample` flat-reward contract that `async_rl_loop` consumes.  The async loop's GRPO advantage (z-score across the group) still propagates the IG signal — just without per-token credit assignment.  If you need the full per-turn formulation, fork `rollout.py` and emit a custom `Rollout`/`PromptGroup` directly.

## Files

| File | Description |
|---|---|
| `train.py` | Entrypoint: parses args, builds the async RL `Config`, invokes `async_rl_loop.main()`. |
| `rollout.py` | Rollout factory (`make_rollout_fn`) that the recipe wires into the loop — multi-turn search/submit + IG scoring + `RolloutSample` assembly. |
| `multihop_qa_rollout.py` | Multi-turn tool-call processor (eval-protocol) reused by `rollout.py`. |
| `search_env.py` | Local TF-IDF search environment over HotpotQA paragraph pools. |
| `prepare_data.py` | Dataset preparation: downloads HotpotQA and writes `dataset.jsonl`. |
| `run.sh` | Convenience wrapper script. |

## Model compatibility

This example uses Qwen3 models by default. The rollout processor strips Qwen3-style `<think>...</think>` reasoning tokens between turns to avoid context overflow.  For other model families, adjust the regex patterns in `multihop_qa_rollout.py`.
