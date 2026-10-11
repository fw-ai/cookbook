# Multi-Hop QA with IGPO

Information Gain-based Policy Optimization (IGPO) for multi-turn agentic reinforcement learning on the HotpotQA dataset.

> **Reference**: Wang et al., *"Information Gain-based Policy Optimization"* ([arXiv:2510.14967](https://arxiv.org/abs/2510.14967), ICLR 2026).

## What is IGPO?

Standard RL for LLMs (GRPO, PPO) assigns a single reward at the end of a trajectory.
In multi-turn agentic settings — where the model issues tool calls across several turns — intermediate turns receive zero reward, making credit assignment difficult.

IGPO adds a **turn-level intrinsic reward** based on **information gain**: how much the model's belief in the correct answer changes after observing new evidence from a tool call.
Concretely, with *p_t* the context before turn *t* of *T* turns, the turn rewards are:

```
IG_t = log P(answer | p_{t+1}) - log P(answer | p_t)    for t < T
IG_T = 0;  outcome reward r_env on the last turn only
```

This gives intermediate turns meaningful signal — a `search()` call that retrieves relevant paragraphs increases `log P(answer)` and earns positive IG reward, while an unhelpful query earns negative IG reward.

## Architecture

Training is [`recipes/rl_loop.py`](../../../recipes/rl_loop.py) with the IGPO
advantage estimator ([`utils/rl/algorithm/igpo.py`](../../../utils/rl/algorithm/igpo.py));
`train_multihop_qa_igpo.py` only supplies the task:
`rl_loop.main(cfg, rollout_fn_factory=..., prepare_batch_fn=igpo.prepare_batch(gamma=...))`.
Each step:

1. The recipe calls `rollout_fn(row)` `completions_per_prompt` times per
   question. Each call plays one search episode with `MultiHopQARolloutProcessor`
   (eval-protocol) against the inference deployment. The conversation stays
   token ids -- each next prompt is the previous prompt, the completion, the
   turn's close and the tool reply -- so an episode packs into one training
   sample whose turn spans IGPO scores (`training.utils.rl.rollout.turn_traces`).
   The run's metadata carries `TURN_SPANS` and the tokenized ground truth as
   `ANSWER_TOKENS`.
2. `prepare_batch_fn`: one `forward` on the policy trainer scores `log P(answer | p_t)` for every
   turn of every rollout. The trainer holds the sampling policy because the
   recipe hotloads it after every update.
3. One `forward_backward` with the trainer's built-in `ppo` loss trains on the
   per-token turn advantages, clipped against the rollout logprobs, then one
   optimizer step and a hotload.

## Quick start

### 1. Set up environment

Follow the setup instructions in [../../../README.md](../../../README.md).

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

python train_multihop_qa_igpo.py \
    --training-shape "$TRAINING_SHAPE" \
    --output-model-id "$OUTPUT_MODEL_ID"
```

Or use the convenience script:

```bash
TRAINING_SHAPE=... OUTPUT_MODEL_ID=... bash run.sh
```

To reuse a running trainer and deployment, add `--policy-job-id <job>` and
`--deployment-id <deployment>`; resources the run did not create are never
cleaned up. `--output-model-id` is optional.

For a GRPO baseline on the same task, call `rl_loop.main` with the same
`make_rollout_fn_factory` and no `prepare_batch_fn`; it trains every run with
one sequence-level advantage and ignores the turn spans.

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--gamma` | `0.95` | Discount factor for turn-level return accumulation (paper uses 0.95). |
| `--completions-per-prompt` | `4` | Group size for GRPO-style advantage normalization. |
| `--prompt-groups-per-step` | `4` | Number of prompts per optimization step (effective batch = this × completions_per_prompt). |
| `--learning-rate` | `1e-5` | Adam learning rate. |
| `--eps-clip` | `0.2` | PPO-style clipping range for the surrogate loss. |
| `--max-steps` | `8` | Maximum search turns per question. |
| `--search-top-k` | `2` | Number of paragraphs returned per search query. |
| `--epochs` | `3` | Number of passes over the dataset. |

### Reward normalization

Following the paper, IG rewards and outcome rewards are **z-normalized independently**
within each prompt group before being combined. This prevents either signal from
dominating regardless of their raw scales. The combined signal then undergoes
backward discounted return accumulation (γ=0.95) to propagate future-turn
information to earlier turns.

## Files

| File | Description |
|---|---|
| `train_multihop_qa_igpo.py` | Task code (episode rows, packing with turn spans, rollout factory), CLI and `rl_loop.main` with `igpo.prepare_batch`. |
| `multihop_qa_rollout.py` | eval-protocol rollout processor: multi-turn tool calls on token ids, parsed by the cookbook renderer. |
| `search_env.py` | Local TF-IDF search environment over each question's paragraph pool. |
| `prepare_data.py` | Dataset preparation: downloads HotpotQA (or MuSiQue, 2WikiMultiHopQA, FRAMES) and writes `dataset.jsonl`. |
| `run.sh` | Convenience wrapper script. |

## Model compatibility

This example uses Qwen3 models by default and samples with thinking disabled.
Tool calls are read in the model's native format by the cookbook renderer for
`--tokenizer-model` (Qwen3 JSON, Qwen3.5/3.6 XML functions, Kimi sections, ...).
A turn without a parseable call is resampled up to three times before the
episode is dropped.
