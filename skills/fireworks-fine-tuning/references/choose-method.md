# Choosing a fine-tuning method + preparing data

*Source of truth: [Fine-tuning intro](https://docs.fireworks.ai/fine-tuning/finetuning-intro.md) — defer to the live docs for current dataset schemas + method support.*

Pick the method that matches the *signal you have*, not the one that sounds most advanced. Start simple (SFT); escalate only when your data demands it. Source: https://docs.fireworks.ai/fine-tuning/finetuning-intro

## Decision tree

```
Do you have labeled ground-truth outputs?
├─ Yes, >1000 examples ............................ SFT
├─ Yes, 100–1000 + reasoning helps ................ RFT
├─ Yes, 100–1000 + no reasoning needed ............ SFT
├─ Only "A is better than B" preference pairs ..... DPO
└─ No labels, but outputs are verifiable/scorable . RFT
```

| Method | You provide | Best for |
|--------|-------------|----------|
| **SFT** | Labeled `messages` (the ideal output) | Classification, extraction, style/format, tool-call shaping. The default. |
| **DPO** | preferred vs non-preferred responses | Aligning tone/quality when you can rank two answers but can't write the one true answer. |
| **RFT** | Prompts + an evaluator/reward (0.0–1.0) | Verifiable tasks (math, code, agents) with few labels. |

## SFT format

JSONL, one object per line; OpenAI-style `messages`. **Min 3, max 3M** (aim for 1000+ for quality). Optional per-message `weight` (0 skips from loss).

```jsonl
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Capital of France?"},{"role":"assistant","content":"Paris."}]}
```

Docs: https://docs.fireworks.ai/fine-tuning/fine-tuning-models · [weighted training](https://docs.fireworks.ai/fine-tuning/weighted-training)

## DPO format

Preference pairs, **one-turn only** (preferred/non-preferred must be the last assistant turn). Two accepted shapes:

```jsonl
{"input":{"messages":[{"role":"user","content":"What is Einstein famous for?"}]},"preferred_output":[{"role":"assistant","content":"His theory of relativity, E=mc²."}],"non_preferred_output":[{"role":"assistant","content":"He was a scientist."}]}
```
(Training API also accepts `chosen`/`rejected`.) Docs: https://docs.fireworks.ai/fine-tuning/dpo-fine-tuning

## RFT — reinforcement fine-tuning

Provide three things (not labeled outputs): a **dataset** of prompts; an **evaluator** that scores an output 0.0→1.0 (the reward), registered via `pytest` or a remote service; and the **agent** being trained. Start with **200–500 diverse prompts**. Docs: https://docs.fireworks.ai/fine-tuning/how-rft-works · [evaluators](https://docs.fireworks.ai/fine-tuning/evaluators)

## LoRA vs full-parameter

**Pick LoRA first** — small adapter, cheaper/faster, fewer GPUs, deployable on the base model. **LoRA rank** default 8 (range 4–32); raise for complex reasoning, keep low for style/format. Go **full-parameter** only when LoRA plateaus and you have GPU budget. Docs: https://docs.fireworks.ai/fine-tuning/parameter-tuning

## Data quality (the main driver of SFT quality)

Quality beats quantity: a small clean set beats a large noisy one. Check two things separately:
- **Format errors, in isolation:** unreadable JSONL, missing roles, missing outputs, malformed tool calls, invalid labels, train/eval leakage (run the validator above).
- **The dataset itself:** distribution, coverage of the real input range, label quality, and whether a held-out split actually measures your goal. Capture train/eval row counts, rough input/output token lengths, the output schema or label set, and the metric that decides whether the tune helped. No held-out split means training-set behavior is not evidence of generalization; say so.

## Hyperparameter starting points

One unpinned first LoRA SFT pass: `lora_rank 8`, `learning_rate ~1.5e-4`, 1 epoch. Change one thing at a time and watch the curves. For a small first search, a conservative 3-cell grid (each cell is a separate run and a separate cost line):

| Config | LoRA rank | Learning rate |
|---|---:|---:|
| 1 | 8 | 1.5e-4 |
| 2 | 16 | 1.0e-4 |
| 3 | 32 | 5.0e-5 |

## RL loss methods (RFT)

- **GRPO** (default) — group-relative, symmetric clip `[0.8,1.2]`, small KL penalty. Conservative baseline; start here.
- **DAPO** — asymmetric clip `[0.8,1.28]`, no KL; more aggressive/faster.
- **GSPO-token** — sequence-level IS, tight clipping; stability + long-form, may need more steps.

## Critical rules

- **Default to SFT.** DPO only with preference pairs; RFT only with a reward/verifiable task.
- **Validate dataset format before uploading** — JSONL, one object/line, right schema, roles in order. Min 3, max 3M.
- **DPO is one-turn only.**
- **Start LoRA + defaults** (rank 8, 1 epoch, LR ~1e-4); change one thing at a time, watch the curves.
- **Iterate cheap first** — validate evaluator/data on a small model before scaling.

## Validate before uploading (run this first)

Catch format errors locally before `firectl dataset create`. A malformed row otherwise surfaces as a late, often masked, failure. Set `method`, run this on your JSONL, and fix any rejection before uploading:

```python
import json, sys
method = "sft"   # "sft" | "dpo" | "rft"
n = 0
for i, line in enumerate(open(sys.argv[1]), 1):
    line = line.strip()
    if not line:
        continue
    o = json.loads(line); n += 1
    if method == "sft":
        assert isinstance(o.get("messages"), list) and o["messages"], f"line {i}: no messages[]"
    elif method == "dpo":
        assert o.get("input", {}).get("messages"), f"line {i}: no input.messages"
        assert o.get("preferred_output") and o.get("non_preferred_output"), f"line {i}: missing preferred_output/non_preferred_output"
    elif method == "rft":
        assert o.get("messages"), f"line {i}: no messages[]"
        assert "ground_truth" in o, f"line {i}: no ground_truth (needed by the inline reward_fn)"
assert n >= 3, "need at least 3 examples"
print(f"OK: {n} valid {method} rows")
```

Reminders: JSONL, one object per line, roles in order, min 3 and max 3M examples. DPO is one-turn only (preferred and non-preferred are the single last assistant turn). RFT rows carry `ground_truth` for the reward. `firectl dataset create` re-validates on upload, but running this first turns a late platform error into an instant local fix.
