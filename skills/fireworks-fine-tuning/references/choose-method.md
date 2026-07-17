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
| **ORPO** | preferred vs non-preferred responses (same shape as DPO) | Same signal as DPO but **no reference model** (one model, less memory/GPU). A reasonable DPO alternative on tight budgets; use the `fireworks-training` companion skill and `training/recipes/orpo_loop.py`. |
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

If the user has prompts but no preference pairs, do not reject the task or silently invent labels. Use `references/preference-data-and-evaluators.md` to plan, cost, generate, review, and preserve pair provenance before upload.

## RFT — reinforcement fine-tuning

Provide three things (not necessarily labeled outputs): a **dataset** of prompts; an **evaluator or inline reward** that scores an output 0.0→1.0; and the **agent** being trained. Managed RFT uses a registered evaluator. Training SDK RFT uses reward code and may read `ground_truth` or any other field declared by that reward. Start with **200–500 diverse prompts**. Docs: https://docs.fireworks.ai/fine-tuning/how-rft-works · [evaluators](https://docs.fireworks.ai/fine-tuning/evaluators)

## Classification (a common SFT task)

Classification is SFT where the assistant turn is the label. Two extra checks matter:

- **Label imbalance.** Compute the ratio of the most-frequent to least-frequent label. If it's high (say > 50:1), the model can score well by always predicting the majority. Flag it in the plan and mitigate: rebalance/subsample, use per-message `weight`, or narrow the label set.
- **Per-label accuracy, not just overall.** Report accuracy per label plus a confusion-matrix-style summary (which labels get confused for which), and always measure on a held-out split. Overall accuracy hides minority-class failure.
- **Baseline first.** Run the base model on the eval split before training so you have a base-vs-fine-tuned delta (see the benchmark step in `references/deploy-and-troubleshoot.md`), not just an absolute number.

If `ground_truth` is a separate field rather than the final assistant turn, map it onto the assistant message before upload.

## Hyperparameter sweep + promotion gate

For anything past a smoke run, don't hand-pick one config: run the small grid below as **separate jobs**, compare on a held-out split, and promote the winner. This is now **agent-driven** (the coding agent runs the loop), not a server-side auto-sweep; the workflow lives in `references/orchestrate-from-agent.md`.

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
method = "sft"   # "sft" | "dpo" | "managed-rft" | "sdk-rft"
# Set only when the selected SDK reward reads a required per-row field.
sdk_reward_required_field = None  # e.g. "ground_truth"
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
    elif method in {"managed-rft", "sdk-rft"}:
        assert o.get("messages"), f"line {i}: no messages[]"
        if method == "sdk-rft" and sdk_reward_required_field:
            assert sdk_reward_required_field in o, f"line {i}: no {sdk_reward_required_field!r} required by reward_fn"
    else:
        raise ValueError(f"unknown method: {method}")
assert n >= 3, "need at least 3 examples"
print(f"OK: {n} valid {method} rows")
```

Reminders: JSONL, one object per line, roles in order, min 3 and max 3M examples. DPO is one-turn only (preferred and non-preferred are the single last assistant turn). Managed RFT validates the prompt schema plus whatever its registered evaluator expects. SDK RFT validates only the fields its selected reward reads; `ground_truth` is common, not universal. `firectl dataset create` re-validates on upload, but running this first turns a late platform error into an instant local fix.
