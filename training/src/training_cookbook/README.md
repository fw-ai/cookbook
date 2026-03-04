# Cookbook (`training_cookbook`)

The cookbook contains compact training loops that are easy to run, read, and customize.
Think of it like a Fireworks-adapted `tinker-cookbook`: same client-side loss pattern, plus Fireworks deployment/hotload orchestration.

## Included recipes

| Recipe | File | Notes |
| --- | --- | --- |
| GRPO / DAPO / GSPO / CISPO | `recipes/rl_loop.py` | Streaming RL loop with greedy rollout batching, pluggable policy loss (`policy_loss="grpo"/"dapo"/"gspo"/"cispo"`), optional TIS and R3 router replay. |
| DPO | `recipes/dpo_loop.py` | Policy + reference trainers; reference logprobs are cached concurrently, then DPO loss is optimized. |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization — no reference model needed. Combined SFT + odds-ratio loss. |
| SFT | `recipes/sft_loop.py` | Single trainer with response-only cross-entropy loss. |

## On-policy RL loop

`rl_loop.py` uses an on-policy streaming architecture: each optimizer step samples exactly `prompt_groups_per_step` prompts (rate-limited by `max_concurrent`), fires `forward_backward_custom` as soon as `min_samples_per_fwd_bwd` samples arrive, then runs `optim_step` + hotload before sampling the next step. Only the current step's prompts are in-flight at any time.

Key scheduling parameters: `prompt_groups_per_step` (prompts per optimizer step), `min_samples_per_fwd_bwd` / `max_samples_per_fwd_bwd` (micro-batch bounds), and `max_concurrent` (in-flight sampling rate limit).

## Shared config blocks

All recipes compose these dataclasses from `utils/config.py`:

- `InfraConfig`: region, accelerators, image tag, node count. Supports `training_shape_id` for auto-config from control-plane training shapes, and `ref_training_shape_id` for a separate reference trainer shape.
- `DeployConfig`: deployment lifecycle, sampling/hotload settings, `tokenizer_model` (required for RL), `disable_speculative_decoding` (for hotload compatibility).
- `HotloadConfig`: hotload cadence, base/delta behavior, timeout.
- `ResumeConfig`: checkpoint source + optional step offset.
- `WandBConfig`: optional experiment logging.
- `ISConfig`, `DAPOConfig`, `GSPOConfig`, `CISPOConfig`: per-algorithm tuning knobs (see source for fields and defaults).

## Prerequisites

Set the following environment variables (or add them to a `.env` file):

```bash
export FIREWORKS_API_KEY=...        # Your Fireworks API key
export FIREWORKS_ACCOUNT_ID=...     # Your Fireworks account ID
```

## Minimal usage — SFT

```python
from training_cookbook.recipes.sft_loop import Config, main
from training_cookbook.utils import InfraConfig

cfg = Config(
    dataset="path/to/chat_data.jsonl",
    tokenizer_model="Qwen/Qwen3-8B",
    base_model="accounts/fireworks/models/qwen3-8b",
    infra=InfraConfig(region="US_OHIO_1"),
)

main(cfg)
```

Dataset format (JSONL, OpenAI chat format):
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Minimal usage — GRPO

```python
from training_cookbook.recipes.rl_loop import Config, main
from training_cookbook.utils import InfraConfig, DeployConfig, HotloadConfig

cfg = Config(
    base_model="accounts/fireworks/models/qwen3-8b",
    max_rows=20,
    epochs=1,
    completions_per_prompt=4,
    policy_loss="grpo",
    infra=InfraConfig(region="US_OHIO_1"),
    deployment=DeployConfig(
        deployment_id="my-grpo-run",
        create_deployment=True,
        tokenizer_model="Qwen/Qwen3-8B",
    ),
    hotload=HotloadConfig(hot_load_interval=1),
)

main(cfg)
```

## What to customize first

- Reward logic in `recipes/rl_loop.py` (`reward_fn`).
- Loss functions in `utils/rl/` (`grpo.py`, `dapo.py`, `gspo.py`, `cispo.py`, `importance_sampling.py`).
- Data adapters in `utils/data.py` for your JSONL schema.
- Resume behavior in `utils/resume.py`.

## Relationship to SDK

The cookbook does not replace the SDK; it composes it:

- lifecycle: `TrainerJobManager`, `DeploymentManager`
- training clients: `FiretitanServiceClient`, `ReconnectableClient`
- checkpoint/hotload chain: `WeightSyncer`
- token-in/token-out sampling: `DeploymentSampler`

If you need a custom algorithm loop, copy the closest recipe and keep using the same SDK primitives.
