# Multi-Turn GSM8K (message-in) RL example

A multi-turn math agent that ports
[AReaL's `examples/multi_turn_math/`](https://github.com/inclusionAI/AReaL/tree/main/examples/multi_turn_math)
to the cookbook's async RL recipe.

For the recipe API and gate sizing, see
[`/skills/dev/references/rl/async-rl.md`](/skills/dev/references/rl/async-rl.md).

The model is asked a GSM8K problem and must put its final answer in
`\boxed{...}`.  If the boxed answer is wrong (verified by `math_verify`),
the rollout appends a fixed user-feedback message and lets the model retry
once more (configurable via `--max-turns`, default 2).

The full trajectory — prompt + first attempt + feedback + second attempt —
is packed into a single `RolloutSample`.  `MessageTrajectoryAssembler`
keeps the per-token loss mask aligned: assistant tokens are trained on
across both turns; the original prompt and the user-feedback bridge tokens
are masked out.

## Files

- `prepare_data.py` — downloads `openai/gsm8k` (config `main`) from HuggingFace
  via ``load_dataset("openai/gsm8k", "main")`` and writes ``train.jsonl``
  (7473 rows) and ``test.jsonl`` (1319 rows) with ``{messages, answer}`` rows.
- `reward.py` — extracts `\boxed{...}` from the completion and verifies
  against the GSM8K ground-truth (`#### N`) via numeric match plus a
  `math_verify` fallback.
- `rollout.py` — per-sample `make_rollout_fn(setup) -> RolloutFn`; runs the
  retry loop and returns one `RolloutSample` per trajectory.
- `train.py` — wires the dataset and the rollout factory into
  `recipes/async_rl_loop.main`.
- `run.sh` — one-shot end-to-end (auto-runs `prepare_data.py` if needed).

## Usage

```bash
# 1. Download GSM8K (writes train.jsonl + test.jsonl in this directory).
python prepare_data.py

# 2. Train.
python train.py \
    --base-model accounts/fireworks/models/qwen3-1p5b-instruct \
    --tokenizer-model Qwen/Qwen2.5-1.5B-Instruct \
    --max-rows 512 \
    --completions-per-prompt 4 \
    --max-turns 2 \
    --output-model-id accounts/<acct>/models/gsm8k-mt
```

Or `bash run.sh` for the canned configuration.

## Reference

- AReaL `examples/multi_turn_math/gsm8k_rl_mt.py` — the original
  `MultiTurnMathAgent` whose retry loop, feedback prompt, and reward
  function this example mirrors.
- AReaL `areal/experimental/openai/types.py::to_tensor_dict` — the
  bridge-masking semantics (prompt and feedback positions get
  `loss_mask=0`, only assistant tokens are trained on) that
  `MessageTrajectoryAssembler.to_flat()` reproduces in this recipe.
