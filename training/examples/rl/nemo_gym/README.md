# NeMo Gym async RL on Fireworks

Trains a policy on a [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environment
(multi-turn tool use) using Fireworks' `async_rl_loop` (Dedicated deployment). NeMo Gym owns the environment, agent, and reward; Fireworks
owns sampling, GRPO, and hotloading.

```
async_rl_loop ── rollout_fn ──POST /run──▶ NeMo Gym agent ──▶ RecordingProxy ──▶ Fireworks sampler
                     ▲                                             │ records exact token ids + logprobs
                     └──────── RolloutRun(segments) ◀──────────────┘ keyed by rollout id
```

## Requirements

- `FIREWORKS_API_KEY` set.
- A NeMo Gym checkout with its venv (`$NEMO_GYM_DIR`, default `<repo>/../nemo-gym`),
  **including [NVIDIA-NeMo/Gym#3783](https://github.com/NVIDIA-NeMo/Gym/pull/3783)**
  (`correlate_via_user_field`). Until it is merged, apply its diff to your checkout.
  Without it the proxy returns HTTP 400 on every call (no `user` field to correlate on).
- This example **writes `$NEMO_GYM_DIR/env.yaml`**. It refuses to overwrite an
  `env.yaml` it did not generate.

## Run

```bash
python -m training.examples.rl.nemo_gym.train
python -m training.examples.rl.nemo_gym.train \
    --resources-server workplace_assistant --agent-name workplace_assistant_simple_agent
```

For `toolsandbox`, add `--exclude-user-simulator` so the simulated user's model
calls are not trained on as policy actions.

## Behavior to know

- **History is append-only per rollout id.** `SimpleAgent` is sequential and never
  rewrites history; harnesses that fork or rewrite history are unsupported.
- **Segments:** when a turn's prompt is not a token-prefix of the previous turn, the
  rollout becomes multiple training segments; all are returned. The per-rollout log
  line reports turns, segments, and trained tokens.
- **Sampling is pinned to the run config:** per-request `temperature` is ignored and
  `max_tokens` can only lower `--max-completion-tokens`.

## Tests

```bash
cd training && uv run --frozen --with pytest python -m pytest tests/test_nemo_gym_proxy.py
```
