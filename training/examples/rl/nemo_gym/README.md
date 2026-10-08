# NeMo Gym async RL on Fireworks

Trains a policy on a [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) environment
(multi-turn tool use) using Fireworks' `async_rl_loop` (Dedicated deployment). NeMo Gym owns the environment, agent, and reward; Fireworks
owns sampling, GRPO, and hotloading.

```
async_rl_loop ── rollout_fn ──POST /run──▶ NeMo Gym agent ──▶ RecordingChatProxy ──▶ Fireworks sampler
                     ▲                                             │ records exact token ids + logprobs
                     └──────── RolloutRun(segments) ◀──────────────┘ keyed by rollout id
```

## Requirements

- `FIREWORKS_API_KEY` set.
- A NeMo Gym checkout with its venv (`$NEMO_GYM_DIR`, default `<repo>/../nemo-gym`) that
  includes [NVIDIA-NeMo/Gym#3783](https://github.com/NVIDIA-NeMo/Gym/pull/3783)
  (`correlate_via_user_field`, merged to `main` on 2026-10-08), so any `main` from that
  date on. On an older checkout Gym rejects the `env.yaml` key this example writes.
  Without the option the proxy returns HTTP 400 on every call (no `user` field to correlate on).
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
- **Sampling follows the recipe, not the request.** The proxy samples with the recipe's
  `RolloutSetup.sample_kwargs` (temperature, `top_p=1.0`, `top_k=0`, ...). A per-request
  `temperature`/`top_p` is ignored, and a request's `max_tokens` can only lower the
  configured `--max-completion-tokens`.
- **Rollout ids are explicit and unique per call** (`_ng_rollout_id`), so a retry never
  appends onto a stale session.
- **Logs:** `gym_env_start.log` (NeMo Gym's startup output) is written under `--log-path`.

## Tests

```bash
cd training && uv run --frozen --with pytest python -m pytest tests/test_nemo_gym_proxy.py
```
