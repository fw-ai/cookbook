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

- `FIREWORKS_API_KEY` set for your resource-owning account.
- Explicit `--base-model`, `--tokenizer-model`, and `--training-shape-id` values.
  The tokenizer must match a certified TiTo renderer in `training.renderer.tito.registry`;
  incompatible tokenizers fail before Gym startup or resource provisioning. The former
  Qwen3.5-27B default is not certified by the current runtime.
- A NeMo Gym checkout with its venv (`$NEMO_GYM_DIR`, default `<repo>/../nemo-gym`) that
  includes [NVIDIA-NeMo/Gym#3783](https://github.com/NVIDIA-NeMo/Gym/pull/3783)
  (`correlate_via_user_field`, merged to `main` on 2026-10-08), so any `main` from that
  date on. On an older checkout Gym rejects the `env.yaml` key this example writes.
  Without the option the proxy returns HTTP 400 on every call (no `user` field to correlate on).
- This example **writes `$NEMO_GYM_DIR/env.yaml`**. It refuses to overwrite an
  `env.yaml` it did not generate.

## Run

```bash
python -m training.examples.rl.nemo_gym.train \
    --base-model "$BASE_MODEL" --tokenizer-model "$TOKENIZER_MODEL" --training-shape-id "$TRAINING_SHAPE_ID"
python -m training.examples.rl.nemo_gym.train \
    --base-model "$BASE_MODEL" --tokenizer-model "$TOKENIZER_MODEL" --training-shape-id "$TRAINING_SHAPE_ID" \
    --resources-server workplace_assistant --agent-name workplace_assistant_simple_agent
```

For `toolsandbox`, add `--exclude-user-simulator` so the simulated user's model
calls are not trained on as policy actions.

## Behavior to know

- **Token lineage is verified by the SDK TiTo runtime.** A rollout id correlates
  calls; it does not prove that the new prompt continues the previous token history.
- **Segments:** when a turn's prompt cannot continue the previous exact token history, the
  rollout becomes multiple training segments; all are returned. Drift starts a
  new segment rather than overwriting earlier sampled tokens. The per-rollout log
  line reports turns, segments, and trained tokens.
- **Sampling follows the recipe, not the request.** The proxy samples with the recipe's
  `RolloutSetup.sample_kwargs` (temperature, `top_p=1.0`, `top_k=0`, ...). A per-request
  `temperature`/`top_p` is ignored, and a request's `max_tokens` can only lower the
  configured `--max-completion-tokens`.
- **Tool calls:** `parallel_tool_calls=false` is enforced on the returned response.
  Multiple calls reject the trajectory locally with HTTP 422 and
  `nemo_parallel_tool_calls` (`origin=adapter`), not a retryable upstream 502.
  The retained failure artifact contains the exact sampled calls; none are dropped
  or reparsed. Grammar-free RL generation does not guarantee single-call adherence.
  Tasks with independent tools may explicitly set `parallel_tool_calls=true` in
  their native `responses_create_params`; dependent-tool tasks should retain their
  stricter policy.
- **Routing replay:** TiTo retains expert routing in trainer coordinates and rejects
  missing or misaligned routing when the recipe requires it.
- **Sessions admit only active rollouts.** The recipe registers an id before calling
  Gym and closes admission before collecting it; late requests receive HTTP 410.
  Failed or timed-out Gym calls abandon the trajectory and cancel active sampling.
- **Rollout ids are explicit and unique per call** (`_ng_rollout_id`), so a retry never
  appends onto a stale session.
- **Logs:** `gym_env_start.log` (NeMo Gym's startup output) is written under `--log-path`.

Pin `--tokenizer-revision` to use the same recipe tokenizer contract in the
proxy and training loop. `--deployment-shape`, `--max-seq-len`, and
`--max-head-offpolicy-versions` pass directly into the shared recipe; context
length and `--max-completion-tokens` are independent limits.

Model-call failures retain TITO's structured origin, request/retry information,
and parser rejection code even if Gym returns a generic HTTP error from `/run`.
Recognized parser failures discard only that trajectory. SDK parser traces
remain under `<log-path>/tito-failures`. Adapter response-policy rejections use
the SDK's bounded failure-artifact store; always use the logged `trace_path`
to locate the retained evidence.
Environment HTTP/timeout failures have separate `nemo_run_http` and
`nemo_run_timeout` reasons. Non-transient HTTP errors and invalid rewards fail
without being relabeled as parser failures. Shutdown terminates the owned Gym
process group, including resource, agent and model servers.

## Tests

```bash
cd training && uv run --frozen --with pytest python -m pytest tests/unit/test_nemo_gym_proxy.py
```
