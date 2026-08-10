# Terminal-Bench RL with Harbor and OpenCode

This example trains an OpenCode agent on the stable Harbor dataset
`terminal-bench@2.0`. Harbor owns the local Docker task environment and verifier;
the shared Harbor/OpenCode rollout function records every model call and returns
one `RolloutRun` to the serverless async-RL recipe.

Use a prepared task tree so every task image has the same pinned OpenCode CLI:

```bash
export HARBOR_RL_DIR="${HARBOR_RL_DIR:-$PWD/.harbor-rl}"

uv run harbor datasets download terminal-bench@2.0 \
  -o "$HARBOR_RL_DIR/tasks/terminal-bench-2.0"

uv run python -m \
  training.examples.rl.harbor.prepare_opencode_tasks \
  --source "$HARBOR_RL_DIR/tasks/terminal-bench-2.0" \
  --destination "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --opencode-version 1.18.8
```

First calibrate five tasks through the exact rollout function used for training.
`--harbor-trials-dir` retains the Harbor results, OpenCode logs, and compressed
token-native trajectory artifacts for inspection:

```bash
uv run python -m \
  training.examples.rl.harbor_rl_terminal_bench.train_serverless \
  --sampling-only \
  --harbor-dataset "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --harbor-trials-dir "$HARBOR_RL_DIR/runs/tbench-calibration/trials" \
  --run-dir "$HARBOR_RL_DIR/runs/tbench-calibration"
```

Router Replay remains enabled by default. Pass `--no-router-replay` when the
selected sampling pool does not expose MoE routing statistics. The rollout still
records exact tokens and log probabilities; it omits inference routing matrices.

The training defaults are 8 completions per prompt, 8 prompt groups per
optimizer step, 2 pipeline chunks, an off-policy budget of 2 policy versions,
32,768 completion tokens per turn, a 196,608-token sequence limit, and 80 rows
(10 optimizer steps). Evaluation uses eight fixed held-out tasks at step 0 and
then every five optimizer steps.

Rollout admission stays on the async coordinator's adaptive default. The shared
Harbor adapter separately limits active local trials to 24 so Docker environment
capacity does not become sampler concurrency policy.

```bash
uv run python -m \
  training.examples.rl.harbor_rl_terminal_bench.train_serverless \
  --harbor-dataset "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --harbor-trials-dir "$HARBOR_RL_DIR/runs/tbench-10-step/trials" \
  --run-dir "$HARBOR_RL_DIR/runs/tbench-10-step" \
  --wandb-entity <entity> \
  --wandb-project harbor-rl-opencode
```

OpenCode title and summary requests do not carry tools and are logged as
auxiliary calls. Tool-bearing turns are trainable. Their exact sampled token IDs,
log probabilities, optional routing matrices, history decisions, and trainable
segment shapes are written under `_fireworks_trajectories/`. A history rewrite
starts a new segment within the same logical rollout; it does not create another
GRPO completion or reward.
