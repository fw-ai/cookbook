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
  training.examples.rl.harbor.opencode.prepare_tasks \
  --source "$HARBOR_RL_DIR/tasks/terminal-bench-2.0" \
  --destination "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --opencode-version 1.18.8
```

First calibrate five tasks through the exact rollout function used for training.
`--harbor-trials-dir` retains the Harbor results, OpenCode logs, and compressed
token-native trajectory artifacts for inspection:

```bash
uv run python -m \
  training.examples.rl.harbor.recipes.terminal_bench.train \
  --sampling-only \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --harbor-dataset "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --harbor-trials-dir "$HARBOR_RL_DIR/runs/tbench-calibration/trials" \
  --run-dir "$HARBOR_RL_DIR/runs/tbench-calibration"
```

Router Replay remains enabled by default. Pass `--no-router-replay` when the
selected sampling pool does not expose MoE routing statistics. The rollout still
records exact tokens and log probabilities; it omits inference routing matrices.
Prompt construction defaults to `full_history`. Use
`--tito-prompt-mode incremental` only as an experimental opt-in with a renderer
whose stronger incremental suffix/junction capability has been implemented and
validated; the harness, Harbor environment, and
sidecar endpoint remain unchanged.

The training defaults are 8 completions per prompt, 8 prompt groups per
optimizer step, 2 pipeline chunks, an off-policy budget of 2 policy versions,
32,768 completion tokens per turn, a 196,608-token total context and
exact-boundary retention limit, and 80 rows (10 optimizer steps). Override the
total limit with `--max-seq-len`. Evaluation uses eight fixed held-out tasks at
step 0 and then every five optimizer steps.

Rollout admission stays on the async coordinator's adaptive default. The shared
Harbor adapter separately limits active local trials to 24 so Docker environment
capacity does not become sampler concurrency policy.

```bash
uv run python -m \
  training.examples.rl.harbor.recipes.terminal_bench.train \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --harbor-dataset "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --harbor-trials-dir "$HARBOR_RL_DIR/runs/tbench-10-step/trials" \
  --run-dir "$HARBOR_RL_DIR/runs/tbench-10-step" \
  --wandb-entity <entity> \
  --wandb-project harbor-rl-opencode
```

The model, tokenizer, and renderer are deliberately required. V1 ships the
GLM-5.2 sidecar renderer; a model/template pair that has only offline renderer
coverage is rejected before creating a Harbor trial.

OpenCode title and summary requests do not carry tools and are logged as
auxiliary calls. Tool-bearing turns are trainable. Their exact sampled token IDs,
log probabilities, optional routing matrices, history decisions, and trainable
segment shapes are written under `_fireworks_trajectories/`. A history rewrite
starts a new segment within the same logical rollout; it does not create another
GRPO completion or reward.
