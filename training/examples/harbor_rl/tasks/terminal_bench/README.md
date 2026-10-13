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
  training.examples.harbor_rl.harnesses.opencode.prepare_tasks \
  --source "$HARBOR_RL_DIR/tasks/terminal-bench-2.0" \
  --destination "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --opencode-version 1.18.8
```

First calibrate five tasks through the exact rollout function used for training.
`--harbor-trials-dir` retains the Harbor results, OpenCode logs, and compressed
token-native trajectory artifacts for inspection:

```bash
uv run python -m \
  training.examples.harbor_rl.tasks.terminal_bench.train \
  --sampling-only \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --harbor-dataset "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --harbor-trials-dir "$HARBOR_RL_DIR/runs/tbench-calibration/trials" \
  --run-dir "$HARBOR_RL_DIR/runs/tbench-calibration"
```

Router Replay is requested automatically for MoE models; the rollout records
exact tokens, log probabilities and inference routing matrices.
Prompt construction defaults to `full_history`. Use
`--tito-prompt-mode incremental` only as an experimental opt-in with a renderer
whose stronger incremental suffix/junction capability has been implemented and
validated; the harness, Harbor environment, and
sidecar endpoint remain unchanged.

The training defaults are 8 completions per prompt, 8 prompt groups per
optimizer step, an off-policy budget of 2 policy versions,
32,768 completion tokens per turn, a 196,608-token total context and
exact-boundary retention limit, and 80 rows (10 optimizer steps). Override the
total limit with `--max-seq-len`. Evaluation uses eight fixed held-out tasks at
step 0 and then every five optimizer steps. Complete accepted groups submit
as soon as they are ready, with one optimizer update after all backwards succeed.

Rollout admission stays on the async coordinator's adaptive default. The shared
Harbor rollout separately limits active local trials (`--max-concurrent-trials`,
default 24) so Docker environment capacity does not become sampler concurrency
policy.

```bash
uv run python -m \
  training.examples.harbor_rl.tasks.terminal_bench.train \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --harbor-dataset "$HARBOR_RL_DIR/tasks/terminal-bench-opencode" \
  --harbor-trials-dir "$HARBOR_RL_DIR/runs/tbench-10-step/trials" \
  --run-dir "$HARBOR_RL_DIR/runs/tbench-10-step" \
  --wandb-entity <entity> \
  --wandb-project harbor-rl-opencode
```

The model, tokenizer, and renderer are deliberately required. The sidecar
refuses a renderer/tokenizer pair without a production TITO certification
(`training/renderer/tito/registry.py`) when the trial's sidecar starts, so
check the pair with a short `--sampling-only` run first.

OpenCode title and summary requests do not carry tools and are logged as
auxiliary calls. Tool-bearing turns are trainable. Each trial's compact `.tito`
artifact (exact sampled token IDs, log probabilities, optional routing
matrices, and segment decisions) is retained under `--harbor-trials-dir`;
`--tito-debug` adds JSONL troubleshooting events. A history rewrite starts a
new segment within the same logical rollout; it does not create another GRPO
completion or reward. The run directory receives `task-split.json`,
`metrics.jsonl`, and `run-state.json`.
