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

## Dedicated Kimi-K3 full-parameter convergence test

The generic OpenCode recipe can attach to an existing full-parameter trainer
and rollout deployment. The following command runs the synchronous full-corpus
convergence workload: 16 prompt groups x 8 rollouts per optimizer step,
shuffled training rows, E2B task environments, no Router Replay, 262K total
context, 32K maximum output per model call, and a fixed evaluation every five
steps. It does not clean up the supplied resources when interrupted. GSPO uses
the paper-recommended asymmetric `[1 - 3e-4, 1 + 4e-4]` clipping interval.

```bash
uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --renderer-name kimi_k3 \
  --trainer-job-id <trainer-job-id> \
  --deployment-id <deployment-id> \
  --deployment-shape <versioned-rollout-shape> \
  --harbor-dataset <prepared-terminal-bench-opencode-directory> \
  --harbor-trials-dir <run-directory>/trials \
  --log-path <run-directory> \
  --harbor-environment e2b \
  --harbor-trial-config training/examples/rl/harbor/recipes/terminal_bench/two_hour_trial.yaml \
  --max-concurrent-trials 128 \
  --evaluation-task count-dataset-tokens \
  --evaluation-task extract-elf \
  --evaluation-task polyglot-rust-c \
  --cycle-selected-tasks \
  --task-seed 20260808 \
  --max-rows 264 \
  --epochs 1 \
  --completions-per-prompt 8 \
  --prompt-groups-per-step 16 \
  --pipeline-chunks-per-step 4 \
  --min-group-size 8 \
  --max-incomplete-group-retries 2 \
  --lora-rank 0 \
  --learning-rate 1e-6 \
  --kl-beta 0 \
  --max-head-offpolicy-versions 0 \
  --policy-loss gspo \
  --no-router-replay \
  --grad-accumulation-normalization num_sequences \
  --grad-clip-norm 1.0 \
  --eps-clip 0.0003 \
  --eps-clip-high 0.0004 \
  --tis-cap 5 \
  --max-seq-len 262144 \
  --max-completion-tokens 32768 \
  --sample-timeout 7200 \
  --harness-tool-timeout-seconds 7200 \
  --evaluation-every 5 \
  --evaluation-concurrency 24 \
  --dcp-save-interval 10 \
  --shuffle \
  --no-cleanup-on-exit \
  --wandb-entity <entity> \
  --wandb-project <project> \
  --wandb-run-name <run-name>
```

Set `FIREWORKS_API_KEY` and, when W&B logging is enabled,
`WANDB_API_KEY` in the environment. Do not put either secret in the command or
the run directory. Use `--init-from-checkpoint step-N` to resume the trainer's
weights and optimizer without recreating the trainer or rollout deployment.

### Concrete inputs for the convergence run

The convergence run uses the following immutable shape versions and concrete inputs.
The trainer and deployment IDs are recorded for provenance; create replacements
from the same shape versions if those resources have expired.

| Input | Value |
| --- | --- |
| Base model | `accounts/fireworks/models/kimi-k3` |
| Tokenizer model | `moonshotai/Kimi-K3` |
| Training shape | `accounts/fireworks/trainingShapes/kimi-k3-262k-gb300/versions/rbb16rr5` |
| Rollout shape | `accounts/fireworks/deploymentShapes/kimi-k3-rl-gb300-fp4-w16-p4/versions/pu8yssdz` |
| Trainer | `accounts/training/rlorTrainerJobs/k3-gspo-all89-20260910-213414` |
| Deployment | `accounts/training/deployments/k3-gspo-all89-20260910-213414` |
| Prepared dataset | `/shared/yuedong/kimi-k3-harbor-convergence-data/terminal-bench-opencode-e2b-v7` |
| Training tasks | All tasks discovered in the prepared dataset (89 in the pinned Terminal-Bench dataset) |
| Evaluation tasks | `count-dataset-tokens`, `extract-elf`, `polyglot-rust-c` |
| Training rows | 264 prompt groups cycled across all 89 tasks (2.97 corpus passes); task order seeded with `20260808`, then shuffled by the RL loop |
| Optimizer batch | 16 prompt groups x 8 rollouts = 128 trajectories; 4 pipeline chunks |
| Training length | 17 optimizer steps (16 full and one 8-group tail); 2,112 sampled trajectories |
| Optimization | full parameter; LR `1e-6`; Adam beta2 `0.95`; Adam epsilon `1e-12`; gradient clipping at `1.0`; sequence-count gradient normalization |
| Policy objective | GSPO sequence-level importance ratio; `kl_beta=0`; asymmetric clip `3e-4` / `4e-4`; TIS cap `5`; synchronous (`max_head_offpolicy_versions=0`) |
| Loss reduction | Mean over active response tokens within each sequence, then equal mean over sequences (`num_sequences`) |
| Routing | Router Replay disabled; GSPO does not require routing replay |
| Harbor backend | E2B; 128 concurrent trials; two-hour outer-trial and tool timeouts |
| Token limits | 262,144 total tokens; 32,768 generated tokens per model call |
| Evaluation/checkpointing | the same three fixed tasks every 5 steps; DCP every 10 steps |
| W&B run | [`u3ibepq0`](https://wandb.ai/myh97/kimi-k3-fullparam-harbor/runs/u3ibepq0) |
| Prior-run evidence | [`9a13a8f5`](https://wandb.ai/myh97/kimi-k3-fullparam-harbor/runs/9a13a8f5); it used LR `2e-6` and no shuffle |

This is the credential-safe command for the convergence run. The SDK/model-request
timeout and the per-tool Harbor timeout are separate controls, so both are set
to 7,200 seconds for long-tail tasks. W&B records the configured clipping
epsilons and the dynamic `train/gspo_sequence_ratio_mean`,
`train/gspo_clip_frac`, `train/gspo_clip_low_frac`, and
`train/gspo_clip_high_frac` metrics.

```bash
RUN_DIR=/shared/yuedong/kimi-k3-harbor-convergence/k3-gspo-all89-20260910-213414

uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --renderer-name kimi_k3 \
  --trainer-job-id k3-gspo-all89-20260910-213414 \
  --deployment-id k3-gspo-all89-20260910-213414 \
  --deployment-shape accounts/fireworks/deploymentShapes/kimi-k3-rl-gb300-fp4-w16-p4/versions/pu8yssdz \
  --harbor-dataset /shared/yuedong/kimi-k3-harbor-convergence-data/terminal-bench-opencode-e2b-v7 \
  --harbor-trials-dir "$RUN_DIR/trials" \
  --log-path "$RUN_DIR" \
  --harbor-environment e2b \
  --harbor-trial-config training/examples/rl/harbor/recipes/terminal_bench/two_hour_trial.yaml \
  --max-concurrent-trials 128 \
  --evaluation-task count-dataset-tokens \
  --evaluation-task extract-elf \
  --evaluation-task polyglot-rust-c \
  --cycle-selected-tasks \
  --task-seed 20260808 \
  --max-rows 264 \
  --epochs 1 \
  --completions-per-prompt 8 \
  --prompt-groups-per-step 16 \
  --pipeline-chunks-per-step 4 \
  --min-group-size 8 \
  --max-incomplete-group-retries 2 \
  --lora-rank 0 \
  --learning-rate 1e-6 \
  --kl-beta 0 \
  --max-head-offpolicy-versions 0 \
  --policy-loss gspo \
  --no-router-replay \
  --grad-accumulation-normalization num_sequences \
  --grad-clip-norm 1.0 \
  --eps-clip 0.0003 \
  --eps-clip-high 0.0004 \
  --tis-cap 5 \
  --max-seq-len 262144 \
  --max-completion-tokens 32768 \
  --sample-timeout 7200 \
  --harness-tool-timeout-seconds 7200 \
  --evaluation-every 5 \
  --evaluation-concurrency 24 \
  --dcp-save-interval 10 \
  --shuffle \
  --no-cleanup-on-exit \
  --wandb-entity myh97 \
  --wandb-project kimi-k3-fullparam-harbor \
  --wandb-run-name k3-gspo-all89-20260910-213414
```

OpenCode title and summary requests do not carry tools and are logged as
auxiliary calls. Tool-bearing turns are trainable. Their exact sampled token IDs,
log probabilities, optional routing matrices, history decisions, and trainable
segment shapes are written under `_fireworks_trajectories/`. A history rewrite
starts a new segment within the same logical rollout; it does not create another
GRPO completion or reward.
