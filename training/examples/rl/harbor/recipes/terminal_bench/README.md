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

Before launching the full fanout, complete the checks in
[`E2B_RUNBOOK.md`](E2B_RUNBOOK.md). They cover task-image compatibility,
template readiness, the certified Kimi tokenizer/renderer pair, and one real
trajectory smoke test.

The generic OpenCode recipe can attach to an existing full-parameter trainer
and rollout deployment. The following command runs the synchronous full-corpus
convergence workload: 16 prompt groups x 8 rollouts per optimizer step,
shuffled training rows, E2B task environments, no Router Replay, 262K total
context, 32K maximum output per model call, and a fixed evaluation every five
steps. The dedicated trial config provisions 4 CPUs and 8 GB per E2B sandbox;
the 2 GB task default is insufficient for the Chrome-heavy
`filter-js-from-html` verifier. It does not clean up the supplied resources
when interrupted. GSPO uses
the paper-recommended asymmetric `[1 - 3e-4, 1 + 4e-4]` clipping interval.

```bash
uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --tokenizer-revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --renderer-name kimi_k3_preserve_thinking \
  --trainer-job-id <trainer-job-id> \
  --deployment-id <deployment-id> \
  --deployment-shape <versioned-rollout-shape> \
  --harbor-dataset <prepared-terminal-bench-opencode-directory> \
  --harbor-trials-dir <run-directory>/trials \
  --log-path <run-directory> \
  --harbor-environment e2b \
  --harbor-trial-config training/examples/rl/harbor/recipes/terminal_bench/two_hour_trial.yaml \
  --e2b-task-memory-mb rstan-to-pystan=16384 \
  --e2b-task-verifier-timeout-seconds torch-tensor-parallelism=1200 \
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
  --pipeline-chunks-per-step 16 \
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
  --harness-tool-timeout-seconds 6900 \
  --evaluation-every 5 \
  --evaluation-concurrency 24 \
  --dcp-save-interval 1 \
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
| Tokenizer revision | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` (the production-certified TITO bundle) |
| Training shape | `accounts/fireworks/trainingShapes/kimi-k3-262k-gb300/versions/rbb16rr5` |
| Rollout shape | `accounts/fireworks/deploymentShapes/kimi-k3-rl-gb300-fp4-w16-p4/versions/pu8yssdz` |
| Trainer | `accounts/training/rlorTrainerJobs/k3-gspo-gradclip-gloo-candidate-20260913-0225` |
| Deployment | `accounts/training/deployments/k3-gspo-gradclip-gloo-candidate-20260913-0225` |
| Prepared dataset | `/shared/yuedong/kimi-k3-harbor-convergence-data/terminal-bench-opencode-e2b-v12` |
| Training tasks | All tasks discovered in the prepared dataset (89 in the pinned Terminal-Bench dataset) |
| Evaluation tasks | `count-dataset-tokens`, `extract-elf`, `polyglot-rust-c` |
| Training rows | 264 prompt groups cycled across all 89 tasks (2.97 corpus passes); task order seeded with `20260808`, then shuffled by the RL loop |
| Optimizer batch | 16 prompt groups x 8 rollouts = 128 trajectories; 16 pipeline chunks (one prompt group per forward/backward call) |
| Training length | 17 optimizer steps (16 full and one 7-group tail); 2,104 trained trajectories. One 8-trajectory group was excluded after a verifier timed out in all three attempts. |
| Optimization | full parameter; LR `1e-6`; Adam beta2 `0.95`; Adam epsilon `1e-12`; gradient clipping at `1.0`; sequence-count gradient normalization |
| Policy objective | GSPO sequence-level importance ratio; `kl_beta=0`; asymmetric clip `3e-4` / `4e-4`; TIS cap `5`; synchronous (`max_head_offpolicy_versions=0`) |
| Loss reduction | Mean over active response tokens within each sequence, then equal mean over sequences (`num_sequences`) |
| Routing | Router Replay disabled; GSPO does not require routing replay |
| Harbor backend | E2B; 128 concurrent trials; 8 GiB normally and 16 GiB for `rstan-to-pystan`; two-hour outer-trial and tool timeouts |
| Token limits | 262,144 total tokens; 32,768 generated tokens per model call |
| Evaluation/checkpointing | the same three fixed tasks every 5 steps; DCP every step |
| W&B run | [`vg0u67hs`](https://wandb.ai/myh97/kimi-k3-fullparam-harbor/runs/vg0u67hs) |
| Prior-run evidence | [`9a13a8f5`](https://wandb.ai/myh97/kimi-k3-fullparam-harbor/runs/9a13a8f5); it used LR `2e-6` and no shuffle |

### Completed convergence-run result

The 17-step run completed the full dataset lifecycle, including a DCP save
after every optimizer step, sampler hot-loads, and the final fixed evaluation.
It validates the harness and recovery path, but it does **not** establish model
convergence: the fixed evaluation did not improve over its step-0 value.

| Result | Observed value |
| --- | --- |
| Source population | 264 shuffled prompt groups over all 89 tasks |
| Trained population | 263 prompt groups; 2,104 trajectories |
| Training-batch reward | `0.5859` at step 1, `0.7143` at step 17; first-five mean `0.7500`, last-five mean `0.7788` |
| Fixed evaluation reward | step 0 `0.5417`; step 5 `0.5417`; step 8 `0.3750`; step 10 `0.6250`; step 15 `0.5417`; final step 17 `0.4583` |
| Evaluation coverage | 24 trajectories per evaluation: 3 fixed tasks x 8 rollouts |
| Train-inference K3 | mean `0.00769`; range `0.00600`-`0.00937`; final `0.00693` |
| Train-inference KLD | mean `0.01693`; range `0.01009`-`0.02268`; final `0.01338` |
| Trainer throughput | mean `29,294` tok/s; median `28,796` tok/s; range `23,813`-`32,732` tok/s |
| Gradient clipping | all 17 steps clipped; pre-clip global norm `17.87`-`235.83`, post-clip norm approximately `1.0` |
| Policy clipping | GSPO clip fraction `0` on every step; maximum TIS clip fraction `0.0071%` |
| Recovery | 6 incomplete-group retries; 8 trajectory drops, all from one excluded `torch-tensor-parallelism` group |
| Final checkpoints | DCP `step-18`; sampler `step-17-cff7e02b` |

The evaluation has only 24 binary-reward samples per point. Its step-0 and
final 95% Wilson intervals overlap substantially, so the observed `-0.0833`
change is not enough to claim either improvement or regression. Training-batch
reward is also not a convergence metric because each shuffled step contains a
different task mix. Before another expensive run, expand the fixed evaluation
set and restore the qualified Kimi-K3 train-inference alignment configuration;
this run explicitly disabled Router Replay and measured K3 well above the
previously qualified approximately `0.0015` level.

This is the credential-safe command for the convergence run. The SDK/model-request
timeout and the per-tool Harbor timeout are separate controls, so both are set
to 7,200 seconds for long-tail tasks. W&B records the configured clipping
epsilons and the dynamic `train/gspo_sequence_ratio_mean`,
`train/gspo_clip_frac`, `train/gspo_clip_low_frac`, and
`train/gspo_clip_high_frac` metrics. The 128-way E2B fanout uses more than the
common 1,024-descriptor shell default, so raise the client process limit before
launching it.

```bash
RUN_DIR=/shared/yuedong/kimi-k3-harbor-convergence/<run-name>
ulimit -n 65536

uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --tokenizer-revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --renderer-name kimi_k3_preserve_thinking \
  --trainer-job-id <trainer-job-id> \
  --deployment-id <deployment-id> \
  --deployment-shape accounts/fireworks/deploymentShapes/kimi-k3-rl-gb300-fp4-w16-p4/versions/pu8yssdz \
  --harbor-dataset /shared/yuedong/kimi-k3-harbor-convergence-data/terminal-bench-opencode-e2b-v12 \
  --harbor-trials-dir "$RUN_DIR/trials" \
  --log-path "$RUN_DIR" \
  --harbor-environment e2b \
  --harbor-trial-config training/examples/rl/harbor/recipes/terminal_bench/two_hour_trial.yaml \
  --e2b-task-memory-mb rstan-to-pystan=16384 \
  --e2b-task-verifier-timeout-seconds torch-tensor-parallelism=1200 \
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
  --pipeline-chunks-per-step 16 \
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
  --harness-tool-timeout-seconds 6900 \
  --evaluation-every 5 \
  --evaluation-concurrency 24 \
  --dcp-save-interval 1 \
  --shuffle \
  --no-cleanup-on-exit \
  --wandb-entity myh97 \
  --wandb-project kimi-k3-fullparam-harbor \
  --wandb-run-name <run-name>
```

OpenCode title and summary requests do not carry tools and are logged as
auxiliary calls. Tool-bearing turns are trainable. Their exact sampled token IDs,
log probabilities, optional routing matrices, history decisions, and trainable
segment shapes are written under `_fireworks_trajectories/`. A history rewrite
starts a new segment within the same logical rollout; it does not create another
GRPO completion or reward.

### 100-step Kimi-K3 convergence run

This run removes the ten measured long-tail tasks, fails unless the resulting
fast-task pool has exactly 79 members, and uses seed `20260808` to reserve 20%
(16 tasks) exclusively for evaluation. The remaining 63 tasks are cycled to
1,600 prompt groups: 100 optimizer steps at 16 groups per step and 12,800
training trajectories at eight completions per group. Each evaluation contains
128 trajectories (16 fixed tasks x eight completions) and runs every five
optimizer steps. The exact split is saved to `task-split.json` before any E2B
template build or rollout begins.

All other arguments remain identical to the completed convergence run above.
The deliberate changes are the disjoint holdout, 100-step horizon, gradient-norm
threshold `100`, 131,072-token per-call output limit, and DCP cadence of two
optimizer steps.

```bash
RUN_DIR=/shared/yuedong/kimi-k3-harbor-convergence/<run-name>
ulimit -n 65536

uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --tokenizer-revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --renderer-name kimi_k3_preserve_thinking \
  --trainer-job-id efq8pkpso5e2x1ks \
  --deployment-id k3-terminalbench-gradclip-20260915 \
  --deployment-shape accounts/fireworks/deploymentShapes/kimi-k3-rl-mercor-gb300-fp4-w16-p4-tp4-dp4-pair1500/versions/n852kghu \
  --harbor-dataset /shared/yuedong/kimi-k3-harbor-convergence-data/terminal-bench-opencode-e2b-v12 \
  --harbor-trials-dir "$RUN_DIR/trials" \
  --log-path "$RUN_DIR" \
  --harbor-environment e2b \
  --harbor-trial-config training/examples/rl/harbor/recipes/terminal_bench/two_hour_trial.yaml \
  --e2b-task-memory-mb rstan-to-pystan=16384 \
  --e2b-task-verifier-timeout-seconds torch-tensor-parallelism=1200 \
  --max-concurrent-trials 128 \
  --exclude-task extract-moves-from-video \
  --exclude-task path-tracing \
  --exclude-task install-windows-3.11 \
  --exclude-task train-fasttext \
  --exclude-task schemelike-metacircular-eval \
  --exclude-task make-doom-for-mips \
  --exclude-task winning-avg-corewars \
  --exclude-task caffe-cifar-10 \
  --exclude-task mcmc-sampling-stan \
  --exclude-task qemu-alpine-ssh \
  --expected-task-pool-size 79 \
  --evaluation-holdout-fraction 0.2 \
  --cycle-selected-tasks \
  --task-seed 20260808 \
  --max-rows 1600 \
  --epochs 1 \
  --completions-per-prompt 8 \
  --prompt-groups-per-step 16 \
  --pipeline-chunks-per-step 16 \
  --min-group-size 8 \
  --max-incomplete-group-retries 2 \
  --lora-rank 0 \
  --learning-rate 1e-6 \
  --kl-beta 0 \
  --max-head-offpolicy-versions 0 \
  --policy-loss gspo \
  --no-router-replay \
  --grad-accumulation-normalization num_sequences \
  --grad-clip-norm 100 \
  --eps-clip 0.0003 \
  --eps-clip-high 0.0004 \
  --tis-cap 5 \
  --max-seq-len 262144 \
  --max-completion-tokens 131072 \
  --sample-timeout 7200 \
  --harness-tool-timeout-seconds 6900 \
  --evaluation-every 5 \
  --evaluation-concurrency 24 \
  --dcp-save-interval 2 \
  --shuffle \
  --no-cleanup-on-exit \
  --wandb-entity myh97 \
  --wandb-project kimi-k3-fullparam-harbor \
  --wandb-run-name <run-name>
```
