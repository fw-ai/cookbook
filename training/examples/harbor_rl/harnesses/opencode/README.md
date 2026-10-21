# Harbor RL with OpenCode

OpenCode runs inside each Harbor environment (local Docker or E2B) next to an
environment-local SDK `TITOSidecar`. OpenCode talks only to the sidecar's
loopback endpoint; the sidecar calls Fireworks inference and records exact
prompt/completion tokens. Harbor resolves the task, runs the verifier, collects
the trajectory artifact, and returns the reward.

The shared TITO/Harbor/harness/sandbox ownership rules are documented in
[`/skills/fireworks-training/references/rl-agentic.md`](/skills/fireworks-training/references/rl-agentic.md).
This package owns only the OpenCode-specific pieces:

| File | Role |
| --- | --- |
| `agent.py` | `ConfigurableOpenCode`: installs the sidecar, writes the OpenCode provider config, runs the CLI |
| `rollout.py` | `make_rollout_fn`: the OpenCode `HarborHarness` (tool-bearing calls are policy, title/summary calls are auxiliary) |
| `artifacts.py` | Counts bash tool timeouts from OpenCode's event log |
| `prepare_tasks.py` | Bakes the pinned OpenCode CLI into copied Harbor task images |

Everything else — the typed `HarborSettings` (trial, retry and TITO sidecar
settings), retries, trial admission, artifact materialization — is the
harness-neutral
[`tito/rollout.py`](../../tito/rollout.py) `HarborRollout` shared with Pi,
Mini-SWE-Agent, and mimoagent.

## Install

The pinned Harbor 0.21 requires Python 3.12 or newer. Install the E2B extra
only when using the remote backend:

```bash
cd training
uv sync
uv pip install --python .venv/bin/python 'harbor==0.21.0' 'dirhash>=0.5,<1'
# E2B only. This pin supports HTTP version selection and the cookbook's
# pyqwest==0.10.0; do not independently upgrade to e2b 2.56 (pyqwest>=0.12).
uv pip install --python .venv/bin/python 'harbor[e2b]==0.21.0' 'e2b==2.55.0'
```

`dirhash` is used only to verify the pinned DABstep task manifest. Use a
dedicated cookbook environment for this example: Harbor and the optional Eval
Protocol dependency require incompatible LiteLLM versions.

## Prepare tasks

Harbor can resolve a registry dataset directly:

```bash
uv run harbor datasets download terminal-bench@2.0 \
  -o ~/.cache/harbor/tasks/terminal-bench-2.0/
```

Bake one pinned CLI version into the task images instead of installing Node
and OpenCode in every rollout:

```bash
uv run python -m training.examples.harbor_rl.harnesses.opencode.prepare_tasks \
  --source ~/.cache/harbor/tasks/terminal-bench-2.0 \
  --destination ~/.cache/harbor/tasks/terminal-bench-opencode
```

The destination must be new. Preparation changes only each copied Dockerfile
and prebuilt-image selection; Docker and network configuration are preserved.
The environment must permit outbound HTTPS to the Fireworks inference endpoint
because the sidecar runs inside it. OpenCode is pinned to `1.18.8`; every trial
verifies the baked CLI version and fails with a rebuild instruction on
mismatch.

For DeepSWE, the dataset wrapper also records the checkout revision, task
membership, and source/prepared hashes in an external manifest:

```bash
uv run python -m training.examples.harbor_rl.tasks.deep_swe.prepare_tasks \
  --source-repository ./deep-swe \
  --destination ./runs/<run-id>/prepared/deep-swe \
  --manifest ./runs/<run-id>/deep-swe-manifest.json
```

The prepared Dockerfile is also the E2B template source. E2B cannot run Docker
Compose tasks. Harbor names E2B templates by the environment content hash and
reuses them; prewarm distinct templates before high fan-out, and delete
obsolete templates only after all trials stop (Harbor deletes per-trial
sandboxes, not templates).

## Train

The sidecar currently ships GLM-5.2 (`glm_moe_dsa_preserve_thinking`),
Qwen3.8-27B (`qwen3_8`), and Muse Glimmer 30B (`muse_glimmer`, full-history
only). Unsupported model/template pairs fail closed.

[`recipes/train_opencode.py`](../../train_opencode.py) trains on any
Harbor dataset with the dedicated `async_rl_loop`. Score an existing
deployment first; sampling-only mode creates no trainer and no hot-load:

```bash
export FIREWORKS_API_KEY=...

uv run python -m training.examples.harbor_rl.train_opencode \
  --sampling-only \
  --base-model accounts/<account>/models/<model> \
  --tokenizer-model <tokenizer> \
  --renderer-name <certified-renderer> \
  --deployment-id accounts/<account>/deployments/<deployment> \
  --harbor-dataset ~/.cache/harbor/tasks/terminal-bench-opencode \
  --harbor-task <exact-task-name> \
  --harbor-trials-dir ./runs/<run-id>/trials \
  --run-dir ./runs/<run-id>
```

Then train (omit `--trainer-job-id`/`--deployment-id` to let the SDK create
resources; `--skip-cleanup` keeps reattached resources alive):

```bash
uv run python -m training.examples.harbor_rl.train_opencode \
  --base-model accounts/<account>/models/<glm-5.2-model> \
  --tokenizer-model zai-org/GLM-5.2 \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --harbor-dataset ~/.cache/harbor/tasks/terminal-bench-opencode \
  --harbor-trial-config harbor-docker.yaml \
  --training-shape-id accounts/<account>/trainingShapes/<shape> \
  --evaluation-task <task-a> --evaluation-task <task-b> \
  --run-dir ./runs/<run-id>
```

`--max-seq-len` is the total prompt-plus-output window and the exact-boundary
training-retention limit; `--max-completion-tokens` caps one assistant turn.
Repeat `--harbor-task` to freeze an ordered cohort and `--evaluation-task` to
evaluate a fixed set every `--evaluation-every` optimizer steps (those rows
stay in training). `--harbor-trials-dir` retains Harbor results, logs, and the
compact `.tito` artifacts; without it each trial uses a temporary directory.
`--tito-debug` also collects JSONL troubleshooting events and publishes
`tito/debug/*` metrics. Credentials are never persisted.

The DABstep and Terminal-Bench recipes run OpenCode on the serverless loop; see
[`tasks/dabstep/`](../../tasks/dabstep/README.md) and
[`tasks/terminal_bench/`](../../tasks/terminal_bench/README.md).

## Harbor trial configuration

Pass a native Harbor `TrialConfig` YAML through `--harbor-trial-config`:

```yaml
timeout_multiplier: 1.0
agent:
  override_timeout_sec: 7200
environment:
  type: docker
  override_cpus: 4
  override_memory_mb: 8192
verifier:
  override_timeout_sec: 300
artifacts:
  - /logs/result.json
```

The adapter always overrides `task`, `trial_name`, the agent's `name` /
`import_path` / `model_name` / `kwargs`, `environment.delete`, and
`environment.type` (from `--harbor-environment`). Other native fields pass
through; `--harbor-trials-dir` takes precedence over `trials_dir`. The per-tool
timeout (`--harness-tool-timeout-seconds`, default 900) must stay below the
resolved agent timeout. Other environments, E2B Compose tasks, install-only
trials, and regrade/source trials are rejected.

## Rollout semantics

Each `rollout_fn` call returns one logical `RolloutRun` or `None`. A failed
attempt (provider/environment failure, missing reward, broken token trace)
retries with a fresh Harbor environment, sidecar trajectory, and credential up
to `--rollout-retries` times (default 3), then returns `None` so the loop drops
the sample instead of training a synthetic reward. Ordinary agent timeouts and
nonzero exits keep the verifier reward when Harbor produced one;
`--terminal-failure-reward 0` explicitly scores terminal failures that have no
verifier reward. Control-plane cancellation propagates.

The sidecar defaults to full-history prompt rendering. `--tito-prompt-mode
incremental` is an experimental opt-in that joins a model-specific suffix to
the prior exact checkpoint; an unsupported junction starts a new masked
segment. Segments split by history rewrites or token drift keep the same
`run_id`, reward, and GRPO advantage — segmentation never adds group members.
Prompt, tool-result, and newly introduced context tokens are masked; sampled
policy tokens are trainable. MoE models replay completion-only Router Replay
routes.

## Evaluation metrics

Evaluation runs the same rollout with `evaluation=True` over fixed rows
(`tito/evaluate.py`):

| Metric | Meaning |
| --- | --- |
| `eval/attempted_trajectories` | Evaluation rows times completions per prompt |
| `eval/completed_trajectories`, `eval/no_trajectory` | Calls that returned a run, or returned `None`/raised |
| `eval/reward` | Mean Harbor verifier reward over completed runs |
| `eval/task_reward_mean`, `_min`, `_max`, `eval/task_count` | Per-task mean reward summarized across tasks |
| `eval/trainable_tokens_mean`, `_min`, `_max` | Loss-bearing tokens per logical run |

Training metrics come from `async_rl_loop`; see
[`/skills/fireworks-training/references/rl-async.md`](/skills/fireworks-training/references/rl-async.md).
