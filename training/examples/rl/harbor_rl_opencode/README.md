# Harbor RL with OpenCode

This example keeps the `async_rl_loop` contract unchanged and uses Harbor's
native `Trial` lifecycle. Fireworks supplies a recording OpenAI endpoint used
by OpenCode inside the task container. Harbor resolves the task, creates the
local Docker environment, runs the verifier, and returns the reward.

The multi-turn responsibility split follows
[Slime's coding-agent RL example](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl):
the example rollout owns the agent harness, environment, grading, retries, and
final keep/drop decision; shared agent utilities own token-exact turn capture
and trajectory materialization; the RL loop owns fan-out, grouping, advantages,
and optimization. Fireworks uses `RolloutRun` as the logical-rollout boundary
instead of Slime's `rollout_id`, but split physical segments retain that same
logical identity.

This is deliberately not a second Harbor implementation. Full Harbor task
configs are carried in dataset rows, and a native `TrialConfig` template is
passed through without reducing Harbor's verifier, timeout, artifact, resource,
or network settings. The adapter owns only the task and trial identity, its
OpenCode agent, and the local Docker lifecycle.

## Install

Harbor 0.20 requires Python 3.12 or newer.

```bash
cd training
uv sync
uv pip install --python .venv/bin/python 'harbor>=0.20,<0.21' 'dirhash>=0.5,<1'
```

`dirhash` is used only to verify the pinned DABstep task manifest. It is not
required for other Harbor datasets.

Use a dedicated cookbook environment for this example. Harbor 0.20 and the
optional Eval Protocol dependency currently require incompatible LiteLLM
versions.

Harbor can resolve a registry dataset directly:

```bash
uv run harbor datasets download terminal-bench@2.0 \
  -o ~/.cache/harbor/tasks/terminal-bench-2.0/
```

The example defaults to the Harbor registry dataset `terminal-bench@2.0`. It
also accepts a downloaded dataset directory or a single task directory.

For a large local-Docker OpenCode run, bake one pinned CLI version into the
task images instead of installing Node and OpenCode in every rollout:

```bash
uv run python -m \
  training.examples.rl.harbor.prepare_opencode_tasks \
  --source ~/.cache/harbor/tasks/terminal-bench-2.0 \
  --destination ~/.cache/harbor/tasks/terminal-bench-opencode \
  --opencode-version <pinned-version> \
  --internal-network harbor-terminal-noegress
```

The destination must be new. The script preserves each task's Docker and
network configuration by default and changes only its copied Dockerfile. With
`--internal-network`, it creates or validates a Docker `--internal` network and
writes an isolated Compose file into each copy. The task container can still
reach the host-side recording endpoint through its link gateway, but it cannot
reach the public network.

Both training entrypoints default to OpenCode `1.18.8`. Prepare images with
that same pin, or pass the same explicit `--opencode-version` to preparation
and training. Every trial verifies the baked CLI version before use; a mismatch
fails with a rebuild instruction instead of silently changing the harness.

## Train

```bash
export FIREWORKS_API_KEY=...

uv run python -m training.examples.rl.harbor_rl_opencode.train \
  --renderer-name qwen3_5_interleaved \
  --harbor-dataset ~/.cache/harbor/tasks/terminal-bench-opencode \
  --harbor-trial-config harbor-docker.yaml \
  --output-model-id accounts/<account>/models/<output>
```

Use `--harbor-trials-dir` to retain Harbor trial results and logs. Without it,
each rollout uses a temporary local trial directory while Harbor still tears
down the Docker environment. When it is set, the adapter also writes one
compressed trace per attempt under `_fireworks_trajectories/`. Each trace keeps
the raw OpenCode messages and tools, rendered tokens, completion logprobs and
routes, history-match decisions, packed-segment shapes, verifier result, and the
shared trajectory analyzer summary. Policy authorization headers and session
keys are never written to that trace.

Malformed tool-call formatting follows Slime's behavior and falls back to the
raw decoded assistant text while preserving the sampled tokens and logprobs.
Prompt-token, logprob, and requested Router Replay alignment failures remain
strict: they invalidate the attempt. The rollout retries the complete Harbor
attempt three times by default and returns `None` after the fourth failure, so
the async loop discards it instead of training a synthetic zero reward.
Ordinary agent timeouts and nonzero exits remain valid task outcomes when
Harbor produced a numeric verifier reward.

Full request traces are debugging artifacts and can be large at long context
lengths. The recording endpoint captures those messages, token arrays,
logprobs, and routes only when `--harbor-trials-dir` is set. Normal training
retains only the token-exact turn state required to build the loss plus small
matching counters.

For native Harbor configuration, pass a YAML file through
`--harbor-trial-config`:

```yaml
timeout_multiplier: 1.0
agent:
  override_timeout_sec: 600
environment:
  type: docker
  override_cpus: 4
  override_memory_mb: 8192
  kwargs: {}
verifier:
  override_timeout_sec: 300
artifacts:
  - /logs/result.json
```

The adapter validates this mapping with Harbor's `TrialConfig`. It always
overrides `task`, `trial_name`, the agent's `name` / `import_path` /
`model_name` / `kwargs`, `environment.type`, and `environment.delete` because
those fields belong to the active Fireworks rollout. Harbor removes each
per-trial Docker image tag, container, network, and volume after verification.
Other native fields pass through. `--harbor-trials-dir` takes precedence over
`trials_dir` in the YAML. Non-Docker environments, install-only trials, and
regrade/source trials are rejected explicitly.

Each `rollout_fn` call returns one logical `RolloutRun` or `None`. A failed call
may create multiple fresh Harbor trials within its bounded retry budget;
sessions, policy keys, and containers are never reused across attempts.
Sampling and trace-integrity failures invalidate the whole attempt, so partial
turns are retried and then discarded instead of being trained with a verifier
reward. Valid agent outcomes keep the reward returned by Harbor.
`openai_policy.py` owns OpenCode history matching and resolves each trainable
request to either the current parent or a new root. The shared token-level
training session records only exact prompt/output tokens, logprobs, and explicit
parent node IDs, then materializes the selected root-to-leaf paths. Title and
summary calls without tools are sampled for the agent but excluded from the
task objective. Prompt tokens, tool results, shared generated prefixes, and
re-emitted context stay masked where they are not training targets.

The logical `RolloutRun.run_id`, the local OpenCode bearer token, and the
serving-affinity key are separate identities. A fresh opaque affinity key is
created for each Harbor environment attempt and is reused by all model calls in
that attempt; it is the only one forwarded as the sampling `user` field. This
key provides affinity for separate sampling requests. The serving layer—not the
example—owns active-request KV, prompt-cache namespaces, and hotload reset
semantics.

All segments created by a history wipe or token drift retain the same
`RolloutRun.run_id` and Harbor reward. `async_rl_loop` therefore computes one
GRPO advantage for the environment trajectory and broadcasts it to the split
segments; segmentation does not create extra group members. Completion-only
Router Replay remains
aligned through inter-turn context with empty routing entries on masked tokens.
The loop continues to own rollout fan-out, grouping, policy-version admission,
optimization, and trainer/deployment lifecycle.

## Pinned DABstep reproduction

For a DABstep run, keep the provenance manifest outside the repository with the
prepared tasks. The manifest fixes the 67 training tasks, eight holdouts, six
sampling-calibration tasks, profile, and content hash of every selected task.
Both sampling and training verify those hashes before creating a Fireworks
resource.

Prepare DABstep with its dataset-specific wrapper before generating those
content hashes:

```bash
uv run python -m \
  training.examples.rl.harbor_rl_opencode.prepare_dabstep_tasks \
  --source ~/.cache/harbor-dabstep/tasks/source \
  --destination ~/.cache/harbor-dabstep/tasks/opencode \
  --opencode-version <pinned-version> \
  --internal-network harbor-dabstep-noegress
```

The upstream DABstep numeric scorer's regular expression omits leading signs,
so `-2.18` can incorrectly match a `+2.18` reference. This wrapper makes the
copied scorer sign-sensitive and fails closed if the upstream scorer no longer
has the known form. Generate or refresh the external manifest after this step;
the corrected scorer intentionally changes each task's content hash.

First run the six-task sampling gate through the shared serverless sampling
pool. This creates neither a trainer job nor a dedicated deployment:

```bash
uv run python -m training.examples.rl.harbor_rl_opencode.train_serverless \
  --sampling-only \
  --manifest ~/.cache/harbor-dabstep/manifests/dabstep.json \
  --harbor-dataset ~/.cache/harbor-dabstep/tasks/opencode \
  --harbor-trial-config ~/.cache/harbor-dabstep/trial.yaml \
  --wandb-entity <entity>
```

Then use the same rollout function for the fixed-size async-RL run:

```bash
uv run python -m training.examples.rl.harbor_rl_opencode.train_serverless \
  --manifest ~/.cache/harbor-dabstep/manifests/dabstep.json \
  --harbor-dataset ~/.cache/harbor-dabstep/tasks/opencode \
  --harbor-trial-config ~/.cache/harbor-dabstep/trial.yaml \
  --max-rows 320 \
  --wandb-entity <entity>
```

The serverless entrypoint fixes the audited K3 defaults: rank-64 LoRA, LR
`3e-5`, 524,288-token context, 32,768 tokens per OpenCode turn, 8 completions x
8 groups, two client-GRPO forward/backward chunks with default token-level TIS,
one optimizer mutation, zero off-policy versions, completion-only Router
Replay, `num_loss_tokens` gradient normalization, and three full-rollout
retries before discard. Rollout admission stays on the coordinator's adaptive
default. Independently, the shared Harbor adapter limits active local trials to
24 so Docker environment capacity does not become sampler concurrency policy.
The local Harbor environment and history-rewrite-aware rollout function are
identical in sampling and training.

Use `--lora-rank`, `--adam-beta2`, `--adam-epsilon`, and `--weight-decay` for
explicit optimizer experiments. `--evaluation-interval` controls the shared
holdout cadence, while `--dcp-save-interval` writes resumable model-and-optimizer
states at that optimizer-step interval in addition to the final checkpoint.
Set `--max-seq-len` to the selected serverless trainer pool's advertised
maximum; client configuration cannot increase a pool trainer's context limit.

Both Harbor entrypoints resolve one evaluation configuration before training
and reuse it throughout the run. Evaluation fan-out always equals training
rollout fan-out; there is no separate eval fan-out setting. The fixed
serverless recipe evaluates eight completions for each of eight holdout tasks
at the initial step, every three optimizer steps, and the final step, including
a non-periodic final step. Sampling parameters, concurrency, renderer, rollout
function, and Harbor grader do not change between evaluations, so each point
attempts 64 logical trajectories and is directly comparable. Its
`--holdout-concurrency` controls only execution parallelism. The generic
entrypoint uses `--completions-per-prompt`, `--holdout-every`, and
`--holdout-concurrency` for the corresponding settings.

The async loop owns initial, periodic, and actual-final scheduling and invokes
the same rollout with `evaluation=True`. OpenCode evaluation uses the supplied
holdout row directly and never reads or updates the adaptive training selector.

## Metric semantics

The async loops put `train/*`, `rollout/*`, `perf/*`, `async/*`, and `eval/*`
on the optimizer-step axis. `producer/*` is asynchronous scheduler state and
uses its own `producer/event` axis.

| Metrics | Source and meaning |
| --- | --- |
| `eval/attempted_trajectories` | Fixed holdout task count times the normal rollout fan-out; 64 with the pinned defaults. |
| `eval/completed_trajectories`, `failed_trajectories`, `no_trajectory` | Counts calls that returned a non-empty `RolloutRun`, raised, or returned `None`. Eval has no optimizer acceptance/rejection stage. |
| `eval/reward`, `eval/task/*` | Mean Harbor verifier `reward` over completed runs, globally and per task. |
| `eval/trainable_tokens_*` | Mean/min/max loss-bearing assistant tokens per logical run, derived from the returned segments. |
| `eval/history_wipes`, `eval/append_token_mismatches` | Structured-history `WIPE` decisions and exact-token ancestry splits. A history wipe does not claim that OpenCode compacted; a token mismatch starts another training segment without discarding sampled output. |
| `rollout/raw_reward`, `rollout/raw_samples` | Logical-run rewards assembled for the optimizer batch before dynamic filtering. |
| `rollout/filtered_reward`, `rollout/filtered_samples` | Logical-run rewards that remain in the `PromptGroup` objects sent to training. Split segments do not increase these counts. |
| `rollout/trainable_tokens_*` | Mean/min/max loss-bearing assistant tokens per logical run in the trained groups, derived by the shared rollout adapter. |
| `rollout/history_wipes`, `rollout/append_token_mismatches` | Total history wipes and exact-token ancestry splits in trained logical runs. Split segments retain the same reward, group member, and advantage. |
| `train/local_input_sequences:sum` | Trainer-reported physical input sequences, summed over forward/backward chunks. This counts packed segments, not logical trajectories. |
| `train/inference_k1`, `train/inference_k3` | Client-loss observability over loss-masked tokens. For `d = log p_train - log p_raw_inference`, these are the per-sequence means of `d` and `exp(d) - d - 1`, averaged across sequences. They do not affect the loss. |
| `producer/*` | Coordinator counters and capacity gauges sampled while rollout production is active. |

Training writes a final model-and-optimizer DCP checkpoint, sampler snapshot,
adaptive-selector state, task-selection ledger, and run-state JSON. A later
stage resumes only when `--resume-from`, `--row-offset`, `--step-offset`, and
`--selector-state-in` are supplied together. W&B-enabled continuation also
requires the prior `--wandb-run-id`; it resumes the same dashboard instead of
creating a second logical run.

The adapter supports completion-only Router Replay. A trial with a valid Harbor
verifier reward remains a task outcome even when Harbor records an agent
exception. A missing or invalid reward, provider/environment failure, or
unusable token trace retries the complete rollout and is discarded after the
configured budget. `--rollout-retries` controls the number of retries after the
initial attempt. The default never converts failures into task reward zero;
`--terminal-failure-reward 0` explicitly enables that algorithm choice for
terminal agent failures without a verifier reward. Unexpected ordinary
exceptions are logged with their traceback and discarded as failed
trajectories; control-plane cancellation still propagates. A discarded
trajectory is counted as an attempted, unsolved task by the adaptive selector,
but it never enters GRPO reward or advantage computation. Selector statistics
are finalized from the first row attempt; later incomplete-group retries do not
rewrite them. Producer telemetry reports drops as
`producer/trajectory_drops_total`.
