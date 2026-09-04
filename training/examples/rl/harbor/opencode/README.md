# Harbor RL with OpenCode

This example keeps the `async_rl_loop` contract unchanged and uses Harbor's
native `Trial` lifecycle. Each Docker container or E2B sandbox runs an SDK
`TITOSidecar`; OpenCode talks only to an environment-local loopback endpoint,
and the sidecar calls Fireworks inference. Harbor resolves the task, runs the
verifier, collects the compact/debug artifacts, and returns the reward.

The shared TITO/Harbor/harness/sandbox ownership rules are documented in
[`/skills/configure/references/rl-agentic.md`](/skills/configure/references/rl-agentic.md).
OpenCode owns only its agent adapter and event conventions; the sidecar contains
no OpenCode-specific logic.

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
OpenCode agent, and the selected Harbor environment lifecycle.

## Install

The pinned Harbor 0.21 requires Python 3.12 or newer. Install the E2B extra
only when using the remote backend:

```bash
cd training
uv sync
uv pip install --python .venv/bin/python 'harbor==0.21.0' 'dirhash>=0.5,<1'
# E2B only:
uv pip install --python .venv/bin/python 'harbor[e2b]==0.21.0'
```

`dirhash` is used only to verify the pinned DABstep task manifest. It is not
required for other Harbor datasets.

Use a dedicated cookbook environment for this example. Harbor and the
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
  training.examples.rl.harbor.opencode.prepare_tasks \
  --source ~/.cache/harbor/tasks/terminal-bench-2.0 \
  --destination ~/.cache/harbor/tasks/terminal-bench-opencode \
  --opencode-version <pinned-version>
```

The destination must be new. The script preserves each task's Docker and
network configuration and changes only its copied Dockerfile and prebuilt-image
selection. The environment must permit outbound HTTPS to the selected Fireworks
inference endpoint because the sidecar is inside that environment. An internal
no-egress Docker network is therefore not a supported TITO configuration.

For DeepSWE, use the dataset wrapper so the complete clean checkout, Git
revision, task membership, source hashes, prepared hashes, and OpenCode version
are captured in one external manifest:

```bash
uv run python -m \
  training.examples.rl.harbor.recipes.deep_swe.prepare_tasks \
  --source-repository ./deep-swe \
  --destination ./runs/<run-id>/prepared/deep-swe \
  --manifest ./runs/<run-id>/deep-swe-manifest.json
```

Both training entrypoints default to OpenCode `1.18.8`. Prepare images with
that same pin, or pass the same explicit `--opencode-version` to preparation
and training. Every trial verifies the baked CLI version before use; a mismatch
fails with a rebuild instruction instead of silently changing the harness.

The same prepared Dockerfile is also the E2B template source; preparation does
not need a backend mode. E2B cannot consume Docker Compose tasks. Both backends
start the sidecar on an ephemeral loopback port inside the agent environment,
so E2B needs no callback URL, tunnel, fixed port, or Fireworks-hosted stateful
service. The sidecar receives a disposable inference credential through the
Harbor agent configuration; the collected `result.json` is redacted before it
is retained.

Harbor names E2B templates from the task-environment content hash. The adapter
therefore allows Harbor to reuse an existing template instead of forcing every
rollout member to rebuild it. Before a high-fanout run, prewarm every distinct
task template and verify the cache hit so concurrent members do not race on the
first build. Harbor deletes each E2B sandbox after its trial, but Harbor 0.21
does not delete the shared templates. Inventory and remove obsolete templates
with E2B account tooling after the cohort is fully stopped; never delete a
template while active trials may reuse it.

## Train

The environment-sidecar runtime in this change currently includes one live
model implementation: GLM-5.2 with
`glm_moe_dsa_preserve_thinking`. The renderer registry also retains offline
characterization for additional model families, but that does not make those
families supported by the sidecar. A different model/template pair needs its own
lightweight conversation renderer, tokenizer-bound certificate, parser and
complete-render/stop/truncation tests, exact sampled-array checks, and live
validation before use; otherwise wait for that support. Unsupported pairs
fail closed.

Use the same rollout function without creating or attaching a trainer for a
sampling-only smoke against an existing deployment:

```bash
export FIREWORKS_API_KEY=...

uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --sampling-only \
  --base-model accounts/<account>/models/<model> \
  --tokenizer-model <tokenizer> \
  --renderer-name <certified-renderer> \
  --deployment-id accounts/<account>/deployments/<deployment> \
  --harbor-dataset ./prepared-tasks \
  --harbor-task <exact-task-name> \
  --harbor-trials-dir ./runs/<run-id>/trials \
  --log-path ./runs/<run-id>
```

Repeat `--harbor-task` to freeze an explicit cohort and set
`--completions-per-prompt` for independent members. Sampling-only mode builds
no trainer, performs no hot-load, and adds no second concurrency semaphore;
`--max-concurrent-trials` remains the Harbor environment admission bound.

```bash
export FIREWORKS_API_KEY=...

uv run python -m training.examples.rl.harbor.recipes.train_opencode \
  --base-model accounts/<account>/models/<glm-5.2-model> \
  --tokenizer-model zai-org/GLM-5.2 \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --harbor-dataset ~/.cache/harbor/tasks/terminal-bench-opencode \
  --harbor-trial-config harbor-docker.yaml \
  --training-shape-id accounts/<account>/trainingShapes/<shape> \
  --output-model-id accounts/<account>/models/<output>
```

The rollout exposes two length controls. `--max-seq-len` is the model's total
prompt-plus-output window and the exact-boundary training-retention limit;
`--max-completion-tokens` caps one assistant turn. Set the total context from
the selected model/deployment's authoritative contract; the generic default is
not evidence that a model/template pair is supported.

Use `--harbor-trials-dir` to retain Harbor trial results and logs. Without it,
each rollout uses a temporary local trial directory while Harbor still tears
down the environment. Every trial collects a mandatory compact trajectory
artifact. `--tito-debug` additionally collects plain JSONL troubleshooting
events before teardown and publishes the complete reducible metric summary
under `tito/debug/*`. Without that flag, W&B receives only the compact
production `tito/*` dashboard. The debug events are supplemental; the compact
`.tito` artifact remains authoritative for training and routine analysis.
Policy authorization headers and inference credentials are never persisted.

Malformed structured output is returned as a typed model outcome unless the
renderer explicitly certifies a protocol-safe lossless text fallback. It is
counted under `parser/model_malformed` and `calls/model_malformed`, not as an
internal sidecar failure. Prompt-token, logprob, and requested Router Replay
alignment failures remain strict: they invalidate the attempt. The rollout
retries the complete Harbor attempt three times by default and returns `None`
after the fourth failure, so the async loop discards it instead of training a
synthetic zero reward.
Ordinary agent timeouts and nonzero exits remain valid task outcomes when
Harbor produced a numeric verifier reward.

For native Harbor configuration, pass a YAML file through
`--harbor-trial-config`:

```yaml
timeout_multiplier: 1.0
agent:
  override_timeout_sec: 7200
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
`model_name` / `kwargs`, and `environment.delete` because those fields belong
to the active Fireworks rollout. `environment.type` is always selected by
`--harbor-environment`, so one YAML can be reused across Docker and E2B; an
explicit type in the YAML is only a default and is overridden. Harbor removes
each per-trial Docker resource or E2B sandbox after verification.
The default per-tool timeout is 900 seconds and must remain below the resolved
agent timeout. Override it explicitly with
`--harness-tool-timeout-seconds` for shorter trial configurations.
Other native fields pass through. `--harbor-trials-dir` takes precedence over
`trials_dir` in the YAML. Environments other than Docker and E2B, E2B Compose
tasks, install-only trials, and regrade/source trials are rejected explicitly.

Each `rollout_fn` call returns one logical `RolloutRun` or `None`. A failed call
may create multiple fresh Harbor trials within its bounded retry budget;
sessions, policy keys, and containers are never reused across attempts.
Sampling and trace-integrity failures invalidate the whole attempt, so partial
turns are retried and then discarded instead of being trained with a verifier
reward. Valid agent outcomes keep the reward returned by Harbor.
The environment-local `TITOSidecar` defaults to full-history prompt rendering.
Pass `--tito-prompt-mode incremental` to opt into the **experimental** reuse of
the prior exact checkpoint and join a model-specific suffix before inference.
Incremental mode does
not compare that prompt against a full replay; an unsupported history or token
junction falls back to a full-rendered new segment, or rejects under strict
policy. In full-history mode, an exact-prefix prompt extends the active segment,
while bounded latest-response drift may be realigned and masked. The cookbook materializes exact prompt/output tokens and
aligned logprobs/R3 without decoding and retokenizing sampled output. Title and
summary calls without tools are classified as auxiliary and cannot mutate the
policy lineage. Prompt tokens, tool results, and newly introduced prompt context remain
masked; sampled policy completion tokens are trainable.

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
  training.examples.rl.harbor.recipes.dabstep.prepare_tasks \
  --source ~/.cache/harbor-dabstep/tasks/source \
  --destination ~/.cache/harbor-dabstep/tasks/opencode \
  --opencode-version <pinned-version>
```

The upstream DABstep numeric scorer's regular expression omits leading signs,
so `-2.18` can incorrectly match a `+2.18` reference. This wrapper makes the
copied scorer sign-sensitive and fails closed if the upstream scorer no longer
has the known form. Generate or refresh the external manifest after this step;
the corrected scorer intentionally changes each task's content hash.

First run the six-task sampling gate through the shared serverless sampling
pool. This creates neither a trainer job nor a dedicated deployment:

```bash
uv run python -m training.examples.rl.harbor.recipes.dabstep.train \
  --sampling-only \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --manifest ~/.cache/harbor-dabstep/manifests/dabstep.json \
  --harbor-dataset ~/.cache/harbor-dabstep/tasks/opencode \
  --harbor-trial-config ~/.cache/harbor-dabstep/trial.yaml \
  --wandb-entity <entity>
```

Then use the same rollout function for the fixed-size async-RL run:

```bash
uv run python -m training.examples.rl.harbor.recipes.dabstep.train \
  --base-model <qualified-base-model> \
  --tokenizer-model <qualified-tokenizer> \
  --renderer-name glm_moe_dsa_preserve_thinking \
  --manifest ~/.cache/harbor-dabstep/manifests/dabstep.json \
  --harbor-dataset ~/.cache/harbor-dabstep/tasks/opencode \
  --harbor-trial-config ~/.cache/harbor-dabstep/trial.yaml \
  --max-rows 320 \
  --wandb-entity <entity>
```

The serverless entrypoint requires an explicitly qualified model, tokenizer,
and renderer contract. V1 ships the GLM-5.2 sidecar implementation; names that
are only characterized offline fail closed before a trial starts. The recipe
otherwise fixes the audited defaults: rank-64 LoRA, LR
`3e-5`, a 524,288-token inference window and training-retention limit, 32,768
tokens per OpenCode turn, 8 completions x 8 groups, two client-GRPO
forward/backward chunks with default token-level TIS, one optimizer mutation,
zero off-policy versions, completion-only Router Replay, `num_loss_tokens`
gradient normalization, and three full-rollout retries before discard. Rollout
admission stays on the coordinator's adaptive default. Independently, the
shared Harbor adapter limits active local trials to 24 so Docker environment
capacity does not become sampler concurrency policy. The local Harbor
environment and history-rewrite-aware rollout function are identical in
sampling and training.

Use `--lora-rank`, `--adam-beta2`, `--adam-epsilon`, and `--weight-decay` for
explicit optimizer experiments. `--evaluation-interval` controls the shared
holdout cadence, while `--dcp-save-interval` writes resumable model-and-optimizer
states at that optimizer-step interval in addition to the final checkpoint.
Set `--max-seq-len` within both the inference model's authoritative context
window and the selected trainer pool's advertised maximum; client
configuration cannot increase either limit.

Both Harbor entrypoints resolve one evaluation configuration before training
and reuse it throughout the run. Evaluation fan-out always equals training
rollout fan-out; there is no separate eval fan-out setting. The fixed
serverless recipe evaluates eight completions for each of eight holdout tasks
at the initial step, every three optimizer steps, and the final step, including
a non-periodic final step. Sampling parameters, concurrency, renderer, rollout
function, and Harbor grader do not change between evaluations, so each point
attempts 64 logical trajectories and is directly comparable. Its
holdout-concurrency setting controls only execution parallelism. The generic
entrypoint uses `--completions-per-prompt`, `--evaluation-every`, and
`--evaluation-concurrency` for the corresponding settings. Without a DABstep
manifest, repeat `--evaluation-task` to freeze a fixed evaluation set from the
same loaded dataset; those rows remain eligible for training.

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
| `train/custom_forward_reused` | `1` when the old-policy forward also supplies the custom-loss input, avoiding a duplicate standalone forward; otherwise `0`. This optimization does not change K1/K3 reduction semantics. |
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
