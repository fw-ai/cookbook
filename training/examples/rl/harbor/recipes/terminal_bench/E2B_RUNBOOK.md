# Kimi-K3 Terminal-Bench E2B runbook

Use this checklist before launching the expensive convergence workload. A
single E2B image-build smoke test is not sufficient: it does not validate the
tokenizer bundle, TITO sidecar, or all task templates.

## Required launch invariants

| Check | Required value/evidence |
| --- | --- |
| Trainer | Existing healthy job from training shape `accounts/fireworks/trainingShapes/kimi-k3-262k-gb300/versions/rbb16rr5` |
| Rollout | Existing healthy deployment attached to that trainer |
| Tokenizer | `moonshotai/Kimi-K3` at revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| Renderer | `kimi_k3_preserve_thinking` |
| E2B templates | Every selected task alias can create a sandbox; no missing `default` tag |
| Trial smoke test | One real task produces a validated TITO trajectory artifact |
| Fanout | Shell open-file limit is at least `65536` before 128-way launch |
| Timeouts | Agent and verifier each allow 7200s; tool timeout remains below the agent timeout (`6900 < 7200`); known verifier-deadlock tasks use a scoped shorter verifier timeout |
| E2B resources | Each trial gets 4 CPUs and 8192 MB; `rstan-to-pystan` gets 16384 MB via `--e2b-task-memory-mb` |
| Secrets | Loaded from a mode-`0600` environment file; never written to commands, logs, or trial artifacts |

Do not restart the trainer or rollout to fix a client/E2B problem. Stop and
restart only the local harness unless remote health evidence requires more.

## Observed failures and fixes

| Symptom | Root cause | Fix | Verification gate |
| --- | --- | --- | --- |
| Debian Bullseye security `InRelease` expired | Two Terminal-Bench Dockerfiles used EOL Bullseye; E2B resolves the base before later Dockerfile repair commands | During task preparation, replace the final Bullseye base with Bookworm and the virtual `netcat` package with `netcat-openbsd` | Build and create a real E2B sandbox for `qemu-startup` |
| `netcat` has no installation candidate | Bookworm exposes `netcat` as a virtual package | Install `netcat-openbsd` | Same `qemu-startup` sandbox gate |
| `Invalid source path "/uv"` | E2B's Dockerfile builder does not accept the external-stage absolute `COPY --from=... /uv /uvx /bin/` form | Replace it with installation of the same pinned `uv` wheel version | Build and create a sandbox for `reshard-c4-data` |
| `tag 'default' does not exist for template ...` | The shared layer pinned NumPy 2.4.6, which cannot install on two Python 3.10 task images; E2B retained stale alias/tag metadata for the failed builds | Pin the Python-3.10-compatible NumPy 2.2.6, rebuild the affected alias, and retry only this exact sandbox-create 404 | Every alias must pass `AsyncSandbox.create(alias)`; alias/tag metadata alone is insufficient |
| `Verifier execution timed out after 900.0 seconds` | The Chrome-heavy `filter-js-from-html` verifier exhausts its 2 GB task default and stalls; extending only the agent timeout also leaves Harbor's independent verifier timeout at 900s | Give each E2B trial 4 CPUs and 8192 MB, and set both agent and verifier `override_timeout_sec` to 7200 | Replay the exact previously stalled candidate: it must finish verification and return a numeric reward; `producer/trajectory_drops_total` must remain zero |
| `build is not in waiting state` | Eight rollouts raced to build the same previously absent task alias | Prebuild each unique alias once; only then start rollout fanout | No template builds occur during the 128-way smoke launch |
| `renderer 'kimi_k3' has no production TITO certification` | The offline cookbook renderer name was passed to the production sidecar | Use `kimi_k3_preserve_thinking` | Sidecar readiness succeeds |
| `tokenizer does not match TITO certification` | The unpinned HF default resolved to revision `f831ab...`; certification is for `9f62e4e9...` | Pass the exact `--tokenizer-revision` above | Host and reloaded bundle fingerprints both equal `3d98398c...` |
| Inner timeout validation failure | Tool timeout equaled the outer trial timeout | Use `--sample-timeout 7200 --harness-tool-timeout-seconds 6900` | CLI validation passes before provisioning |
| Too many open files | 128 concurrent environments exceed a 1024-FD shell limit | Run `ulimit -n 65536` before Python | `/proc/<pid>/limits` reports `65536` |
| `ReconnectableClient.optim_step()` rejects `emit_grad_norm_metrics` | The recipe requested optimizer diagnostics that the reconnecting wrapper did not forward to the SDK client | Forward the optional argument through synchronous and asynchronous optimizer calls | Unit-test both wrapper paths, then require `train/grad_norm` and `train/grad_norm_post_clip` from a real optimizer step |
| Gradient clipping fails on mixed FSDP/EP `DTensor` meshes, followed by NCCL/CUDA OOM | The generic norm implementation combines dense `fsdp=128` and routed-expert `efsdp=16,ep=8` tensors before reducing their local squared norms | Use a trainer image with the mixed-mesh scalar-reduction fix; do not work around it by removing clipping | Complete forward/backward, report finite pre/post-clip norms, complete the optimizer step, hot-load its delta, and save DCP without a pod restart or OOM |
| Agent stops after starting a persistent service | A generated `nohup ... &` command can leave its wrapper shell attached to the healthy daemon, so OpenCode waits until the per-tool timeout even though the task service is ready | Confirm the daemon is healthy, then terminate only the orphaned wrapper shell; do not terminate the daemon, sandbox, client, trainer, or rollout | The existing OpenCode process resumes, writes its trajectory, and the verifier reaches the preserved daemon |
| `ConnectError: ... error reading a body ... timed out` or `peer closed connection without sending TLS close_notify` during a long tool call | The E2B command stream disconnected after opening; the sandbox workload may have completed even though Harbor could not receive its result | Treat only these typed failures with the exact Harbor/E2B command-stream traceback as recoverable | The retry uses a new sandbox and produces a checksum-valid TITO artifact |
| One member of an 8-rollout prompt group fails after its siblings complete | Rebuilding the whole group repeats seven expensive, already-valid E2B trials | Retain successful members in the producer and resubmit only missing indices at the same policy version | Unit test observes one call for successful members and two calls only for the failed member; the recovered group still contains all eight members |
| A repeated prompt group reads stale or colliding TITO files | Group-level retries reused the same Harbor trial directory even though trajectory-level retry counters reset | Give every physical Harbor attempt a unique artifact directory while preserving its logical rollout ID | Repeated logical rollouts have distinct trial paths and cannot consume prior-attempt artifacts |
| PyStan agent appears active indefinitely | Its main Python process is OOM-killed at 8 GiB while orphaned chain workers retain the command pipe | Prebuild and run only `rstan-to-pystan` with `--e2b-task-memory-mb rstan-to-pystan=16384` | The task alias launches with 16384 MB and all chain workers complete without a kernel OOM kill |
| Candidate distributed test hangs after partial verifier progress | Invalid candidate code deadlocks a multiprocess collective, so the verifier cannot reach its remaining tests | Keep the two-hour agent budget, but set `--e2b-task-verifier-timeout-seconds torch-tensor-parallelism=1200` | A hung verifier is discarded and replaced within 20 minutes; unrelated tasks retain the full verifier budget |
| The same `torch-tensor-parallelism` verifier times out on every retry | Retrying the same generated candidate cannot repair a deterministic collective deadlock | Preserve the scoped 1,200-second verifier bound, exclude the incomplete prompt group, and record all eight group members as dropped; fix or preflight this verifier before using the task in another convergence claim | The completed run excluded exactly one prompt group after three attempts; every trajectory admitted to training had a checksum-valid TITO artifact |
| `Sandbox not found` during artifact cleanup | Secondary cleanup after sandbox creation/build failed | Diagnose the earlier exception; do not treat cleanup noise as the root cause | Root exception is absent on rerun |
| `PyTorch was not found` | Informational Transformers warning in the lightweight sidecar | No fix required; TITO needs tokenizer utilities, not Torch | Ignore unless followed by a different fatal exception |

## Snapshot synchronization checks

Inspect **every** entry in the hot-load status endpoint's `replicas` array.
For this deployment there must be four peers, each with the requested
`current_snapshot_identity`, `readiness=true`, and `loading_state.stage=idle`.
Control-plane `READY` does not prove that the sampler weights finished loading.
The SDK version used in this run polls only `replicas[0]`, which hid errors
reported by the other peers; inspect their `readiness_reason` as well.

On 2026-09-15, the Mercor TP4/DP4 shape omitted
`FIREWORKS_P2P_COLLECTIVE_FUSED_BASE_EXCHANGE=1`. The initial full-snapshot request
timed out after 600 seconds; three peers reported
`ValueError('Tensors must be contiguous')` while another stayed in `updating`.
The non-fused exchange passes individual weight views to the collective; the
fused path exchanges contiguous byte ranges from weight slabs allocated at
startup. A full traceback identifying the specific tensor was unavailable.

Restoring the flag and actually recycling the rollout processes resolved the
observed failure with the same TP4/DP4 topology and image. The fresh snapshot
`step-0-15892378` then loaded in 57 seconds; the client's initial synchronization
timer, including trainer export, was 156.8 seconds. Verify new peer process
identities after a requested restart: a deployment extra-value update and a
control-plane `READY` transition alone did not establish that the old processes
had been replaced in this incident. Preserve the trainer throughout recovery.

The active command uses `--weight-sync-timeout 1800`. This extends observation
time only. An `error`/`internal_error` peer requires investigation even while
the client is still waiting; repeated snapshot posts or a larger timeout cannot
repair a non-contiguous collective input.

Use a fresh `WANDB_RUN_ID` for a new experiment. Reuse it only to resume that
experiment, after checking its config and checkpoint. A reused ID from an older
run initially displayed seven stale optimizer steps before this run had
completed its first synchronization. The corrected run is `2ja2bva6`.

## Trainer log observation

If a broad Cloud Logging query returns HTTP 500, treat it as an observation
failure, not a trainer failure. On this run, querying recent entries succeeded,
while error searches using only the nested trainer-ID label failed. Reading a
recent entry supplied the actual pod names; a resource-scoped error query then
succeeded. Example for the current run (replace the resource labels and start time
when investigating another run):

```sh
gcloud logging read \
  'resource.type="k8s_container" AND resource.labels.cluster_name="gmi-ap-taiwan-1" AND resource.labels.namespace_name="default" AND resource.labels.container_name="trainer" AND resource.labels.pod_name=~"^trainer-training-rlor-efq8pkpso5e2x1ks-0($|-follower-)" AND timestamp>="2026-09-15T02:54:06Z" AND jsonPayload.message=~"ERROR|Traceback|OutOfMemoryError|CUDA out of memory"' \
  --project=fw-ai-cp-prod --limit=10 \
  --format='json(timestamp,resource.labels.pod_name,jsonPayload.message)'
```

This searches message text because not every trainer stdout record has an
ERROR severity label. An empty successful response proves only that the query
found no matches in that window; it does not replace live process, heartbeat,
and optimizer/checkpoint checks. Do not restart a service due to a logging API
timeout or HTTP 500.

## Artifact archival safety

Monitor both the harness process's RSS/high-water mark and the host's
`MemAvailable` from `/proc/meminfo`; process RSS alone cannot detect pressure
from other users of this shared host. Do not use `MemFree` as the available
capacity: reclaimable filesystem cache can make it misleading. Record free disk
space on both `/` and the actual trial-storage mount, even with remote E2B:
completed artifacts are collected locally before upload.

`artifacts/tito/compact/COMPLETE` means the compact trajectory is ready, not
that Harbor has finished verification and written its final result. In the
2026-09-15 run, six trials had that marker but no `result.json` yet. Archive a
trial only after both the marker and a valid final `result.json` exist. Mark it
uploaded only after a successful full-directory transfer and confirmation of
both files at the destination. Prune local files only after that upload and a
successfully saved DCP checkpoint covering the consumed training row; never
delete active or uncheckpointed trials to recover disk space. If GCS credentials
expire, retain local artifacts and report the upload failure separately from RL
health.

## Background-server tool waits

The 2026-09-15 first batch included an `hf-model-inference` sample with no new
agent output for over 17 minutes. The sandbox had ample memory/disk and no
sidecar exception. Read-only inspection of OpenCode's SQLite `part` records
identified a still-running bash call with no explicit timeout:

```sh
cd /app && nohup python3 app.py > /app/server.log 2>&1 & sleep 8 && tail -5 /app/server.log
```

The server and its launcher shell remained alive after the foreground shell
exited. This compound-background-command pattern can retain an inherited output
pipe: a local reproduction using a two-second `sleep` returned foreground exit
code 0 within 0.3 seconds, but collecting captured output took 2.01 seconds.
That reproduces the pipe-lifetime hazard, not a full OpenCode fix. The configured
6,900-second default tool timeout applies when the model omits a timeout;
`sleep 8` is not an eight-second bound on the whole tool call.

The same run also had three `configure-git-webserver` evaluation samples
waiting over 30 minutes on compound commands that backgrounded `nohup python3
-m http.server`, then ran a short `sleep` and a `curl`/`ps` check. Their bash
parts remained `running` without an explicit timeout. In contrast, pending
cryptanalysis samples had active CPU-bound search processes. Classify these
separately rather than treating every long sample as an infrastructure failure.

For quiet trials, inspect the actual pending tool and sandbox processes before
blaming inference or E2B capacity. Record this as a harness/task interaction to
investigate; do not silently kill the background server, shorten the agreed
timeout, discard the sample, or restart RL. Any change to tool subprocess/output
handling needs its own regression test, including intended background-server
survival, before use in a subsequent run.

### Tool still marked running after a candidate-code crash

In the same run, evaluation trial
`harbor-opencode-largest-eigenval-0-11-3-05800feb-e6a8c97b` had a different
failure. The bash tool started at `2026-09-15T03:37:58Z`; its stored metadata
contained `Fatal Python error: Segmentation fault` at `/app/eigen.py:90` in
`_dominant_lapack`. At approximately 04:00 UTC, process inspection found no
remaining candidate-test Python process, but OpenCode's SQLite tool state was
still `running`. OpenCode and the TITO sidecar remained alive. This is not
evidence of a trainer crash, GPU OOM, or active numerical computation.

Source inspection found a matching failure mechanism in the pinned release:
[OpenCode v1.18.8 shell tool](https://github.com/anomalyco/opencode/blob/3c81a5d1ddceab377d9ad71c14899e6935333fdd/packages/opencode/src/tool/shell.ts)
races `handle.exitCode`, cancellation, and timeout with `Effect.raceAll`.
In Effect `4.0.0-beta.83`, that operator waits for the first **success**, ignoring
an early failure while other branches remain pending. The matching
`@effect/platform-node-shared` process adapter returns a failed effect for a
signal-terminated child. Consequently, such an exit can be hidden until timeout.

An isolated local reproduction with that exact Effect version and a real
signal-terminated child reproduced the delay: the child exited after 68 ms, but
the race returned `timeout` after 605 ms with a 600-ms timer. Racing a captured
termination outcome (`Effect.exit(handle.exitCode)`) instead returned the failure
after 68 ms. Normal exits 0 and 3 also passed. This validates the race mechanism,
not a complete fix in the live OpenCode binary; no runtime patch was applied.

Preserve the captured traceback, tool state, process inventory, and timestamps.
Do not repair the model's candidate code or silently label the sample completed.
Any integrated fix must retain output and report signal termination correctly,
with separate coverage for background-server survival, cancellation, and timeout.

### Command-stream failure and evaluation coverage

At 04:01:10 UTC, the waiting `hf-model-inference` training trial and one
`configure-git-webserver` evaluation trial failed with
`connectrpc.errors.ConnectError: Error reading content`, caused by a
`pyqwest._errors.StreamError` in the E2B command-event stream. This was not the
6,900-second tool timeout. The exact upstream connection-reset cause is unknown.

The training coordinator retained the seven valid group members and retried the
missing eighth member. Its new physical trial ended successfully with reward 1
at approximately 04:02:22 UTC. The cumulative drop and incomplete-group-retry
counters each became 1; these counters do not imply a seven-member optimizer
group was accepted. Verify completed batch membership separately.

Evaluation does not use that coordinator-level group retry. In
`tito/evaluate.py`, a rewardless `None` result contributes to `eval/no_trajectory`
and is excluded from the denominator of `eval/reward`. Always report attempted,
completed, failed, and no-trajectory counts with reward; an incomplete evaluation
must not be presented as full 128-trajectory holdout coverage. No live recovery
signal, task-code edit, or trainer/rollout/harness restart was applied here.

## Retry and progress counters

### Initial evaluation and step-publication gates

In `training/recipes/async_rl_loop.py`, initial evaluation starts concurrently
with training sampling. The loop consumes the training chunks and calls
`optimizer_step` **before** joining that evaluation. It then waits for evaluation
to finish **before** exporting/hot-loading the updated weights and publishing the
step metrics. This keeps evaluation on the previous rollout snapshot.

Consequently, a pending training sample can delay the optimizer, while a pending
evaluation sample can delay weight synchronization and the published step even
after the optimizer has completed. Check trainer operation logs as well as W&B;
absence of a new published step alone does not prove that no optimizer update
occurred. Neither an optimizer operation nor a published step proves a DCP save.
Do not bypass the evaluation join or hot-load new weights into its active trials.

Do not infer failures from a counter name alone. Use these metrics together:

| Metric | Meaning | Failure signal |
| --- | --- | --- |
| `producer/completion_refill_attempts_total` | Coordinator passes that look for more completed work and refill available capacity | None by itself; it grows during healthy long-running sampling |
| `producer/incomplete_group_retries_total` | Prompt groups resubmitted because too few valid completions survived | Any increase requires inspecting the affected group and its trial artifacts |
| `producer/trajectory_drops_total` | Individual trajectories discarded after rollout or validation failure | Any increase requires a typed root exception before proceeding |
| `producer/recoverable_errors_total` | Errors handled by the producer's explicit recovery policy | Confirm every increment matches an allowlisted transient failure |
| `tito/calls/upstream_retries` | Model-request retries inside TITO | A small nonzero count can be transient; sustained growth requires deployment/client-log inspection |

Before restarting a client, record these counters and preserve its run directory.
Never restart the trainer or rollout for a client-only failure. Resume only from
the latest server-confirmed DCP checkpoint and keep the same W&B run ID so the
step numbering remains monotonic. The 100-step run saves DCP every two steps;
an optimizer-step metric alone does not prove that a resumable checkpoint exists.

## Launch sequence

1. Run unit tests for task rewrites, timeout ordering, resource overrides, and the dedicated config.
2. Compute the fingerprint before and after serializing the pinned tokenizer;
   both must match the certification.
3. Build/repair every unique E2B task template without concurrent duplicate
   builds, and create then delete one sandbox from each alias.
4. Run one real Kimi TITO trial and require a validated trajectory artifact.
5. Launch the 128-way client against the existing trainer and rollout.
6. Confirm the first optimizer step and W&B metrics before considering the
   convergence run healthy.

For a failed launch, append the root exception and its verification gate to
this table before retrying. Preserve the failed run directory as evidence.
