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
| Distributed verifier hangs after partial progress | Candidate collectives or verifier/runtime teardown can block; inspect the live stack before assigning a cause | Keep the two-hour agent budget, with the configured `--e2b-task-verifier-timeout-seconds torch-tensor-parallelism=1200` | The verifier is bounded; unscored artifacts are retained and only missing group members are retried. Replacement sampling adds time beyond this verifier bound |
| The same `torch-tensor-parallelism` verifier times out on every retry | Retrying the same generated candidate cannot repair a deterministic collective deadlock | Preserve the scoped 1,200-second verifier bound, exclude the incomplete prompt group, and record all eight group members as dropped; fix or preflight this verifier before using the task in another convergence claim | The completed run excluded exactly one prompt group after three attempts; every trajectory admitted to training had a checksum-valid TITO artifact |
| `Sandbox not found` during artifact cleanup | Secondary cleanup after sandbox creation/build failed | Diagnose the earlier exception; do not treat cleanup noise as the root cause | Root exception is absent on rerun |
| `PyTorch was not found` | Informational Transformers warning in the lightweight sidecar | No fix required; TITO needs tokenizer utilities, not Torch | Ignore unless followed by a different fatal exception |

## Snapshot synchronization checks

### Reserved-rack launch preflight

Inspect node **labels and taints**, and the rendered GPU pod spec, before
sampling. A `fullparam-k3-rl=true` node selector restricts placement but does not
tolerate the separate `fullparam-k3-rl` taint. On 2026-09-15 this omission left
the fresh rollout pending although the reserved pool contained available nodes.
The deployment uses `extraNodeSelectors={"fullparam-k3-rl":"true"}` and
`additional_toleration_keys=fireworks.ai/global-reserved,fullparam-k3-rl`, in
addition to the existing `fireworks.ai/rftj` toleration. Verify current taints;
do not treat these historical values as a substitute for a live check.

`firectl-admin deployment update --extra-values ...` replaces the extra-values
map rather than merging individual keys. Fetch the existing deployment, modify
the full map, update with `--file`, and read it back. In particular, retain
fused-base exchange, heartbeat timeout, session routing, and all shape values.
Use `maxSurge=0,maxUnavailable=1` for an explicitly authorized replacement that
must fit the same 64 GPUs; this allows downtime and is not permission to restart
a running experiment. Confirm the latest rendered pod revision, not only the
control-plane values.

Canceling a trainer can also remove its attached rollout. A deliberately fresh
experiment must recreate/re-attach the rollout and start from actual base
weights, not simply create another SDK session on updated resident weights.

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

At approximately 04:55 UTC, the two remaining webserver tools returned through
their configured timeout without manual intervention. OpenCode's stored tool
records reported `exceeding timeout`: sample 5 had start/end timestamps
`1789441194865` / `1789448093927`, and sample 6 had
`1789441207952` / `1789448106944` (milliseconds since epoch, approximately
6,899 seconds each). Both original OpenCode processes then executed additional
tools. This verifies recovery from these two waits, not successful task
verification or a general fix for background-process handling. Check final
`result.json` and reward separately; do not equate tool recovery with a completed
evaluation or optimizer-step publication.

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

### A live traced child can deadlock a tool independently of pipe completion

In the fresh `k3-gspo-tbench78-c16x8-100step-20260915-0640` run, training
trial `harbor-opencode-vulnerable-secret-0-2-3-3b9ecc4a-1dfbecf1` stopped
making progress despite using the patched OpenCode binary. Read-only inspection
of its SQLite `part` records showed a running bash tool since 07:12:14 UTC,
with no explicit timeout. The model-authored Python command called
`subprocess.run(['/app/vulnerable'], input=payload, capture_output=True)`.

At approximately 07:21 UTC, the Python parent was in `do_poll` and its child
was in `ptrace_stop`, with `TracerPid` equal to the Python parent's PID. Both
had accumulated zero CPU time over more than eight minutes. The parent was
waiting for output/exit rather than driving the traced child. This is a live
process deadlock, not an exited shell retaining a pipe or a lost signal-exit
notification; the native completion patch does not make that command finish.
With no model-specified timeout, the existing 6,900-second default still applies.

Record the tool input, start time, PID/PPID/process-group identity, `TracerPid`,
CPU time, and wait channels before proposing recovery. Do not restart RL,
rewrite candidate code, fabricate a reward, or silently shorten the global
timeout. With operator approval, terminate only the verified deadlocked tool
process group so OpenCode receives the real command failure and can continue;
preserve the sample and audit the eventual verifier result separately. Such an
intervention must be identified in run results, not described as an untouched
sample. No automatic ptrace-stop recovery policy has been validated here.

The CPU-only `shell_completion_probe.py --cases ptrace_timeout` regression
probe uses a mock model with the real patched OpenCode CLI. Its Python parent
confirms its child entered a traced stop, then blocks awaiting the child's
captured output. With an explicit 500ms tool timeout, the CLI returned the real
timeout error and requested the next model response in 3.81s (including its
termination grace period). The test passed on September 15. This validates the
timeout escape path in isolation, not successful completion of the live task.
The live command's unchanged default deadline is 09:07:14 UTC; do not replace
that deadline with the probe's shorter timeout or infer that recovery occurred.

Live outcome: the saved `agent/opencode.txt` subsequently recorded
`exceeding timeout 6900000 ms`. OpenCode continued and the trial finalized at
09:07:36 UTC with verifier reward 1.0 and `exception_info: null`; verification
took 4.64s. The ephemeral sandbox was then deleted normally. No operator signal,
sample retry, timeout change, or RL/service restart was applied. All 128 training
samples completed. This confirms timeout recovery for this instance, but the
115-minute default still missed the desired 30-minute sampling target.

The read-only progress recorder emits `suspected_traced_child_stall` when two
successive observations contain the same active tool and the same traced child
and parent (including process start identities), with unchanged CPU counters.
This is an inspection warning, not a failure verdict or recovery policy. A new
sandbox, tool, process identity, CPU progress, resumed child, or missing
observation suppresses the warning. It never terminates a process.

### Agent completion does not bound verifier duration

The same run's other pending eigenvalue trial,
`harbor-opencode-largest-eigenval-0-11-5-32f03a19-0072f9e9`, captured its agent
trajectory at approximately 05:15:27 UTC, then remained in verification.
Read-only inspection at approximately 05:17 UTC found a Python process using
one CPU core and verifier output showing 27 collected tests with no completed
test reported. Its candidate `/app/eigen.py` invokes LAPACK through `ctypes`;
the exact native-code stall cause has not been established.

The dataset's `largest-eigenval/tests/test_outputs.py` calls the candidate
directly in `test_eigen_pair` and `test_dominance_eigenvalue`, without a
per-call timeout. The later speed tests' `future.result(timeout=30)` does not
bound those earlier calls. Harbor's verifier has its own 7,200-second budget,
separate from the agent budget. Thus an agent reaching its two-hour limit does
not imply the trial, evaluation join, or next weight sync will finish promptly.
Track the active phase and its result separately. Do not silently shorten the
verifier budget, edit candidate code, or turn a pending test into a scored zero.

Bounded diagnostic subprocesses in that same E2B sandbox did not establish a
safe recovery: direct candidate calls sometimes returned normally, while one
exited with SIGSEGV after returning. The isolated first pytest case timed out
with plugin autoload disabled and with plain assertions as well. Its
faulthandler traceback reached the first result assertion; the live process's
Python stack remained at the candidate-call line with result locals populated.
These observations do not prove a LAPACK stall or a pytest-plugin bug. Preserve
the original verifier and candidate; do not substitute a diagnostic result for
the official trial reward.

A subsequent native GDB backtrace of the live verifier placed the active thread
inside `libpython3.13.so.1.0`, called by `_PyDict_LoadGlobal` and
`_PyEval_EvalFrameDefault`, rather than inside LAPACK or a network wait. GDB and
its dependencies were extracted into a diagnostic-only `/tmp` prefix; no system
Python packages were replaced. The debugger detached and the original process
continued. The result narrows the observed stall to interpreter execution; it
does not yet prove which candidate operation or dependency corrupted state.

A bounded `runpy.run_path` probe reproduced the first test's hang without the
pytest runner: `test_eigen_pair(2)` exceeded six seconds with the candidate.
A fresh diagnostic subprocess replacing only that test function's in-memory
candidate binding with the task's `ref_solution` passed its assertions and
exited zero. No candidate/test file or official reward was changed. This
isolates the candidate execution as necessary for that reproduction; a passing
reference control is not a passing evaluation result or an authorized recovery.

### Sampling wall time versus model-request time

For the first 128-trajectory training batch on 2026-09-15, the sampling window
was 02:57:03.043–04:02:22.562 UTC. All 129 attempts, including the one failed
attempt and its replacement, had compact TITO artifacts. The union of 1,737
policy-call intervals measured model-request occupancy, not GPU kernel time.
The following attribution is non-overlapping, unlike sums of concurrent
trajectory durations:

| Wall-clock category | Seconds |
| --- | ---: |
| At least one training policy request in flight | 1,617.411 |
| Only the stalled `hf-model-inference` attempt remained | 1,833.294 |
| Other agent execution/wait with no policy request in flight | 439.569 |
| Verifier-only time after the categories above | 7.765 |
| Agent setup after the categories above | 18.353 |
| Environment setup after the categories above | 2.364 |
| Remaining handoffs | 0.763 |
| Total sampling window | 3,919.519 |

Apply attribution in table order, subtracting intervals already attributed.
The other 127 trajectories finished at 03:30:37 UTC; the stalled attempt ended
with an E2B command-stream `ConnectError` at 04:01:10 UTC. Its preserved process
audit showed an orphan shell waiting beside a background server, with the tool
still marked running. Only the missing trajectory was retried, successfully,
by 04:02:22 UTC. Its approximately 72 seconds are already included above.
Do not describe this stall-inflated batch duration as steady-state performance,
or label all non-model agent time as useful tool computation.

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

### Verifier timeout versus producer retry (2026-09-15, step 4)

**Follow-up:** a non-blocking `py-spy==0.4.1` snapshot of replacement index3
located the surviving worker in `test_outputs.py:40`,
`dist.destroy_process_group()`, after the numerical assertions and cleanup
barrier for `test_column_parallel_linear[4-False]`. The parent was waiting in
`torch.multiprocessing.spawn.join`. With Python3.13.9 / torch2.7.0, a separate
CPU-only E2B sandbox using the same template and exact candidate plus unmodified
test files passed all13 tests in22.76seconds. This narrows the live failure to
teardown; it does not yet establish the underlying runtime cause or validate a
fix. The diagnostic sandbox was deleted, and its result was NOT substituted for
the live reward. Two other replacement samples finalized with actual scored
reward0 and no harness exception.

Three further isolated runs passed all13 tests in21.12s,19.65s,19.09s. Thus the
teardown hang is observed in the live worker but is not reproduced by these
isolated runs; do not call a workaround validated on the basis of clean runs.

Another step-4 delay was agent-generated: `compile-compcert` issued `sleep900`
and then `sleep300`, followed by `pgrep -f "make -j4"`. A read-only process
inspection found that this pattern matched the querying bash itself, with no
compiler running and the compiler binary already present. Distinguish this
false-positive polling loop from a harness completion bug; do not rewrite the
candidate's commands or task result as an infrastructure recovery.

Three `torch-tensor-parallelism` samples reached their existing 1,200-second
verifier deadline at 11:48:55, 11:49:32, and 11:50:07 UTC. Their result files
recorded `VerifierTimeoutError` and no reward. The materializer returned `None`:
these attempts were preserved but not trained or assigned synthetic zero reward.
After the last sibling resolved, the producer retried only missing indices
3, 5, and 6, retaining the five scored siblings. All three replacement E2B
sandboxes were independently observed running at 11:50 UTC.

Here `producer/trajectory_drops_total` counts the three unscored attempts and
`producer/incomplete_group_retries_total` counts the one group-refill attempt;
neither establishes that the eventual training batch is missing trajectories.
Verify the final admitted group separately. A successful timeout bounds a stuck
verifier, not the entire sampling batch: fresh attempts can still exceed the
30-minute target. Quiet pytest output and sleeping workers did not establish
the exact blocked collective or an OOM; do not label that root cause proven.

### Harbor retries can precede producer accounting

Inspect `client.log` and failed trial artifacts even when all producer retry/drop
counters are zero. `run_with_fresh_trajectory_retries` can recover an individual
Harbor attempt before returning a result to the producer.

On 2026-09-15, `fix-git` cursor 55 / sample 5 exited with
`assistant output cannot be represented losslessly on this protocol`. Its TITO
artifact recorded `model_malformed` / `tito_model_malformed_output`, one model
call, and **zero trainable segments**. The parser rejected the sampled output
and its text fallback could not represent it safely. The compact artifact did
not retain the rejected completion, so it does not establish which tokens or
parser condition caused rejection; do not call this a proven renderer bug.

Although Harbor's verifier returned reward 0, the materializer rejected the
empty trajectory rather than admitting that zero as training data. The client
logged `Harbor task fix-git failed transiently (attempt 1/4)` and retried only
that sample after 15 seconds. The replacement finished with reward 1 and no
exception. Producer drop/incomplete-group counters remained zero throughout.
Check the replacement's artifact and logical sample identity before declaring
recovery; do not count both physical attempts as separate training trajectories.

### Trajectories versus gradient-normalization counts

Do not interpret the trainer's `norm_factor` as a Harbor trajectory count. In
the first batch of this run, all 128 trajectories were available, but 14 of 16
prompt groups had uniform rewards and therefore zero group-relative advantage.
Re-materializing the artifacts with the unchanged TITO converter produced 80
and 33 trainable segments for the two mixed-reward groups, matching the logged
`num_sequences` factor of 113. The image's custom-gradient counting path counts
only segments with nonzero incoming gradients; unlike its built-in GSPO path,
it does not opt into counting zero-gradient sequences. This is an existing
algorithm/normalization distinction to disclose, not evidence of lost samples.
Do not change that normalization mid-run or describe the custom path as proven
equivalent to the built-in GSPO outer mean.

The same optimizer log's `pre_norm=196.97578414` and `post_norm=1.74314853`
refer to **before/after accumulation normalization**, not before/after clipping.
Clipping uses the normalized norm, giving coefficient 1 at threshold 100 for
this step. Check the image's actual logging and clipping code before comparing
these values with W&B's `grad_norm_post_clip` metric.

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

## Read-only live progress recorder

Verifier stdout metadata (size, modification time, inode, and age) is recorded
without uploading its contents in the progress stream. Two observations of the
same unchanged file, at least 15 minutes old and still in verification, emit
`verifier_log_unchanged`. This is an inspection warning, not proof of deadlock:
check worker activity and the configured verifier deadline. Never automatically
kill a sample or assign zero reward just because its output is quiet.

When the run has the minute-level `health.jsonl` recorder, run
`python training/examples/rl/harbor/recipes/terminal_bench/monitor_e2b_progress.py --run-dir RUN_DIR --pid HARNESS_PID` in a persistent server session, with
`E2B_API_KEY` supplied through the environment. Add `--once` for a preflight.
Redirect stdout to a run-local JSONL artifact. Every three minutes it inspects
only pending trials older than 15 minutes, matched by their exact E2B session
metadata. It records tool activity ages, exit status and process/tracer states;
it never logs tool inputs or credentials, signals processes, changes timeouts,
or retries samples. Observation errors do not mean the sample failed. Review
these records alongside CPU progress and verifier logs before any intervention.
The recorder exits when the original harness PID disappears or is reused.

The recorder also captures assistant-message creation/completion times and
agent-log size/age, without recording message text or tool inputs. A stale
`part` timestamp with no running tool can be a long model turn: one observed
`circuit-fibsqrt` turn completed normally after 833 seconds and the agent
continued. Message elapsed time is not GPU-only decode time, and an unfinished
message alone does not prove that an upstream request is progressing. Compare
successive observations and finalized TITO request timing; OpenCode message
token counters were zero in this integration and must not be used for throughput.

Root-disk headroom still matters with remote E2B: the client and monitoring
tools can write local metadata even when trials live on `/shared`. On the shared
host, concurrent container activity nearly filled `/`; moving inactive owned
backup archives intact to `/shared` recovered headroom without restarting RL.
Inspect both filesystems, preserve active code/checkpoints, and coordinate other
users' builds instead of pruning shared Docker/containerd storage blindly.

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
