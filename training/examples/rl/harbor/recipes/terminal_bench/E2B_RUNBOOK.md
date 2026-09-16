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

Do not restart the trainer or rollout to fix a client/E2B problem. Diagnose the
specific failed component first; a local harness restart also requires a
checkpoint-safe, authorized transition and must preserve completed samples.

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

## Sidecar readiness timeout (batch19, 2026-09-16)

Thirty original batch19 attempts reached the existing 360-second agent-setup
deadline while `install_sidecar()` was sleeping in its readiness loop. All30
retained tracebacks identify this same stage, **not** `upload_file` or E2B
`files.write_files`. Do not attribute these failures to the earlier upload issue
or broaden the upload-specific retry classifier to include them.

One inspected retained sidecar log had only the informational no-PyTorch warning
and no exception stack; this does not locate the underlying startup stall.
The outer setup deadline can expire before the sidecar's own600-second readiness
deadline. No deadline was changed. Existing incomplete-group handling retries
missing members after their group resolves and preserves successful siblings;
at13:08UTC cumulative drops were68, retries33, and rejected rows remained1.

The read-only observer now records endpoint **existence only**, numeric PID,
process state/start identity and CPU time. It warns when the endpoint is absent
after four minutes, allowing inspection before the setup deadline when the
observation arrives in time. It never reads endpoint/spec contents, sends a
signal, retries a sample, or changes a timeout as a result of this warning.
An existing PID or `kill -0` success alone does not prove a healthy sidecar:
inspect zombie state as well. Missing readiness is a diagnostic, not a proven
deadlock or permission to discard a sample.

All65 monitor unit tests and Ruff passed. A read-only live E2B probe returned
an existing endpoint and sleeping sidecar PID1644, with no warning; no
candidate, scoring or RL-client behavior changed. This improves diagnosis,
not a demonstrated fix for the underlying readiness failure.

## Installation-file upload timeout (2026-09-16)

Batch 16 / evaluation 15 hit 38 `AgentSetupTimeoutError` attempts (27 training,
11 evaluation). Retained tracebacks show the 360-second installation deadline
expired inside `environment.upload_file` → E2B `files.write_files`, before
agent execution. The pinned OpenCode shell-fix binary is 149,715,072 bytes and
is uploaded per sandbox. This identifies the failed stage, not whether the
underlying bandwidth limit is on the client or provider.

The adapter now recognizes only this exact E2B installation-upload stack as
recoverable when no valid trajectory is available. Existing bounded per-sample
retries/backoff apply; successful sibling samples remain preserved. Other
setup timeouts, task timeouts, and candidate failures are not reclassified.
Tests cover the adapter's missing-artifact branch and negative classifications;
the classifier also matched all 38 retained failures and rejected unrelated
exceptions in that observation window.

Distinguish **live sandboxes** from **logical samples still needed**. In batch16,
gcode245/member0 failed setup at06:51:19UTC, while member2 was still running
after08:00. The producer retires each draw but submits incomplete-group retries
only after the assembler resolves the whole group (`producer._retire` and
`_resolve_row`). Thus one live training sandbox did not mean127/128 samples
were scored: the artifact audit found126scored plus one pending and one missing.
The adapter-level retry above handles the recognized upload failure before
returning a dropped draw, without waiting for a slow sibling. Do not change
whole-group scheduling or resample scored failures silently.

The original live client predates this fix: updating the checkout does **not**
reload its Python functions. Do not claim the fix is live or restart the
client/trainer/rollout implicitly. Avoid recurring bulk transfer in a future
approved template by preinstalling the **same checksum-pinned binary**, with
version/hash verification; retries alone do not remove upload overhead.

An incomplete evaluation's `eval/reward` averages only returned trajectories.
Always report attempted/completed/no-trajectory counts alongside it, and do not
compare it as a complete fixed 64-sample evaluation. Never turn setup failures
into synthetic zero rewards or silently omit them from the report.

## Snapshot synchronization checks

### Candidate hangs inside a verifier

In the 2026-09-16 `filter-js-from-html` sample (cursor 246/member 1), the
agent exited successfully but its generated parser did not advance its
cursor on a `>` inside a tag body. A bounded, read-only probe reproduced the
same non-advancing cursor on the live 352-byte input. The official verifier's
filtering subprocess has no per-file timeout; its Chrome timeout is separate.

Do not kill just the candidate child to unblock this verifier: this particular
verifier skips nonzero child exits, potentially producing a misleading pass.
An early intervention must stop the **whole verifier attempt as unscored**,
preserve artifacts, and retry only the missing sample, with explicit approval
when that changes the agreed deadline. Otherwise retain the existing timeout.
Never edit the candidate solution or verifier, or synthesize a reward. If it
finishes after a child OOM, audit which cases were actually tested before
claiming that the reward is valid.

For that attempt, the E2B verifier command stream eventually failed with
`ConnectError: Error reading content`; no verifier reward was returned. The
client retained the exact artifact untrained and retried only the two missing
members of its eight-sample group. No manual interruption occurred. This
transport message alone does not establish why the stream failed. A compact
artifact marked `completed` means agent completion, not successful verification.

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

Root exhaustion can also cause `gcloud` file-logging errors independently of
the transfer result. For the archive worker only, use a private `CLOUDSDK_CONFIG`
directory on a filesystem with headroom and `CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1`.
Keep its temporary files off the full root filesystem as well. Preserve the
existing authenticated configuration; when copying live SQLite credential/token
stores, use SQLite's backup API rather than copying an open database file.
Require directory mode `0700` and credential-file mode `0600`, and keep this
directory outside the archived run and repository. Never upload it. Verify
authentication and an exact GCS object before replacing only the archive worker;
do not restart RL or clear upload/checkpoint markers. A later manual login may
require refreshing this isolated configuration. Check transfer exit status and
destination objects, not merely the presence of local logging errors.

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

### A command's inner GNU timeout may not terminate Node

On September 15, batch 8 of the 78-task run exposed another instance in
`make-mips-interpreter`, cursor 114, completion 6. The model requested
`timeout 120 ...`; its Node child was still running after 240 seconds, while
GNU timeout waited in `sigsuspend`. The child had a SIGTERM handler and kept
accumulating CPU time. This was an overdue **command**, not a stalled trainer
or justification to replace the whole trajectory.

Under the user's proactive-recovery instruction, at 17:41:31 UTC we revalidated
the exact sandbox, Node PID/start time, timeout PID/start time and literal
120-second argument, and the shell-to-OpenCode ancestry. A pidfd-targeted
SIGKILL was sent **only to the overdue Node child**, after 270.6 seconds.
The same OpenCode process immediately resumed and issued another command.
The agent, sandbox, trainer and rollout were not restarted; no score or
candidate code was changed. Final task success must still come from the
verifier, not the fact that tool execution resumed.

This initial recovery was performed manually. The monitor now provides an
explicit opt-in `--recover-overdue-node` guard, disabled by default. It requires
the same sandbox, active bash tool, and Node -> timeout -> bash -> OpenCode
process identities across two observations. Immediately before signaling it
opens a pidfd and revalidates the full ancestry and the actual remote command.
Only plain `timeout DURATION node ...` is accepted, after its own deadline plus
30 seconds of grace. Unknown option forms, changed identities, verifier
processes and commands still within their deadline fail closed. Actions are
recorded separately from scores. Do not apply a generic 30-minute kill limit
to valid tasks or interpret recovery as verifier success.

The observer defaults to 180 seconds between cycle starts. For short explicit
command deadlines, `--interval-seconds 60` reduces recovery-detection delay;
it does not change any agent, verifier, tool or sampling timeout. A cycle that
takes longer than the interval finishes before another starts (no overlapping
polls). The original RL-client process identity is still checked each cycle.

Observed live outcome for cursor114/index6: after two operator-triggered child
recoveries and two periodic guard actions (17:51:17 and 17:57:20 UTC), the same
agent finished at 18:03:29 UTC with verifier reward 1.0, no Harbor exception,
and a completed TITO artifact. Total sampling time was about 42 minutes, so
this does **not** establish the under-30-minute target or prove that recovery
guarantees success on other tasks. Preserve the intervention audit alongside
the actual verifier result.

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

### Gloo teardown native-stack evidence (September 15, step 5)

Trial `harbor-opencode-torch-tensor-parallelism-0-70-1-f3cd6fc0-b8a50ce8`
stalled after five tests with Python 3.13.9 / PyTorch 2.7.0. A native stack
capture found the main worker in `ProcessGroupGloo::~ProcessGroupGloo`, waiting
in `pthread_mutex_lock` during `dist.destroy_process_group()`. A Gloo worker
thread was in `gil_scoped_acquire`, destroying tensors from
`AsyncAllgatherWork` / `AsyncBarrierWork` under `ProcessGroupGloo::runLoop`.
The parent was waiting in `torch.multiprocessing.spawn.join`; the preceding
test cleanup barrier had returned. This is not a rollout request wait.

These stacks, together with PyTorch 2.7.0's work-mutex and Python-holder
destruction paths, support a GIL/work-mutex lock-order diagnosis. Mutex
ownership was not independently inspected, so do not claim a proven fix from
these stacks alone. Native capture briefly attaches to the worker; distinguish
it from nonblocking Python-stack observation.

An isolated CPU-only experiment retained the typed Gloo backend across generic
process-group destruction, then released it through the typed no-GIL holder.
It was **not applied to live trials**. Initial isolated probes:

| Isolated probe | Stock teardown | Experimental teardown | What it establishes |
| --- | --- | --- | --- |
| 400 one-rank init/all-gather/barrier/destroy cycles | 400 passed | 400 passed | Smoke compatibility; no reproduction of the hang |
| Exact retained step-4 candidate and unmodified verifier (ranks 1, 2, 4) | 13 passed | 13 passed | Matching test outcomes; no reproduction of the hang |
| Exact step-5 candidate and unmodified verifier (ranks 1, 2, 4) | 9 passed, 4 failed | 9 passed, 4 failed | Same pass/fail pattern; neither run hung |

Those initial probes alone did not validate a remedy. Do not substitute
diagnostic results for a live reward or edit candidate code/verifier assertions.
Preserve the existing scoped verifier deadline and recover only missing group
members through the normal producer path. The current candidate's four test
failures are separate from the live teardown hang.

**Later reproduction:** repeating the literal `tests/test.sh` entrypoint in
one isolated sandbox produced stock results of (1) 9 passed / 4 failed, then
(2) a hang after five tests, terminated by the diagnostic's 150-second bound.
The two interleaved retained-backend runs completed with the same 9/4 test
outcomes and reward 0. The hung stock worker was in `std::thread::join` inside
Gloo destruction; its worker thread was waiting for the GIL during tensor
destruction. This reproduces GIL-held teardown waiting for a worker that needs
the GIL, complementing the live mutex-wait stack. It is not a candidate
collective mismatch or a wait for rollout tokens.

The first literal fixture copy had an extra trailing newline, with unchanged
executable code; the recorded hashes distinguish it from the source bytes.
A byte-exact follow-up then completed with the same 9-pass/4-fail result and
reward 0 under stock (25.23s) and retained-backend (28.88s) teardown. A separate
positive-candidate activation audit passed all 13 tests in 19.95s and recorded
exactly one retained/released pair in each of 28 verifier workers (world sizes
1, 2, and 4). This verifies that the hook actually executed, rather than merely
observing another lucky completion. The hook is scoped to PyTorch 2.7.0 and the
default Gloo backend; no collective math or assertions are changed.

The observer now alerts after five minutes of unchanged verifier output across
two matching sandbox observations (previously fifteen minutes). This is an
inspection trigger, not proof of deadlock or permission to terminate a verifier.
The change was prompted by batch 18 stopping after six tests while its scoped
verifier deadline was only twenty minutes; waiting fifteen minutes left little
time to investigate. The alert does not change task timeouts or rewards.

The workaround remains isolated pending approval for future task verifiers;
these tests do not establish correctness for every backend or PyTorch version.
Do not
reuse a diagnostic score for training, and do not interpret shell exit 0 as
test success: this verifier writes reward 0 after pytest failures and then
exits successfully. Read the actual pytest result and reward file.

The third live attempt (`f3cd6fc0-b4f9561f`) reproduced the same teardown
deadlock after six tests. Its worker PID 2503 waited in `std::thread::join`
inside Gloo destruction; thread 2515 waited for the GIL while destroying
all-gather/barrier work tensors. All three attempts therefore need runtime
triage, not a larger model-generation timeout. The cleanup workaround remains
unapplied pending approval; diagnostic outcomes are not live rewards.

Batch 9 reproduced the same issue on September 15 around 20:10 UTC in
`torch-tensor-parallelism`, cursor143/index1. Seven siblings finished; this
verifier stopped after six tests. A native stack capture showed worker2508
joining a Gloo thread while thread2520 waited for the GIL during all-gather/
barrier tensor destruction. The debugger detached successfully. Increasing
the agent timeout cannot resolve this verifier teardown deadlock. Any scoped
verifier retry must preserve candidate/test bytes and integrate with Harbor's
original exec/result lifecycle: killing pytest alone can finalize an incorrect
failure reward. Do not inject debugger calls to release the GIL, reuse a
diagnostic score, or restart the trainer/rollout to treat this condition.

**When retries are exhausted:** the coordinator rejects the incomplete prompt
group and admits the next source row if available. It does not train the seven
valid siblings as a complete eight-trajectory group. The unit test
`test_exhausted_missing_member_refills_full_batch_without_redrawing_siblings`
uses 16 groups × 8 rollouts, two missing-member retries, and a replacement row;
it verifies a full batch, three dropped attempts, one rejected row, and no
regeneration of successful siblings. This is a coordinator test, not evidence
that the live verifier recovered. Monitor `producer/rows_rejected_total` and
actual batch size. With finite `max_rows`, rejected groups reduce the number
of accepted groups and can produce a smaller final batch or fewer optimizer
steps; the initial step estimate is not a completion guarantee. Do not silently
change the dataset budget to compensate.

### Recursive grep of kernel memory (September 15, batch 9)

A password-recovery agent's bash pipeline ran `grep` over `/proc/kcore` for
more than ten minutes, reading over 1.3TB with zero output while downstream
grep/head waited. CPU activity here was not evidence of useful task progress.
After two matching process/tool observations and fresh in-sandbox validation,
SIGTERM was sent through a pidfd to only that grep child. The same OpenCode
PID/start identity continued and completed subsequent tool work; the sample
was not restarted or rescored.

`monitor_e2b_progress.py --recover-kcore-grep` enables this narrow guard.
It requires the unchanged grep→bash→OpenCode chain, a running bash tool older
than ten minutes, and an exact live `/proc/kcore` descriptor. Ordinary files,
other devices/processes, changed identities and finalized trials fail closed.
It is disabled by default and is not a general slow-task termination policy.
The existing blocked-kernel-stream guard remains separate and unchanged.

### Sandbox memory pressure versus an observation timeout

In step 5, `mteb-leaderboard` cursor 77 / sample 2 had kernel-confirmed OOM
kills of candidate Python processes (PIDs 1909 and later 2030). The TITO
sidecar and OpenCode survived, and new tool calls were observed afterward.
The task declared 4 GiB; its actual sandbox already had approximately 8 GiB.
Do not silently increase task resources or classify this as a trainer OOM.

Sample 0's control-plane identity still existed, but guest reads timed out
and its last metrics were stale with about 96% memory usage. That is evidence
of an unresponsive guest, **not confirmation of its cause or terminal state**.
Recheck the same sandbox and command handle; never restart based solely on an
observation timeout. Record metric timestamps, kernel OOM evidence when
available, which process died, whether the agent continues, and the final
scored/unscored outcome separately.

The read-only observer emits `repeated_sandbox_observation_error` after two
consecutive failed observations of the same identified sandbox. Recovery or a
replacement sandbox clears that warning. This flags a monitoring blind spot;
it does not mark the sample failed, retry it, or authorize a restart.

The observer also records guest `MemTotal`/`MemAvailable` and each observed
process's `VmRSS_bytes`/`VmPeak_bytes`. `sandbox_memory_pressure` warns when
available guest memory falls to 10% or less. Inspect process growth and kernel
OOM evidence; the warning is not proof of OOM and never kills a process,
changes task resources, or retries a sample. Do not sum process RSS to estimate
guest usage: shared pages overlap. This is sandbox RAM, not trainer GPU memory.

### SSH askpass can loop on a host-key confirmation

In batch 14, `git-multibranch` cursor 219 / sample 0 created an askpass
helper that always returned `password`, then ran `git clone` over local SSH
with `SSH_ASKPASS_REQUIRE=force` and no explicit tool timeout. After more
than ten minutes, direct `/proc` inspection caught that helper being called
with `Please type 'yes', 'no' or the fingerprint:`. The response did not answer
the question, causing repeated helper launches. Git and `head -5` waited for
output; this was not a rollout, verifier, or E2B-capacity stall.

Inspect the exact SSH child's identity, parent chain, and helper prompt before
diagnosing this loop. Do not fix the candidate solution, accept its host key,
change SSH security settings, or terminate the agent. A narrowly approved
recovery may stop only the identified SSH child and let the original tool
failure reach the agent. At diagnosis time no such signal had been sent;
approval and actual subsequent tool/sample outcomes must be recorded separately.

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

### MIPS cleanup command can kill its own agent launcher (2026-09-15)

Step-4 cursor 50/index 1 and index 2 both ended with
`NonZeroAgentExitCodeError` (E2B exit `-1`). The final tool calls recovered from
their checksum-validated TITO artifacts included `pkill -f "node vm.js"`.
The OpenCode launcher itself contains `node vm.js` in the task prompt passed
on its command line, so this broad match can terminate the launcher too.

A separate CPU-only E2B reproduction confirmed that a harmless launcher with
that text in its argv was matched and terminated, returning the same `-1`.
The diagnostic sandbox was deleted; no live trial or candidate was modified.
Retained evidence is under the September 15 run's GCS `run/` prefix:
`reproduce_mips_pkill_exit.py` and `mips-pkill-reproduction.log`.

These two failed-agent artifacts still have real verifier scores (1 and 0).
The existing materializer accepts their exact retained segments: 39,113 and
73,334 trainable tokens respectively. A scored result therefore does **not**
prove clean agent completion. Do not relabel these as infrastructure-free
successes, silently regenerate them, or replace their scores with diagnostics.
Neither prompt delivery nor failure admission was changed in this live run.

The PR now contains a separately verified transport fix for pinned OpenCode
1.18.8: upload the instruction outside `/logs` and feed it on stdin, keeping task
text out of launcher argv. Preserve the CLI's existing argv normalization first:
an argument containing an ASCII space is wrapped in double quotes, with embedded
double quotes escaped. **Sending raw stdin would change the model prompt.**
Other OpenCode versions retain the original argv path until independently
verified. No tokenizer, renderer, sampling, permission, or timeout setting changes.

`opencode/prompt_transport_probe.py` exercises the actual patched CLI against a
loopback provider in an isolated CPU-only E2B sandbox. It verifies equality of
all initial messages and tool definitions, detects the raw-stdin mismatch, and
checks that the agent continues after the same broad `pkill` tool command.
The tool shell may terminate itself; success is agent continuation, not forcing
the tool to succeed. The verified run used sandbox `iwntjjnhq8yztae6pl86h`, deleted
afterward; evidence is `run/prompt-transport-full-input-verified.log` in the same
GCS prefix. The 189-test targeted suite also covers pinned-version normalization,
other-version fallback, context-overflow exit handling, metrics, and monitoring.

This code is **not active in the existing September 15 client**, which was not
restarted or reloaded. Activate only at an approved checkpoint-safe client
transition. It does not fix a candidate's blocked SIGTERM handler or authorize
shortening any live task timeout, replacing its reward, or killing its process.

The same launcher problem recurred in batch 8 at cursor114/index4 and
cursor119/index2 and index3. All three checksum-validated TITO artifacts ended
with tool calls containing `pkill -f "node vm.js"`; all recorded model calls
succeeded. They received real verifier rewards 0, 0, and 1, respectively.
Cursor119/index2 was misleadingly labeled `ApiRateLimitError`: Harbor's
`_classify_exec_error` searched all stdout using `rate.?limit` and matched the
agent's discussion of GitHub repository-search limits. This is not evidence
of an HTTP429 from rollout. Check structured call outcomes and the final
emitted tool action before changing capacity, credentials or retry settings.
These samples were not regenerated, rescored, or silently removed from training.

The same signature subsequently affected cursor119/index1 and index0, bringing
the batch-8 audit to five scored agent exceptions. Index0 finished at
2026-09-15 18:39:29 UTC with verifier reward `1.0` and a failed artifact; all 131
recorded model calls succeeded, and the final emitted tool action again
contained `pkill -f "node vm.js"`. A passing verifier score does not establish
clean agent completion. This is further evidence for deploying the already
verified prompt-transport fix at a safe approved transition, not a reason to
retry successful model calls or silently replace the recorded reward.

The read-only observer now audits finalized local results for
`scored_trial_with_exception`, independently of pending trials and producer
drop counters. Its scope excludes artifacts already pruned after checkpoint
and archive verification; it is not a lifetime failure count. Exception messages,
launch configuration, and tool text are never emitted by this audit.

### Unclipped gradient metrics

The metrics adapter previously omitted `train/grad_clip_coefficient` when the
trainer's pre-clip and post-clip norms were equal. It now reports `1.0` when both
reported norms are equal and positive, while still omitting the duplicate
post-clip norm. Missing, zero-denominator, or non-finite diagnostics do not imply
a coefficient. This is logging only, not a change to gradient clipping.

The September 15 live client was not restarted to pick up this change. Do not
interpret absent historical coefficient points as evidence of clipping, and do
not present coefficients inferred from a configured threshold as logged values.

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

The default recorder is read-only. After approval to recover confirmed tool
hangs, add `--recover-kernel-stream-grep` to the **monitor command**, not the RL
training command. This opt-in guard requires the same active grep call and child
PID in two observations without CPU progress, and a grep age of at least five
minutes. Inside the exact sandbox it rechecks PID/start time, OpenCode parent
identity, root working directory, unchanged CPU time, open kernel-stream FDs,
and blocked kernel wait channels. Only then does it send SIGTERM through a
pidfd to that search child. Unsupported pidfds, changed/missing evidence, or a
finished trial cause no action. It never kills the agent, sandbox, process group,
trainer, or rollout; never changes a score, retries a sample, or caps a valid
model/CPU computation. Every attempted recovery is recorded in `recovery_actions`.
Observation errors mean the action outcome is unknown and require reinspection.

The first real recovery preserved the same OpenCode PID and trajectory. The
agent continued immediately with new tool calls, then completed with reward0
and no Harbor exception. A recovered harness does not imply that the model
solved the task; keep the actual verifier result.

`long_grep_wait` flags an OpenCode grep call running for at least five minutes.
Inspect before the 30-minute sampling target; do not classify the sample as failed
based on elapsed time. A confirmed `headless-terminal` incident searched `.` from
`/`: ripgrep workers blocked in `kmsg_read` and `tracing_read_pipe`, with open
descriptors to `/proc/kmsg` and `/sys/kernel/.../trace_pipe*`. The parent waited
on a futex. OpenCode's result-count limit did not provide an elapsed-time bound.
This was a tool/pseudo-filesystem read, not a slow model call or capacity issue.
Record the exact sandbox/process identity and request approval before terminating
only the stuck search. Preserve the agent/sample and let the tool failure be
handled normally; never synthesize a result or silently restart the RL run.

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
only pending phases older than three minutes, matched by their exact E2B session
metadata. It records tool activity ages, exit status and process/tracer states;
it never logs tool inputs or credentials or changes timeouts; without the explicit
recovery option it never signals processes. It never retries samples.
Observation errors do not mean the sample failed. Review
these records alongside CPU progress and verifier logs before any intervention.
The recorder exits when the original harness PID disappears or is reused.

For plain `timeout DURATION COMMAND` processes it records declared duration and
process age, not command text. `inner_timeout_overrun` means the wrapper is still
alive more than30seconds beyond that duration; it is an inspection warning, not
a new timeout or kill policy. Unknown option forms are skipped. One live MIPS
sample ran `timeout60 node vm.js` for over15minutes: the generated JS registered
a SIGTERM callback but blocked its event loop in synchronous execution, preventing
that callback from exiting. GNU timeout's initial SIGTERM alone did not stop it.
Do not silently rewrite the candidate or assign a reward; preserve the exact
process identity and obtain approval for a narrowly scoped intervention.

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
For a reversible worktree relocation, first check active users and Git status,
retain the original path, and verify both the root repository and submodules.
Cross-filesystem moves can partially fail on root-owned generated files; preserve
and checksum-copy remaining files before completing the relocation. Relative
submodule Git pointers may need repair. Do not treat a partially moved tree as
complete or remove its remaining files without verifying their preservation.

## Do not classify model/tool text as a provider failure

Harbor's installed-agent classifier searches the failed command's entire output
with broad error regexes such as `rate.?limit`. OpenCode JSON output includes
reasoning and tool results, so an agent mentioning a GitHub download rate limit
can incorrectly turn an unrelated launcher exit into `ApiRateLimitError`.

The OpenCode adapter now removes recognized JSON **content** events only from
the classification input for its own JSON-mode launcher. It preserves session
`error` events, raw startup diagnostics, unknown formats and setup-command
behavior. Original execution output and trajectory artifacts remain unchanged.
Tests cover these distinctions and real provider-error preservation. Replaying
the saved MIPS cursor119/index2 output (478,974 bytes, launcher exit-1) changed
the classification from `ApiRateLimitError` to `NonZeroAgentExitCodeError`.
This fixes diagnosis, not the underlying pkill/launcher bug; it is not evidence
of a throughput improvement. It does not rewrite historical sample results or
change an already-running client. Verify TITO call outcomes before attributing
a failure to provider capacity, credentials, or HTTP429.

## Distinguish observation failures from normal sandbox teardown

An E2B inspection can time out while Harbor finishes the trial and deletes its
ephemeral sandbox. Recheck the local finalized result after a list/connect/read
failure before treating the sandbox as missing. The progress observer reports
`already_finalized` when that race occurs; it does not infer a successful agent
exit or a passing reward. The separate finalized-result audit still reports
recorded exceptions. Without a result, retain the observation error or zero
sandbox count and investigate; never restart the sample solely on this signal.

Observed example: batch11 `db-wal-recovery` cursor172/member6 finalized at
2026-09-15 22:57:31 UTC with reward1 and no exception during an observer timeout.
The subsequent sandbox-not-found response was normal teardown, not lost work.
Mocked tests cover list, connect and command failures, both with and without a
newly finalized result, plus sandbox-list disappearance. This is an observer
diagnostic fix, not a change to sample deadlines, rewards or the RL algorithm.

## APT setup can stall before verifier tests start

Batch12 `sqlite-db-truncate`, cursor180/member6, stopped in the official
verifier's `apt-get update`: stdout stayed at425 bytes for nearly11 minutes,
with the parent and HTTP/acquisition workers waiting in `select`. A fresh GET
of the same Debian InRelease file returned151075 bytes in36ms. The sandbox
was not frozen, CPU-throttled or memory-constrained. This was not a model
request or an executing pytest test.

A single idle TCP connection was reset after PID/start-time/socket checks;
**that did not unblock APT**. Idle socket counters alone do not establish the
root cause. Workers were waiting on parent IPC, not actively reading that
socket; do not turn TCP reset into an automatic recovery policy.

After verifying `curl` was already installed, only the exact stalled
`apt-get update` process was sent SIGTERM through a revalidated pidfd. The
unchanged official `test.sh` does not use `set -e` and continued normally:
`apt-get install` reported **zero package changes**, pinned uv/pytest setup
completed, and the real test passed (CTRF:1 passed,0 failed;0.01s).
The trial finalized2026-09-16 00:07:44UTC with reward1 and no exception.
Agent output, tests, verifier shell, trainer and rollout were not restarted
or rewritten. No reward was assigned by the recovery code.

This is a case-specific setup recovery, **not permission to kill arbitrary
APT operations**. Never interrupt package unpack/configure operations based
only on elapsed time. Inspect the exact subcommand, parent/child progress,
locks, repository access, installed dependencies and verifier error handling;
preserve the real test result and intervention audit.

The observer now warns after two matching observations of a quiet verifier
log (five minutes) plus the same APT PID/start-time/parent and unchanged CPU
counter. It does not classify a failed sample or send signals. Tests cover
progress, process reuse, missing fields, differing phases and sandbox changes.
Updating this module does not hot-reload an already-running observer/client.

## Process-name polling can match the waiting shell itself

Batch12 `compile-compcert`, cursor185/member5, ran
`while pgrep -f "make -j4 all" >/dev/null; do sleep 10; done`.
The shell command line itself contained the search string. A `/proc` audit
found only that waiting shell matching it, with no matching build process.
Consequently the loop could not report completion even after the build ended.
This was a model-authored shell command, not an E2B capacity problem.

A local CPU-only reproduction confirmed that the original loop exceeded a
five-second test deadline with no build running, while the same test with
`pgrep --ignore-ancestors -f` exited normally. Prefer waiting for the actual
captured build PID and collecting its exit status; a process-name search
does not establish build success. Ancestor exclusion is supported by the
tested procps version, not necessarily every sandbox implementation.

The live command reached its existing ten-minute tool timeout and the agent
continued. The real verifier subsequently passed; the trial finalized at
2026-09-16 00:32:40 UTC with reward1 and no exception. A proposed targeted
recovery failed an argv identity check **before sending any signal**; no live
recovery or command rewrite was applied. Do not credit this completion to an
intervention, and do not signal the PID after sandbox finalization.

Another sample in the same batch repeatedly used fixed 570–590-second sleeps
before checking a background build. These sleeps can outlive the build and
delay sampling without any hung process. They are distinct from the self-match
bug: do not terminate valid commands merely to meet a thirty-minute target.
Before changing the agent prompt, task code, or polling policy, record the
change and its effect on comparability. These observations do not justify
discarding samples, fabricating build success, or altering verifier tests.

## CPU usage is not sufficient evidence of useful agent progress

Batch13 `feal-linear-cryptanalysis`, cursor204/member0, ran a generated C
search with a fixed4194304-entry hash table. The insertion loop had no
full-table termination condition. A read-only live audit found every slot
occupied; the whole96MiB table checksum and output remained unchanged over
75seconds while the process consumed another75CPU-seconds. This strongly
supports a capacity-probing spin, not useful enumeration. The current key and
instruction pointer were not sampled, so keep that inference distinct from
the directly measured saturation and unchanged state.

Do not call a long-running tool healthy solely because its CPU counter rises.
Check boundedness, output/state progress and the existing deadline. Conversely,
an unchanged output file alone does not prove a hang: valid programs buffer
output or compute before writing. This diagnosis involved task-specific source
and memory inspection; it is **not an automatic arbitrary-memory-reading or
process-killing rule** for the monitor.

The fault is in generated candidate code, not a demonstrated E2B defect.
Do not repair the candidate algorithm, fabricate a tool response, or retry a
valid zero-reward sample to improve its score. For this case, a request to
interrupt only the search and let the existing agent receive the actual command
failure was submitted for approval; no response arrived and no command was
manually signaled or rewritten. The existing two-hour agent deadline fired
at2026-09-16 02:50:53UTC. The real verifier then failed `test_attack` and
returned reward0; finalization completed at02:50:59UTC. The compact artifact
validated with4segments/11turns and `abandoned/agent_cancelled` status, and
the ephemeral sandbox was removed normally. The128-sample batch finished
with one scored AgentTimeoutError, no additional drops/retries, and sampling
wall time123m10s. All other127 samples took less than30minutes.

This was natural deadline handling, not a successful manual recovery. The
old command-interruption request is obsolete; never signal its former PID.
Future deadline or agent-policy changes must be explicit and preserve the
real reward and exception record. Do not report this batch as meeting the
thirty-minute target or retry the valid zero merely to improve its score.

Batch 18 reproduced the same source-level risk in cursor 288/member 1:
the generated `trail` program had all 8,388,608 hash slots occupied, and its
linear-probing insertion loop had no full-table escape. This was measured
read-only from the matching live executable's `hused` array; it was not
inferred from CPU utilization alone. The instruction pointer was not captured,
so distinguish the demonstrated saturation/code defect from the inferred
current spin. At the audit, the original process remained live and a scoped
interruption request was unanswered; no manual recovery was claimed.

The trial config and two allowlisted fields from the live OpenCode process
also confirmed a 6,900-second default bash-tool timeout inside a 7,200-second
agent limit. A command started more than five minutes into the agent budget
can reach the outer deadline before its own timeout, leaving no recovery turn.
Frequent observation alone cannot make this meet a thirty-minute sampling
target. A future shorter command budget or remaining-budget-aware recovery
policy requires an explicit, recorded change; do not silently shorten this
run's deadlines or repair candidate code. When inspecting configuration or
process environments, select named non-secret fields only: agent kwargs also
contain credential-bearing sidecar launch metadata and must not be dumped.

## Preserve scored exceptions when reusing retained trials

A real verifier reward does not imply clean agent execution. Harbor can
produce a valid score after `AgentTimeoutError` or `NonZeroAgentExitCodeError`.
The fresh-trial path preserves that exception, but retained-result reuse
previously replaced it with `None`, hiding the original failure in rollout
metadata. Reuse now copies the stored exception type/message and reports
`harbor_exception_type` consistently with a fresh result.

This is a diagnostic correction, not a reward or retry-policy change. Tests
cover clean reuse, a scored timeout with reward0, and a scored nonzero exit
with reward1; none launches a fresh trial or alters the verifier reward.
The currently running Sep15 client predates this fix and has not been
restarted or hot-reloaded. Inspect its result.json exception_info directly
when auditing scored failures.

## Evaluation coverage and stale dashboard summaries

An evaluation can lose setup attempts while later samples are still running.
In the Sep16 evaluation15 audit, 52/64 logical samples had scores, eleven had
AgentSetupTimeoutError during installation upload, and one was still running.
The dashboard continued showing evaluation10's completed64/reward0.578125:
those summary gauges did not establish evaluation15 progress or success.

Evaluation now emits `eval/step`, `eval/coverage` (returned trajectories divided
by attempted trajectories), and `eval/is_complete`. Empty evaluations are not
complete. These are final evaluation metrics, not live progress counters.
Interpret reward alongside coverage and the evaluation step; reward remains
the mean over returned scored trajectories. Missing samples are not silently
converted to zero, retried for reward improvement, or treated as successes.
Partial evaluations are not directly comparable with the complete fixed pool.

Tests cover complete, partial/None, raised-exception and empty evaluations,
including a custom metric prefix. This is diagnostic-only: no changes to
sampling, retries, tokenizer, weight versions or reward calculation. The
original Sep15 RL client predates these metrics and has not been restarted
or hot-reloaded; inspect its artifacts and exact log epoch for live coverage.

## OpenCode tool timestamps reset by progress metadata

In the Sep16 regex-chess audit, live snapshots observed a model-authored
`timeout 900 python3 fuzz.py ... | tail ...` running for about 900 seconds.
After it returned, the completed SQLite tool state reported only 9 ms.
The local OpenCode metadata callback in `src/session/tools.ts` writes
`time.start = Date.now()` on progress updates; completion retains the latest
value. Do not infer actual tool duration from completed start/end alone.
Likewise, the pipeline's exit 0 can be tail's status, not a passed fuzz test.

The E2B observer preserves the earliest recorded start across consecutive
observations, keyed by trial, sandbox and unique tool-part ID. It logs this
as `observed_elapsed_s_lower_bound` alongside unmodified raw `start_ms` and
`elapsed_s`, and uses the lower bound for inspection-only long-tool warnings.
Calls with the same tool name never share timing state. Missing observations
or observer restarts can still underestimate elapsed time; this is not an
exact profiler. A late first observation cannot reconstruct earlier resets.

This correction changes no agent binary, rollout settings, deadlines,
rewards, retries, or process-based recovery guards. A long-tool warning is
not authorization to terminate a candidate's legitimate computation.
Tests cover repeated timestamp resets, identity boundaries, malformed times,
legacy observations and missing history. Activate an updated observer
separately; the RL client/trainer/rollout do not need a restart.

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
