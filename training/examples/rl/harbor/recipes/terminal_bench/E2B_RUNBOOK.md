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
| Timeouts | Agent and verifier each allow 7200s; tool timeout remains below the agent timeout (`6900 < 7200`) |
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
| `Verifier execution timed out after 900.0 seconds` | Extending only `agent.override_timeout_sec` leaves Harbor's independent verifier timeout at its 900s default | Set both agent and verifier `override_timeout_sec` to 7200 in the dedicated trial config | A verifier running longer than 900s returns a numeric reward instead of a rewardless trajectory |
| `build is not in waiting state` | Eight rollouts raced to build the same previously absent task alias | Prebuild each unique alias once; only then start rollout fanout | No template builds occur during the 128-way smoke launch |
| `renderer 'kimi_k3' has no production TITO certification` | The offline cookbook renderer name was passed to the production sidecar | Use `kimi_k3_preserve_thinking` | Sidecar readiness succeeds |
| `tokenizer does not match TITO certification` | The unpinned HF default resolved to revision `f831ab...`; certification is for `9f62e4e9...` | Pass the exact `--tokenizer-revision` above | Host and reloaded bundle fingerprints both equal `3d98398c...` |
| Inner timeout validation failure | Tool timeout equaled the outer trial timeout | Use `--sample-timeout 7200 --harness-tool-timeout-seconds 6900` | CLI validation passes before provisioning |
| Too many open files | 128 concurrent environments exceed a 1024-FD shell limit | Run `ulimit -n 65536` before Python | `/proc/<pid>/limits` reports `65536` |
| `Sandbox not found` during artifact cleanup | Secondary cleanup after sandbox creation/build failed | Diagnose the earlier exception; do not treat cleanup noise as the root cause | Root exception is absent on rerun |
| `PyTorch was not found` | Informational Transformers warning in the lightweight sidecar | No fix required; TITO needs tokenizer utilities, not Torch | Ignore unless followed by a different fatal exception |

## Launch sequence

1. Run unit tests for task rewrites, timeout ordering, and the dedicated config.
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
