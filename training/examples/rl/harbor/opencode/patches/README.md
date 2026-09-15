# OpenCode 1.18.8 shell-completion fix

This is a **native OpenCode patch**, not a prompt change or shorter task timeout.
Apply only to upstream commit `3c81a5d1ddceab377d9ad71c14899e6935333fdd`
(1.18.8). The renderer, tokenizer, model requests, task/verifier timeouts, and
training configuration are unchanged.

## Failure and fix

- The process adapter resolves completion on Node's `close` event. A foreground
  shell can exit while a background HTTP server retains stdout/stderr, delaying
  `close` indefinitely. The patched **shell-only** adapter resolves on `exit`,
  allows 250 ms to drain output, then drains/discards inherited pipe output
  without killing the server or retaining its output in memory. Generic process
  consumers retain their original close-based semantics.
- Signal termination makes `handle.exitCode` fail. Effect's `raceAll` waits for
  the first **success**, so that failure does not beat an outstanding timeout.
  Materializing the exit result makes both success and failure terminal events;
  the signal is returned in tool output, not misreported as a timeout.
- On cancellation/timeout, terminate the process group and escalate to SIGKILL
  after three seconds, even if the foreground shell exits before its children.
  Normal completion preserves background processes, including their pipe writes.

This targets the command-execution failure class implicated in the
`hf-model-inference` straggler. It does **not** fix an independently hanging
candidate verifier or guarantee that every long task becomes fast.

## Reproduce

In a fresh checkout of the pinned OpenCode commit, with Bun 1.3.14 installed:

```bash
git apply --check /absolute/path/to/1.18.8-shell-completion.patch
git apply /absolute/path/to/1.18.8-shell-completion.patch
bun install --frozen-lockfile --ignore-scripts
cd packages/opencode
bun typecheck
bun test test/tool/shell.test.ts
OPENCODE_VERSION=1.18.8-fw-shell.2 bun run script/build.ts \
  --single --skip-install --skip-embed-web-ui
```

Upload `dist/opencode-linux-x64/bin/opencode` and the adjacent
`../shell_completion_probe.py` from this cookbook to an **isolated E2B sandbox
using the actual task template**, not the small default E2B base sandbox. Then:

```bash
chmod +x /tmp/opencode-fixed
/tmp/opencode-fixed --version
python3 /tmp/shell_completion_probe.py /tmp/opencode-fixed
```

The probe runs the actual CLI and built-in bash tool against a deterministic
loopback provider. No GPU/model requests, candidate task files, or live RL
processes are involved. `--cases signal,background` selects just those probes.
An unpatched binary is expected to fail these assertions.

## Verified artifact — 2026-09-15

- Binary: `1.18.8-fw-shell.2`, Linux x64.
- SHA-256: `6b160847e94b9ecfa608436d23497de08ff5c93f30d9174bcf3fb0c45134ca88`.
- E2B task template: `7tpfet9m64fl12oaeafj`, 8 GiB; isolated test sandbox
  `i1709jru3sr7soat37b35`.
- Type check passed; 27 shell-tool tests and 48 generic process/spawner tests
  passed. Coverage includes cancellation, output truncation, permission checks,
  server survival, and a child that ignores SIGTERM.
- All five actual-CLI probes passed. Stub dispatch to next provider request:

| Probe | Time | Result |
| --- | ---: | --- |
| Nonzero exit | 0.164 s | Exit 7; final stdout/stderr preserved |
| SIGTERM | 0.163 s | Signal reported; no timeout wait |
| SIGSEGV | 0.163 s | Signal reported; no timeout wait |
| Background server launch | 0.445 s | Exit 0; server remains responsive on next call |
| 300-ms timeout | 3.538 s | Timeout reported, including 3-second group cleanup grace |

These are command round trips, not RL/model throughput measurements.
The unmodified `1.18.8` binary was tested in the **same sandbox**: SIGTERM waited
5.244 s and was incorrectly reported as a timeout; the background command waited
5.251 s, timed out, and its server was unavailable to the next tool call. These
probes use a 5-second timeout intentionally, rather than the live 6,900 seconds.

## Activation boundary

The running RL client, trainer, rollout, and existing sandbox processes were
**not restarted or hot-patched** during this work. The patch is not yet active in
that run. Activation requires installing the patched binary in fresh task
sandboxes and explicitly updating the supported/pinned OpenCode build. The
current harness intentionally rejects unrecognized versions; do not relabel
this binary as unmodified `1.18.8` to bypass that check. Preserve the current
training state and use a checkpoint-safe client transition, not a step-0 restart.
