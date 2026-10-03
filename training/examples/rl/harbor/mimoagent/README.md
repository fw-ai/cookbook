# mimoagent harness adapter

Runs Xiaomi MiMo's **mimoagent** as a Harbor agent against the
environment-local TITO sidecar: the trial lifecycle, allowlist networking,
and task dirs come from Harbor; the agent loop, tools, and sampling come from
mimoagent itself.

## Provenance and license

mimoagent (<https://github.com/XiaomiMiMo/mimoagent>, branch `mimo-oss`) is
Xiaomi MiMo's agentic rollout framework, an edited fork of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 1.9.0.
Both are MIT-licensed: mimoagent copyright (c) 2026 Xiaomi Corporation;
mini-swe-agent copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez.
This adapter installs mimoagent from the pinned upstream commit at image
build time and redistributes no upstream source; the upstream `LICENSE.md`
and `NOTICE` apply to the installed package.

## Layout

| File | Role |
|---|---|
| `constants.py` | pinned upstream commit and version |
| `prepare_tasks.py` | bake uv + standalone CPython 3.12 + pinned mimoagent into task images (mimoagent requires Python 3.12 exactly, which task images do not provide) |
| `driver.py` | one-shot in-sandbox driver: `BashOnlyAgent` + `LocalEnvironment` + `OpenAIChatModel`, sampling temperature 1.0 / top-p 0.95 / top-k 20 as in the MiMo code recipe |
| `agent.py` | `ConfigurableMimoAgent`: installs the TITO sidecar, uploads the instruction, runs the driver |
| `rollout.py` | rollout runner (`make_rollout_fn`) over Harbor task rows |

## Tested

- Image preparation pins the commit and preserves the final image `USER`
  (unit test in `tests/unit/test_harbor_tito.py`); the venv works for a
  non-root user (build-time `nobody` import check in the image layer).
- Driver arms: `bash` and `cc` (mini-claude-code) both construct against
  real mimoagent 0.1.0 with their recipe tool catalogues; unknown arms fail
  fast.
- MCP: `datasets/mimo/mcp/client.py` round-trips against a real
  streamable-HTTP MCP server in tests (list/call/error); the tool glue
  registers `mcp__<server>__<fn>` tools into the agent catalogue.
- End-to-end smoke on 2 DeepSWE control tasks × 1 repeat through Harbor +
  TITO: trajectory captured and materialized, verifier scored, egress
  allowlist enforced.

## Supported but not yet run

- Code-pool / cyber / terminal_bench task dirs (any `datasets.mimo` task with
  a baked image works the same way; only DeepSWE has been smoke-run).
- general_agent tasks with MCP tools through the driver (unit-tested wiring;
  no live smoke yet).
- The `mimocode` arm: chat-protocol like the others, a driver entry plus its
  own smoke.
- Multi-repeat RL batches (the smoke used 2 × 1).

## Not supported

- **Music**: not a Harbor task at all — it is direct token sampling scored by
  `datasets.mimo.score_music`, with no sandbox and no agent loop.
- `codex-agent` (mini-codex): the only arm on the OpenAI **Responses**
  protocol; the TITO sidecar serves chat completions only.
- mimoagent's blackbox third-party CLI installers — they need the open
  internet, which the sandbox allowlist forbids.

## Usage

```bash
python -m training.examples.rl.harbor.mimoagent.prepare_tasks \
  --source <harbor task dir> --destination <prepared dir>
# build the images, set [environment] docker_image to the built tag, then run
# through mimoagent.rollout.make_rollout_fn.
```
