# Renderer verifier

Two complementary probes, intentionally separate so CI doesn't depend
on a deployment and the live-only checks don't pretend to validate
renderer↔upstream HF intent on their own.

| Layer | Question it answers | Code | Output | Where it runs |
|---|---|---|---|---|
| **CPU — HF parity** | Does the renderer match upstream HF's canonical chat template (byte-for-byte)? | `hf_parity.py` + `tests/unit/test_renderer_hf_parity.py` | pytest assertions (no JSON) | Cookbook PR CI on every change |
| **Live — empirical probe** | Does the renderer match what the deployed model actually emits? | `probe.py` (`verifier render` CLI) | JSON audit-table artifact | Manual / nightly, needs a Fireworks API key |

A renderer can pass one and still fail the other; both are needed.
A live FAIL with a CPU PASS means the **gateway is stale** relative
to upstream HF. A CPU FAIL with a live PASS means the renderer +
gateway agree but both have drifted from upstream HF. A FAIL on
both means the renderer is wrong by every reference.

The live probe is observational only — it produces no verdicts.
Spec-driven verdict layers (Phase 1) ship in follow-up PRs and
consume the same JSON envelope.

## Install

The verifier is part of the cookbook package. Nothing extra to install
beyond the standard cookbook dev environment.

## Run

### CPU layer — HF parity tests

```bash
# Runs as part of cookbook's standard pytest pass; nothing extra to invoke
# in CI. To run locally:
python -m pytest training/tests/unit/test_renderer_hf_parity.py -v
```

The tests skip cleanly when the upstream tokenizer can't be downloaded
(network outage, gated repo) so CI without HF Hub access still
completes. Cases marked `xfail` track known divergences (see
`renderer-verifier-findings.md` in this workspace) and flip to PASS
when fixed; a fix landing makes the test green automatically.

### Live layer — empirical probe

```bash
# 1) Spin up a deployment. The default --shape is the GLM5-on-B300
#    versioned deployment shape; --shape also accepts a training-shape
#    resource and resolves its pinned deployment_shape_version.
MODEL=$(python -m training.verifier.spinup_deployment up \
    --base-model accounts/fireworks/models/glm-5p1 \
    --shape accounts/fireworks/deploymentShapes/glm-5p1-b300/versions/jqami1br \
    --deployment-id my-glm5-probe | tail -1)

# 2) Probe one input
python -m training.verifier render \
    --renderer glm5 \
    --tokenizer-model zai-org/GLM-5.1 \
    --model "$MODEL" \
    --input training/verifier/examples/glm5_single_turn.json \
    --output probes/glm5-single-turn.json

# 3) Tear down when done
python -m training.verifier.spinup_deployment down --deployment-id my-glm5-probe
```

`--model` is the Fireworks model identifier passed to
`chat.completions.create`. For a personal deployment the spinup helper
prints `accounts/<acct>/deployments/<id>` on its last stdout line; pipe
that through to the probe via `--model`.

`spinup_deployment` accepts either a deployment shape or a training
shape; for either, versioned input is used as-is and unversioned input
resolves to the latest validated version. Training-shape input is
resolved to the pinned `deployment_shape_version` from its profile.

`--api-key` falls back to `FIREWORKS_API_KEY`. `--base-url` falls back
to `FIREWORKS_BASE_URL`.

### Input file shape

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "tools": [],
  "renderer_config": {}
}
```

`tools` and `renderer_config` are optional. `renderer_config` is
recorded in the artifact for traceability; renderer-specific
configuration (e.g. `strip_thinking_from_history`) is not yet wired
into instantiation in this PR.

## Output artifact

JSON, one file per probe. Top-level fields:

| field | meaning |
|---|---|
| `schema_version` | bump on incompatible schema changes |
| `kind` | `"probe"` (later artifacts: `"l1"`, `"l2"`) |
| `renderer` | name + config + train_on_what mode |
| `tokenizer` | model id + special-token id→string map |
| `deployment` | model id, optional deployment id, sampling settings |
| `input` | raw messages and tools (before apply_chat_template) |
| `render.prompt` | renderer's prompt-only render: tokens + decoded |
| `render.full` | renderer's full conversation render (orig + assistant completion) |
| `completion` | model's emitted text + token ids + stop reason |
| `sanity` | quick consistency checks (renderer prompt vs API prompt, etc.) |
| `audit_table` | one row per token of the full render — see below |

### Audit table row

```json
{
  "idx": 12,
  "token_id": 14990,
  "decoded": "Hello",
  "chunk_source": "output",
  "msg_idx": 1,
  "role": "assistant",
  "renderer_claim_weight": 1.0,
  "provenance": "native_generated"
}
```

- `chunk_source` ∈ `{bos, header, output, stop_overlap, generation_suffix}`
  — what the renderer says this token is.
- `renderer_claim_weight` — the training weight the renderer would
  assign at this position under the chosen `train_on_what` mode.
- `provenance` ∈ `{prompt_hard_append, native_generated,
  trailing_hard_append, tokenization_diverged}` — empirical:
  - `prompt_hard_append`: token was in the prompt the deployment received.
  - `native_generated`: token matches what the deployment emitted.
  - `trailing_hard_append`: renderer added tokens after the model's
    emission ended (e.g. a next-turn role tag included as
    `stop_overlap`). This is the position class the GLM5 #400 fix
    touched.
  - `tokenization_diverged`: the renderer's re-tokenisation of the
    completion text disagrees with the deployment's token stream.

### What to look for as a spec author

- Rows where `renderer_claim_weight = 1.0` but `provenance =
  trailing_hard_append`. The renderer is training the model to emit a
  token the model never actually emits — the GLM5 #400 bug class.
- Rows where `renderer_claim_weight = 0.0` but `provenance =
  native_generated`. The renderer is masking out a token the model
  must emit at inference.
- Any `tokenization_diverged` rows. Re-tokenisation mismatches between
  the renderer and the deployment usually indicate special-token
  registration drift or BPE-merge differences.
- `sanity.renderer_prompt_matches_api_prompt = false` means the
  renderer's prompt-time bytes disagree with what the deployment sees.
  That blocks any further conclusion — fix it before reading the audit
  table.
- `sanity.completion_stop_reason != "stop"` means the model didn't
  finish naturally (`length` truncation hit `--max-tokens`). The
  stop-overlap row in the audit table can't be empirically classified
  in that case — bump `--max-tokens` and re-run before drawing
  conclusions about the trailing-token bucket.

### How the round-trip is constructed

After getting the model's completion tokens back, the probe runs them
through `renderer.parse_response()` to recover a *structured* assistant
message (with thinking / tool_calls separated from `content`) and
re-renders the conversation. Without this, Camp A renderers (GLM5)
double-count the trailing role tag — once as part of the embedded
content string, once as `stop_overlap` — and the audit table becomes
unreadable. `sanity.parse_response_ok` reports whether the parser
accepted the tokens. If it's `false`, the probe falls back to feeding
the raw text in as `content` (best-effort) and you should treat the
trailing rows of the audit table with care.

## Verified against (this PR)

CPU HF parity (CI):

| Renderer | Tokenizer | Result |
|---|---|---|
| `glm5` | `zai-org/GLM-5.1` (single + multi-turn) | PASS |
| `qwen3` (default thinking) | `Qwen/Qwen3-8B` (single + multi-turn) | PASS |
| `qwen3_disable_thinking` | `Qwen/Qwen3-8B` (`enable_thinking=False`) | PASS |
| `kimi_k25` | `moonshotai/Kimi-K2.5` | PASS |
| `minimax_m2` | `MiniMaxAI/MiniMax-M2` | XFAIL — extra `\n` after `<think>` in renderer's generation suffix |

Live empirical probe (manual):

| Renderer | Camp | Model | Result |
|---|---|---|---|
| `glm5` | A | `accounts/fireworks/models/glm-5p1` | Clean. Trailing `<\|user\|>` is `stop_overlap` / `w=1.0` / `native_generated` (model emits it; training reinforces). Leading `<think>` is `output` / `w=0.0` / `prompt_hard_append`. |
| `qwen3` (thinking) | B | `accounts/fireworks/models/qwen3-8b` | Clean. Trailing `<\|im_end\|>` is `output` / `w=1.0` / `native_generated`. |
| `qwen3_disable_thinking` | B | `accounts/fireworks/models/qwen3-8b` | FAIL prompt parity vs gateway, but **CPU passes** → gateway is stale on the disable-thinking flag, not a renderer bug. |

The combination is what makes triage tractable. CPU-pass + live-fail
isolates gateway staleness; CPU-fail + live-fail isolates renderer
bug; CPU-pass + live-pass is full validation.

## What's next

Follow-up PRs in this series:

1. `chunk_rules` YAML spec format and `verifier check` command (L1).
2. Specs for tier-1 renderers (GLM5, Kimi K2.5) authored from probe
   artifacts; PR fix-history coverage tests.
3. Corpus runner and L2 deployment checks; CI wiring.
