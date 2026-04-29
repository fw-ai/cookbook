# Renderer verifier

Phase 0 of the renderer verifier: an **empirical probe**. Renders a
conversation with the cookbook renderer, asks a deployed Fireworks model
to complete the assistant turn, and emits a JSON artifact whose audit
table pairs the renderer's per-token claim against empirical provenance.

The probe is observational. It does not produce verdicts. Verdict
layers (L1 spec-driven CPU checks, L2 corpus-scale deployed checks)
ship in follow-up PRs and consume the same JSON envelope.

## Install

The verifier is part of the cookbook package. Nothing extra to install
beyond the standard cookbook dev environment.

## Run

```bash
python -m training.verifier render \
    --renderer glm5 \
    --tokenizer-model zai-org/GLM-5.1 \
    --model accounts/fireworks/models/glm-5p1 \
    --input examples/glm5_single_turn.json \
    --output viz/probes/glm5-single-turn.json
```

`--model` is the Fireworks model identifier passed to
`chat.completions.create`. For a personal deployment use
`accounts/<acct>/deployedModels/<id>`; spinning the deployment up or
down is out of scope for this tool — pair it with the existing helper
script (the same pattern `rl_loop` uses).

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

## What's next

Follow-up PRs in this series:

1. `chunk_rules` YAML spec format and `verifier check` command (L1).
2. Specs for tier-1 renderers (GLM5, Kimi K2.5) authored from probe
   artifacts; PR fix-history coverage tests.
3. Corpus runner and L2 deployment checks; CI wiring.
