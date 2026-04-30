# Renderer Verifier — Skill

Validate that a cookbook renderer produces the same tokens the live
Fireworks gateway emits, and that loss weights are consistent with the
"hard-append → weight 0, native-generated → weight 1" rule. Visual
inspection happens in a local React GUI seeded by a Python probe.

## 0. Pre-requisites

- Cookbook checked out at `~/workspace_batching/cookbook` (or adapt the
  paths in the workspace runners).
- A dev `FIREWORKS_API_KEY` exported in your shell. None of the runners
  carry the key any more — they error out clean if it's not set.

```bash
export FIREWORKS_API_KEY=fw_...
```

## 1. Confirm the inspection rules

Open `training/verifier/inspect_rules.yaml` and read the rule list.
This file is the single source of truth for "worth a closer look"
combinations. The GUI tints matching tokens amber and the CLI scans
list each match's reason.

A rule has three keys:

```yaml
- id: trains-on-prompt-prefix
  when:
    provenance: prompt_hard_append
    trainable: true
  reason: trains on prompt_hard_append (prompt prefix tokens should have weight 0)
```

`when` is an AND of equality (or list-membership) conditions on row
fields. Supported fields: `provenance`, `chunk_source`, `role`,
`trainable` (derived from weight > 0.5), `special` (derived from
`token_id ∈ tokenizer.special_tokens`).

**Edit before each session.** Common reasons to change rules:
- Add a renderer-specific anomaly you want flagged.
- Soften / remove a rule that fires on intentional behaviour.

The Python evaluator and the JS evaluator are pure equality matchers
with zero hardcoded knowledge — delete the YAML and both surfaces
flag nothing.

## 2. Pick a workflow

You have two paths. They use the same underlying probe.

### 2a. Interactive (one prompt at a time)

Type messages in a form. Each `Run probe` appends a case below; cases
stack with delete buttons. Good for ad-hoc questions ("what does this
specific chat render to?").

```bash
./run.sh
# open http://localhost:8765/
```

### 2b. Batch (many prompts from a JSON catalogue)

Provide a JSON file with multiple prompts (or use the default
catalogue). The runner probes them all and the GUI opens with every
case stacked side-by-side. Good for regression sweeps after a renderer
change.

Default catalogue lives at `training/verifier/default_prompts.json`.
Schema (each entry is one probe input):

```json
{
  "cases": [
    {
      "name": "simple-math",
      "messages": [
        {"role": "system", "content": "Answer with one integer."},
        {"role": "user",   "content": "2 + 2 = ?"}
      ]
    }
  ]
}
```

Optional per-case keys: `tools`, `renderer_config`.

To run:

```bash
./triage.sh                                          # glm5, default catalogue
./triage.sh qwen3 Qwen/Qwen3-8B
./triage.sh glm5 zai-org/GLM-5.1 ./my-prompts.json   # custom corpus
```

## 3. Pre-flight (the runners do this for you)

Before any prompt-level API call, the triage runner prints and asks
you to confirm:

1. **RENDERER**
   - `name`, `status` — `registered ✓` if the renderer is in
     `RENDERER_SERVERLESS_DEFAULTS`; otherwise a warning that you must
     pass `--model` or `--deployment-id`.
   - `tokenizer` — the HF tokenizer that will be loaded.
   - `dispatch` — `serverless | deployment | explicit` and the resolved
     model identifier the gateway will see.
   - `ping` — a 1-token completion against the dispatch target.
     `reachable ✓` means the API answered; otherwise the runner aborts
     with the gateway's actual error (404, auth, quota, …) so you find
     out before you commit to the full corpus.

2. **PROMPTS**
   - Count, source path, and a one-line snippet of the last user
     message in each case.

Type `Y` to proceed, `N` (or Ctrl-C) to abort. Pass `--yes` / `-y` to
skip the prompt in scripted contexts.

## 4. Inspect in the GUI

The page opens with all cases loaded:

- Each case has its own token stream (left) and **sticky** detail
  sidebar (right). Below 900 px viewport the layout collapses to one
  column.
- **Background tint** = provenance (`prompt_hard_append`,
  `native_generated`, `trailing_hard_append`, `tokenization_diverged`).
- **Amber background + ripple** = the token matches at least one rule
  in `inspect_rules.yaml`. Hover the token to see the reasons.
- **Pink + bold** text = token id is in `tokenizer.special_tokens`.
- Hover any token → its full audit row updates the sidebar
  (idx, token_id, decoded, chunk_source, role, msg_idx, weight,
  provenance, inspect reasons).
- **Filters** at the top are unified chips by attribute (provenance,
  chunk_source, trainable, special, inspect-flag). They apply across
  every case.
- **`Sanity flags / Renderer args / Deployment / API args`** are
  collapsed under each case — open when you need them.
- Per-case **× delete** removes a case; **Clear all** wipes the page.
  The form values stay, so you can tweak and run again.

## 4b. (Optional) Sweep serverless availability across renderers

If you don't yet know which renderer to target, run a cross-renderer
sweep. Each entry in `RENDERER_SERVERLESS_DEFAULTS` is pinged with a
1-token completion and the result is reported as `✓` / `✗`.

```bash
PYTHONPATH=cookbook \
  python -m training.verifier.check_renderers          # all renderers
PYTHONPATH=cookbook \
  python -m training.verifier.check_renderers --renderer glm5
```

Exit code is 0 only when every probed renderer is reachable. Useful as
a first step when something looks off — the triage / interactive flow
both call the same `_check_serverless` helper for the single renderer
you're working with.

## 5. Commands cheat-sheet

```bash
# 0. Set credentials once per shell.
export FIREWORKS_API_KEY=fw_...

# 1. (Optional) Edit the rules.
$EDITOR cookbook/training/verifier/inspect_rules.yaml

# 2. Pick one of:

#    Interactive — single GUI, type prompts.
./run.sh

#    Batch — pre-flight + corpus + GUI auto-seeded.
./triage.sh                                                  # default catalog
./triage.sh qwen3 Qwen/Qwen3-8B                              # change renderer
./triage.sh glm5 zai-org/GLM-5.1 ./my-prompts.json           # custom corpus

#    One-shot smoke checks (advisory only):
./run-bug-check.sh           # MiniMax newline bug repro
./run-bug-glm5-think.sh      # GLM5 weight-rule hand-scan

# 3. Stop with Ctrl-C.
```

## 6. Files

| Path | Purpose |
|---|---|
| `training/verifier/inspect_rules.yaml` | Source of truth for inspection rules. |
| `training/verifier/inspect_rules.py` | Python equality-condition evaluator. |
| `training/verifier/default_prompts.json` | Default batch corpus. |
| `training/verifier/triage.py` | Batch runner with pre-flight + reachability ping. |
| `training/verifier/check_renderers.py` | Cross-renderer serverless reachability sweep. |
| `training/verifier/probe.py` | Core probe: render → API → align → audit table. |
| `training/verifier/serve.py` | Dev server (`/probe`, `/inspect_rules`, `/session`). |
| `training/verifier/viewer/index.html` | React GUI (single-file, CDN-hosted). |
| `triage.sh`, `run.sh`, `run-bug-*.sh` | Workspace-root convenience runners (never committed). |

## 7. Add a new case

```bash
$EDITOR cookbook/training/verifier/default_prompts.json
# add: { "name": "...", "messages": [ ... ], "tools": [ ... ] }
./triage.sh   # picks up the change on the next run
```

## 8. Add a new rule

```bash
$EDITOR cookbook/training/verifier/inspect_rules.yaml
# add a rule with id / when / reason — see existing entries for shape
# refresh the GUI page (server re-reads the YAML per request)
```
