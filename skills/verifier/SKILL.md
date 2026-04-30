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
- `HF_TOKEN` exported if the tokenizer you intend to load is gated /
  private (most public ones don't need it). On first load the
  tokenizer is fetched from HF and cached under
  `~/.cache/huggingface/`; later loads are offline-friendly. The
  verifier wraps tokenizer-load failures with a friendly error
  pointing at this prereq.

```bash
export FIREWORKS_API_KEY=fw_...
export HF_TOKEN=hf_...        # only if the tokenizer repo is gated
```

## 1. Pick a renderer / tokenizer / Fireworks model

The verifier carries **no static renderer→model mapping**. Every probe
needs you to supply both an HF tokenizer id and a Fireworks model id
(or a `deployment_id`). Discover what's currently serverless via:

```python
from fireworks import Fireworks
for m in Fireworks(api_key=...).models.list(account_id="fireworks"):
    print(m.name)
```

…or just open the GUI — the dev server's `/models` endpoint feeds the
`model` dropdown live.

Reference table of common pairings (verify before each session — the
serverless ids drift over time):

| Renderer | HF tokenizer | Fireworks serverless model (typical) |
|---|---|---|
| `glm5` | `zai-org/GLM-5.1` | `accounts/fireworks/models/glm-5p1` |
| `qwen3` | `Qwen/Qwen3-8B` | `accounts/fireworks/models/qwen3-8b` |
| `qwen3_disable_thinking` | `Qwen/Qwen3-8B` | `accounts/fireworks/models/qwen3-8b` |
| `kimi_k25` | `moonshotai/Kimi-K2.5` | `accounts/fireworks/models/kimi-k2p5` |
| `kimi_k25_disable_thinking` | `moonshotai/Kimi-K2.5` | `accounts/fireworks/models/kimi-k2p5` |
| `deepseekv3` | `deepseek-ai/DeepSeek-V3` | `accounts/fireworks/models/deepseek-v3p1` |
| `minimax_m2` | `MiniMaxAI/MiniMax-M2` | `accounts/fireworks/models/minimax-m2p7` |
| `llama3` | `meta-llama/Llama-3.3-70B-Instruct` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |

If the table looks stale, the live `/models` listing wins. Treat this
as a starting point, not a contract.

## 2. Confirm the inspection rules

Open `training/renderer/verifier/rules/inspect_rules.yaml` and read the rule list.
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

## 3. Pick a workflow

You have two paths. They use the same underlying probe.

### 3a. Interactive (one prompt at a time)

Type messages in a form. Each `Run probe` appends a case below; cases
stack with delete buttons. Good for ad-hoc questions ("what does this
specific chat render to?").

```bash
./run.sh
# open http://localhost:8765/
```

### 3b. Batch (many prompts from a JSON file)

Author your own JSON file with multiple prompts. The runner probes
them all and the GUI opens with every case stacked side-by-side. Good
for regression sweeps after a renderer change. **There is no default
corpus** — pick the cases that exercise the surface you care about.

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
./triage.sh <renderer> <tokenizer-model> <prompts.json>
# e.g.:
./triage.sh glm5  zai-org/GLM-5.1  ./my-prompts.json
./triage.sh qwen3 Qwen/Qwen3-8B    ./my-prompts.json
```

## 4. Pre-flight (the runners do this for you)

Before any prompt-level API call, the triage runner prints and asks
you to confirm:

1. **RENDERER**
   - `name`, `status` — `registered ✓` when the name is in the live
     `tinker_cookbook` renderer registry; otherwise `NOT REGISTERED`
     and the runner aborts.
   - `tokenizer` — the HF tokenizer that will be loaded.
   - `dispatch` — `deployment | explicit` and the resolved model
     identifier the gateway will see. There is no `serverless` mode and
     no static renderer→model fallback — you must pass `--model` (a
     Fireworks model id) or `--deployment-id`.
   - `ping` — a 1-token completion against the dispatch target.
     `reachable ✓` means the API answered; otherwise the runner aborts
     with the gateway's actual error (404, auth, quota, …) so you find
     out before you commit to the full corpus.

2. **PROMPTS**
   - Count, source path, and a one-line snippet of the last user
     message in each case.

Type `Y` to proceed, `N` (or Ctrl-C) to abort. Pass `--yes` / `-y` to
skip the prompt in scripted contexts.

## 5. Inspect in the GUI

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

## 5b. Reachability check

The triage runner pings the dispatch target with a 1-token completion
during pre-flight (the `ping` line in section 1 of the summary) — that
is the canonical reachability check. If you only want to verify a
renderer without running a full corpus, point triage at a one-case
prompt JSON and abort at the confirmation prompt. The pre-flight will
already have reported `reachable ✓` or the gateway error.

## 6. Commands cheat-sheet

```bash
# 0. Set credentials once per shell.
export FIREWORKS_API_KEY=fw_...

# 1. (Optional) Edit the rules.
$EDITOR cookbook/training/renderer/verifier/rules/inspect_rules.yaml

# 2. Pick one of:

#    Interactive — single GUI, type prompts.
./run.sh

#    Batch — pre-flight + corpus + GUI auto-seeded.
./triage.sh glm5  zai-org/GLM-5.1  ./my-prompts.json
./triage.sh qwen3 Qwen/Qwen3-8B    ./my-prompts.json

#    One-shot smoke checks (advisory only):
./run-bug-check.sh           # MiniMax newline bug repro
./run-bug-glm5-think.sh      # GLM5 weight-rule hand-scan

# 3. Stop with Ctrl-C.
```

## 7. Files

Layout (everything under `cookbook/training/renderer/verifier/`):

```
training/renderer/verifier/
├── SKILL.md                       this document
├── cli.py                         python -m training.renderer.verifier render | inspect
├── serve.py                       python -m training.renderer.verifier.serve
├── triage.py                      python -m training.renderer.verifier.triage
├── spinup_deployment.py           personal-deployment helper
├── utils/                         the verifier engine (importable)
│   ├── probe.py                   core probe: render → API → align → audit table
│   ├── inspect_rules.py           YAML rule loader + equality evaluator
│   ├── inspect.py                 pretty-printer for probe artifacts
│   └── hf_parity.py               CPU HF chat-template parity comparison
├── rules/                         data
│   └── inspect_rules.yaml         single source of truth for "worth inspecting"
└── viewer/
    └── index.html                 React GUI (single-file, CDN-hosted)
```

Workspace-root runners (never committed): `run.sh`, `triage.sh`,
`run-bug-*.sh`.

## 8. Author a prompt corpus

```bash
cat > my-prompts.json <<'EOF'
{
  "cases": [
    { "name": "...", "messages": [ {"role": "user", "content": "..."} ] }
  ]
}
EOF

./triage.sh glm5 zai-org/GLM-5.1 ./my-prompts.json
```

## 9. Add a new rule

```bash
$EDITOR cookbook/training/renderer/verifier/rules/inspect_rules.yaml
# add a rule with id / when / reason — see existing entries for shape
# refresh the GUI page (server re-reads the YAML per request)
```
