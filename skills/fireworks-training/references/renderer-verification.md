# Verifying a renderer

Validate that a cookbook renderer produces the same tokens the live
Fireworks gateway emits, and that loss weights are consistent with the
"hard-append → weight 0, native-generated → weight 1" rule. Visual
inspection happens in a local React GUI seeded by a Python probe.

## 0. Pre-requisites

- Cookbook checked out locally, with commands run from its repository root.
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

Reference table of common pairings (verify before each session — the
serverless ids drift over time):

| Renderer | HF tokenizer | Fireworks serverless model (typical) |
|---|---|---|
| `glm5` | `zai-org/GLM-5.1` | `accounts/fireworks/models/glm-5p1` |
| `qwen2_5` | `Qwen/Qwen2.5-32B-Instruct` | `accounts/fireworks/models/qwen2p5-32b-instruct` |
| `qwen3` | `Qwen/Qwen3-8B` | `accounts/fireworks/models/qwen3-8b` |
| `qwen3_disable_thinking` | `Qwen/Qwen3-8B` | `accounts/fireworks/models/qwen3-8b` |
| `kimi_k25` | `moonshotai/Kimi-K2.5` | `accounts/fireworks/models/kimi-k2p5` |
| `kimi_k25_disable_thinking` | `moonshotai/Kimi-K2.5` | `accounts/fireworks/models/kimi-k2p5` |
| `deepseekv3` | `deepseek-ai/DeepSeek-V3` | `accounts/fireworks/models/deepseek-v3p1` |
| `minimax_m2` | `MiniMaxAI/MiniMax-M2` | `accounts/fireworks/models/minimax-m2p7` |
| `llama3` | `meta-llama/Llama-3.3-70B-Instruct` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |

Treat this table as a starting point, not a contract.

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

## 3. Run the JSON-driven live UI

The web verifier has one path: load a validated JSON input catalog, execute
its cases sequentially against the configured live model or deployment, and
display only the fresh results from that run. It does not accept form-edited
requests or load saved output artifacts.

Catalog schema:

```json
{
  "schema_version": 1,
  "profiles": [
    {
      "id": "local-model",
      "label": "Local model checks",
      "defaults": {
        "renderer": "glm5",
        "tokenizer_model": "zai-org/GLM-5.1",
        "deployment_id": "accounts/your-account/deployments/your-deployment",
        "max_tokens": 256,
        "temperature": 0,
        "train_on_what": "last_assistant_turn"
      },
      "examples": [
        {
          "id": "simple-math",
          "label": "Simple math",
          "messages": [
            {"role": "system", "content": "Answer with one integer."},
            {"role": "user", "content": "2 + 2 = ?"}
          ]
        }
      ]
    }
  ]
}
```

Profiles provide shared defaults; examples can override probe fields. Optional
fields include `image_processor_model`, `tools`, and
`extra_completion_kwargs`. Keep credentials out of this file; the validator
rejects credential-bearing completion arguments.

```bash
python -m training.renderer.verifier.serve \
  --port 8765 \
  --input-file /path/to/verifier-input.json
# open http://localhost:8765/
```

The server re-reads and validates the input file for every case. The browser
posts only the selected profile and example IDs, clears previous results at
the beginning of a run, and sends the cases in file order.

### Offline triage

`training.renderer.verifier.triage` remains available for confirmed batch
inference and writes an offline result JSON. The web verifier intentionally
does not load that output.

The triage preflight itself sends a one-token paid inference ping before its
internal prompt. Obtain the user's confirmation before launching this command.

To run:

```bash
python -m training.renderer.verifier.triage \
  --renderer <renderer> \
  --tokenizer-model <tokenizer-model> \
  --model <accounts/fireworks/models/model-id> \
  --prompts <prompts.json> \
  --output /tmp/renderer-results.json
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

The page opens with the validated input cases and a `Run all sequentially`
button. Fresh completed results appear below:

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
- A malformed renderer parse keeps the raw completion and parse status but
  omits the supervised round-trip and token audit.

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

# 2. Run the live JSON suite.
python -m training.renderer.verifier.serve \
  --port 8765 \
  --input-file ./verifier-input.json

# 3. Optional offline triage (requires confirmation before paid inference).
python -m training.renderer.verifier.triage \
  --renderer glm5 \
  --tokenizer-model zai-org/GLM-5.1 \
  --model accounts/fireworks/models/glm-5p1 \
  --prompts ./my-prompts.json \
  --output /tmp/renderer-results.json

# 4. Stop the live UI with Ctrl-C.
```

## 7. Files

Layout (everything under `cookbook/training/renderer/verifier/`):

```
training/renderer/verifier/
├── README.md                      runtime overview
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

## 8. Author an offline triage corpus

```bash
cat > my-prompts.json <<'EOF'
{
  "cases": [
    { "name": "...", "messages": [ {"role": "user", "content": "..."} ] }
  ]
}
EOF

python -m training.renderer.verifier.triage \
  --renderer glm5 \
  --tokenizer-model zai-org/GLM-5.1 \
  --model accounts/fireworks/models/glm-5p1 \
  --prompts ./my-prompts.json \
  --output /tmp/renderer-results.json
```

## 9. Add a new rule

```bash
$EDITOR cookbook/training/renderer/verifier/rules/inspect_rules.yaml
# add a rule with id / when / reason — see existing entries for shape
# refresh the GUI page (server re-reads the YAML per request)
```
