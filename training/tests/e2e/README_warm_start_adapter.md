# E2E: HF adapter warm-start

Two complementary tests exercise the `warm_start_from_adapter` pipeline end-to-end against real Fireworks infrastructure.

| Test | Scope | Account | How to run |
|---|---|---|---|
| `test_sft_warm_start_adapter_e2e.py` | Cookbook-direct: `sft_loop.main` → SDK → Tinker trainer. Skips CP/gateway. | `pyroworks` | `pytest -m e2e training/tests/e2e/test_sft_warm_start_adapter_e2e.py` |
| `run_warm_start_managed_v2.sh` | Full managed V2: firectl → gateway → CP activity → orchestrator → cookbook → SDK → Tinker trainer. | `pyroworks` | `./training/tests/e2e/run_warm_start_managed_v2.sh` |

Both use `accounts/fireworks/models/qwen3-4b-minimal-lora` as the warm-start source (pre-promoted LoRA adapter retained in the `pyroworks` account for this test).

## Pass criteria

**Cookbook-direct test:**
- `sft_loop.main()` completes, producing at least one optim step.
- `initial_loss < 6.0` — proves LoRA weights were populated from the adapter before the first `forward_backward`. Random-init loss on Qwen3-4B chat is ~9-11; the minimal-lora adapter on the canned arithmetic dataset lands around 1-3.
- A "fresh start" control run (same config, no `warm_start_from_adapter`) completes for regression comparison.

**Managed V2 test:**
- `firectl create sft --use-v2 --warm-start-from <Model>` is accepted by the gateway.
- Trainer pod reaches `RUNNING`.
- Orchestrator logs contain both:
  - `Fresh start with HF adapter: gs://…` — proves `resolve_resume` took the adapter branch.
  - `Adapter loaded (X.Xs)` — proves the SDK `load_adapter` RPC returned success.
- Job reaches `JOB_STATE_SUCCEEDED`.
- Output Model is `Kind=HF_PEFT_ADDON`, `State=READY`.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | *(required)* | pyroworks API key |
| `FIRECTL_PROFILE` | `pyroworks` | firectl profile name (shell script only) |
| `FIREWORKS_E2E_BASE_MODEL` | `accounts/fireworks/models/qwen3-4b` | Base model under the adapter |
| `FIREWORKS_E2E_TOKENIZER` | `Qwen/Qwen3-4B` | HF tokenizer name |
| `FIREWORKS_E2E_WARM_START_ADAPTER_URI` | *(auto-resolved via `firectl get model`)* | Override adapter `gs://` URI |
| `FIREWORKS_E2E_REGION` | `US_OHIO_1` | Training region |
| `FIREWORKS_CUSTOM_IMAGE_TAG` | `0.33.0` | Trainer image tag |
| `JOB_TIMEOUT_SECS` | `3600` | Shell script polling timeout |

## Why two tests

- **Cookbook-direct** is fast, checks the SDK + cookbook layers in isolation, and asserts on a quantitative signal (initial loss). Good for catching regressions in `resolve_resume` / `load_adapter` wiring before they reach CP.
- **Managed V2** catches CP-side breakage: Model → GcsUri resolution, Kind/State validation, `cfg.WarmStartFromAdapter` plumbing through the Go struct → JSON → Python dataclass boundary. Covers the path a real customer takes via `firectl` / UI.

Run both in CI before shipping changes to any of: `resolve_resume`, `validate_warm_start_config`, `FiretitanTrainingClient.load_adapter`, `CookbookTrainingConfig.WarmStartFromAdapter`, `resolveWarmStartAdapter`.

## When the minimal-lora Model is re-promoted

If the backing `qwen3-4b-minimal-lora` adapter is re-promoted, its `GcsUri` changes but the resource name stays. Both tests auto-resolve the URI from the resource name each run, so no test edit is needed.

If the resource name itself changes, update `DEFAULT_WARM_START_MODEL` in the pytest file and `WARM_START_MODEL` default in the shell script.
