# Recipes: what to pass, how resources are reused

Every recipe is a single Python file in `training/recipes/` with a `Config` dataclass at the top and a `main()` entry point. This page documents **what fields you pass in** and **the reuse semantics for trainer jobs and deployments** -- it does not explain what each algorithm does (that's in the Fireworks docs).

## Config fields that appear in every recipe

| Field | Type | Required? | Purpose |
|-------|------|-----------|---------|
| `dataset` | `str` | yes | JSONL path or `gs://` URL |
| `base_model` | `str` | yes | Fireworks model ID (e.g. `accounts/fireworks/models/qwen3-8b`) |
| `max_seq_len` | `int` | yes | Max token length |
| `log_path` | `str` | yes | Where to write `checkpoints.jsonl`, `metrics.jsonl`, `status.json` |
| `infra` | `InfraConfig` | yes | Trainer infra; see table below |
| `wandb` | `WandBConfig \| None` | no | `entity`, `project`, `run_name` |
| `resume` | `ResumeConfig \| None` | no | Resume from a previous checkpoint |

Recipe-specific required fields:

| Recipe | Also required |
|--------|---------------|
| `sft_loop.py` | `tokenizer_model` (HF name) |
| `dpo_loop.py` | `tokenizer_model`, `ref_training_shape_id` (or auto) |
| `orpo_loop.py` | `tokenizer_model` |
| `rl_loop.py` | `deployment: DeployConfig`, `weight_sync: WeightSyncConfig`, `reward_fn`, `policy_loss` |
| `igpo_loop.py` | Same as `rl_loop.py` plus `turn_boundary_detector` |

## InfraConfig fields

Trainer infra. **All optional** -- auto-selection fills in whatever you leave unset.

| Field | Behavior when set | Behavior when unset |
|-------|------|--------|
| `training_shape_id` | Uses this exact shape (must be validated) | Cookbook auto-selects a validated policy shape for `base_model` |
| `ref_training_shape_id` | Forward-only shape for DPO reference | Cookbook auto-selects |
| `region` | Creates trainer in this region | Inferred from shape |
| `accelerator_type` / `accelerator_count` / `node_count` | Manual-path launch (server skips shape validation) | Backend owns these via shape |
| `custom_image_tag` | Pins trainer image | Uses shape's default tag |
| `trainer_timeout_s` | Seconds to wait for `RUNNING` + healthy | Default 3600 |
| `extra_args` | Passed through to trainer container | None |
| `skip_validations` | Skip server-side shape validation | Validate |

**Do not mix**: if `training_shape_id` is set, leave `accelerator_type`/`accelerator_count` unset -- the server rejects the combo.

## DeployConfig fields (RL recipes only)

| Field | Behavior when set | Behavior when unset |
|-------|------|--------|
| `deployment_id` | **Reuse**: fetches existing deployment; re-attaches to the new trainer if live, creates if `FAILED`/`DELETED` (see below) | Auto-generated ID derived from model name + timestamp -- fresh deployment every run |
| `deployment_shape` | Uses this validated shape | Auto-select from control-plane data |
| `deployment_region` | Override region | Inferred from shape |
| `hot_load_bucket_type` | Default `"FW_HOSTED"` | — |
| `hot_load_trainer_job` | Pre-points the deployment at a specific trainer's bucket | Set automatically when `setup_or_reattach_deployment` wires trainer ↔ deployment |
| `deployment_timeout_s` | Default 5400 | — |
| `tokenizer_model` | **Required for RL** (client-side tokenization) | Fails fast |
| `disable_speculative_decoding` | Default `True` for RL (numerical stability) | — |
| `replica_count` | Fixed replica count | Default from shape (usually `min=0, max=1`) |
| `extra_args` | Pass-through to serving container (e.g. `--enable-moe-stats`) | None |
| `extra_values` | Extra helm values (e.g. `devShmSize=200Gi`) | None |

## Trainer reuse semantics

**`create_trainer_job(..., job_id=None)`** -- creates a new trainer.

**`create_trainer_job(..., job_id="my-trainer-v1")`** -- reuses an existing trainer:

- If job is `RUNNING` + healthy: returns the endpoint immediately (no HTTP round-trip to recreate)
- If job is `FAILED`/`CANCELLED`/`PAUSED`: resumes it, waits for `RUNNING`
- If job is `CREATING`/`DELETING`: waits briefly (up to `max_wait_for_resumable_s`) then errors
- Uses `TrainerJobManager.reconnect_and_wait()` under the hood

To **reuse across runs**, save the `job_id` from the first run and pass it on subsequent runs. Example:

```python
# First run: create fresh
endpoint = create_trainer_job(rlor_mgr, base_model=..., infra=infra, profile=profile)
print(endpoint.job_id)  # save this: "dev-chengxili-grpo-v1"

# Subsequent runs: reuse
endpoint = create_trainer_job(rlor_mgr, base_model=..., infra=infra, job_id="dev-chengxili-grpo-v1")
```

**Base-URL shortcut**: if you have both `job_id` and `base_url_override`, the cookbook skips all health polling and constructs the endpoint directly. Useful for smoke tests that already know the gateway URL.

## Deployment reuse semantics

RL recipes use `setup_or_reattach_deployment()` (see [`reattach.md`](reattach.md)):

| Given | Behavior |
|-------|----------|
| `deployment_id=None` | Creates fresh deployment with auto-generated ID |
| `deployment_id` set, deployment exists and is `READY`/`UPDATING` | **Reuses**: PATCHes `hotLoadTrainerJob` to point at the new trainer, waits for rolling restart, resets `WeightSyncer` delta chain |
| `deployment_id` set, deployment is `FAILED`/`DELETED`/`DELETING` | Creates fresh deployment with that ID |
| `deployment_id` set, deployment exists but different shape needed | **Will NOT** change shape -- you must delete and recreate |

SFT/DPO/ORPO don't create deployments (they're training-only).

## Reuse pattern for rapid iteration

Keep the deployment warm, cycle trainers:

```python
# Config persisted across runs:
config = Config(
    deployment=DeployConfig(deployment_id="my-rft-deploy"),    # same ID every time
    infra=InfraConfig(training_shape_id="ts-qwen3-8b-policy"),
    ...
)

# Run 1: both created fresh
main(config)   # trainer T1, deployment D is READY

# Run 2: create fresh trainer T2, re-point D at T2 (~30s vs 15min cold start)
main(config)   # new T2, D re-attached
```

The reuse + re-attach flow is handled entirely by `setup_or_reattach_deployment` -- recipes don't need custom logic.

## Shape auto-selection

When `training_shape_id` is unset, the cookbook queries the control plane for validated shapes matching `base_model` and picks one. RL recipes also auto-select a validated deployment shape; DPO also auto-selects a forward-only reference shape.

Override explicitly when:

- You have a custom-built trainer image you need to pin (`custom_image_tag`)
- You want a specific GPU type / count (switch to manual path: set `accelerator_type`, omit shapes)
- The auto-selected shape is region-locked out of your account

## Loss selection (RL)

`rl_loop.py`'s `policy_loss` field picks the algorithm: `"grpo"` (default), `"dapo"`, `"gspo"`, `"dro"`, `"cispo"`, `"importance_sampling"`, `"reinforce"`. All accept the same `(policy_logprobs, inference_logprobs, advantages)` signature. See [`rl.md`](rl.md) for the numerical details of each.

## Sample CLI invocations

Fresh run, auto-select everything:

```bash
python -m recipes.rl_loop
# Uses defaults from the Config at the bottom of the file
```

Reuse trainer and deployment:

```python
# Edit recipes/rl_loop.py's __main__ block:
config = Config(
    dataset="my_prompts.jsonl",
    base_model="accounts/fireworks/models/qwen3-8b",
    deployment=DeployConfig(deployment_id="my-rft-deploy", tokenizer_model="Qwen/Qwen3-8B"),
    weight_sync=WeightSyncConfig(weight_sync_interval=1),
    infra=InfraConfig(training_shape_id="ts-qwen3-8b-policy"),
    # No trainer_id here -- create_trainer_job will create fresh.
    # To reuse a trainer too, pass job_id in create_trainer_job(...)
)
main(config)
```

Manual-path (pin everything):

```python
config = Config(
    ...,
    infra=InfraConfig(
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        region="US_OHIO_1",
        custom_image_tag="0.0.0-dev-chengxili-grpo-v1",
        skip_validations=True,
    ),
)
```
