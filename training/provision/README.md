# Fireworks training provisioning

`provision.py` stands up the Fireworks infrastructure a training recipe needs
(trainer job, optional rollout deployment, optional reference/teacher), keeps it
alive while you work, and tears it down on exit. It is config-driven: you pick a
**recipe** from a YAML file and the script provisions the matching resources.

This is a provisioning helper, not the training loop itself. Use it to hold a
trainer + deployment open (for example while iterating on a recipe), then point
your recipe at the resources it created.

## Quick start

```bash
# From this directory, with its own uv environment:
export FIREWORKS_API_KEY=fw-...

uv run python provision.py --config-name fireworks_sft
uv run python provision.py --config-name fireworks_rft
uv run python provision.py --config-name fireworks_dpo
uv run python provision.py --config-name fireworks_distillation
```

The process prints a heartbeat line per interval and periodically health-checks
the resources. Press `Ctrl+C` to clean up and exit.

## How it works

### Config files

Each YAML file has three reusable building blocks plus a `recipe` block:

- `common` — defaults merged into every recipe (`base_model`, `tokenizer_model`,
  `lora_rank`, `learning_rate`, `max_seq_len`, `step_timeout`,
  `weight_sync_timeout`).
- `deployments` — named deployment configs (e.g. `rollout`).
- `trainers` — named trainer configs (e.g. `policy`, `reference`) referenced by
  recipe entries via `trainer:` / `reference_trainer:`. A trainer may set
  `weight_sync_deployment:` to name the deployment that receives hot-loaded
  weights.

Each entry under `recipe` is a recipe you launch by name. A recipe inherits
`common`, then references the building blocks it needs:

```yaml
common:
  base_model: accounts/fireworks/models/qwen3p5-9b
  tokenizer_model: Qwen/Qwen3.5-9B
  lora_rank: 16

deployments:
  rollout:
    replica_count: 1

trainers:
  policy:
    training_shape_id: accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora
    weight_sync_deployment: rollout

recipe:
  rl:
    trainer: policy
```

### Modes

Each recipe resolves to a **mode** that determines what gets provisioned. The
mode is taken from an explicit `mode:` key, else inferred from the recipe name
(`sft`, `rl`, `distillation`, `dpo`, or a `<mode>_*` prefix).

| Mode | Provisions |
|------|------------|
| `sft` | policy trainer (no deployment) |
| `rl` / `rft` | policy trainer + rollout deployment + sampler; reference trainer when `kl_beta > 0` |
| `dpo` | policy trainer + frozen reference (no deployment) |
| `distillation` | policy trainer + rollout deployment + teacher inference deployment |

`rft` is an alias for `rl` (same training mode).

### Selecting a config

- `--config-name NAME` loads `NAME.yaml` from this directory.
- `--config PATH` loads an explicit file path.
- With neither, it defaults to `fireworks.yaml` (the all-in-one file with
  `sft`, `rl`, and `distillation` recipes).
- `--recipe NAME` selects a recipe within the loaded file. If a file defines a
  single recipe it is auto-selected and `--recipe` is optional.

```bash
# All-in-one file: pick a recipe explicitly
uv run python provision.py --recipe rl

# Per-method file: recipe is auto-selected
uv run python provision.py --config-name fireworks_sft

# Arbitrary path
uv run python provision.py --config /path/to/my_fireworks.yaml --recipe rl
```

### Lifecycle and flags

- Resources the script creates are cleaned up on exit; resources you reattach to
  (by setting `job_id` / `deployment_id` in the YAML) are left running.
- `--cleanup-existing` also tears down reattached resources on exit.
- `--progress-interval-s` (default 15) controls heartbeat cadence.
- `--health-check-interval-s` (default 60) controls how often resources are
  health-checked; if any becomes unhealthy the monitor reports it and exits.

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `FIREWORKS_API_KEY` | yes | API key for provisioning |
| `FIREWORKS_BASE_URL` | no | API base URL (default `https://api.fireworks.ai`) |
| `FIREWORKS_API_EXTRA_HEADERS` | no | Extra request headers (JSON or `k=v,k=v`) |

## Shipped configs

| File | Recipe(s) |
|------|-----------|
| `fireworks.yaml` | `sft`, `rl`, `distillation` (all-in-one) |
| `fireworks_sft.yaml` | `sft` |
| `fireworks_rft.yaml` | `rft` (alias for `rl`) |
| `fireworks_dpo.yaml` | `dpo` |
| `fireworks_distillation.yaml` | `distillation` |

Copy any of these and edit the `common` / `trainers` / `deployments` blocks for
your model and shapes.

## Tests

```bash
uv run pytest
```
