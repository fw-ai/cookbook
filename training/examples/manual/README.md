# Manual infra tests

Two scripts that drive `training/recipes/rl_loop.py` against real
trainers + deployments to cover paths the unit tests can't and that the
[single-shape e2e CI](https://github.com/fw-ai/fireworks/actions/runs/24703610604)
does not hit.

| Script | Scope under test | What it runs |
|---|---|---|
| `test_reattach_manual.py` | `WeightSyncScope.PER_TRAINER` **re-attach** path | `rl_loop.main` twice sharing one `deployment_id`; the second run must PATCH `hot_load_trainer_job` onto the existing deployment. |
| `test_per_deployment_manual.py` | `WeightSyncScope.PER_DEPLOYMENT` | Single `rl_loop.main` run where `setup_infra` provisions the deployment-owned bucket, then launches trainers with `hot_load_deployment_id`. |

Both scripts reuse the deepmath reward + dataset from
`training/examples/rl/deepmath/` and pin to the 1×GPU qwen3-4b minimum
shapes so one run is ~15–20 min on shared dev.

## Usage

Set a pyroworks-dev-scoped key (targets `https://dev.api.fireworks.ai`):

```bash
export FIREWORKS_API_KEY=<pyroworks key>
python training/examples/manual/test_reattach_manual.py
python training/examples/manual/test_per_deployment_manual.py
```

Flags worth knowing:

- `--deployment-id <id>` — reuse a warmed deployment across invocations
  (e.g. inspect the pod after the second run).
- `--keep-resources` — skip trainer cancellation + deployment
  scale-to-zero on exit. Default is to clean up.

## What to look for in the logs

**Re-attach (PER_TRAINER, second run):**

```
Re-attached deployment <dep_id> to trainer <new_trainer> (prev_pod=<old>) —
  settling in parallel with trainer waits
Re-attach settled: new pod <new> replaced <old>
```

The script also asserts that run-1 and run-2 produced *different* trainer
job IDs — a silently reused trainer would mean the re-attach path was
never exercised.

**PER_DEPLOYMENT:**

```
Creating deployment: <dep_id>          # deployment-owned bucket provisioned first
Creating policy trainer job '...' ...  # then trainers, with hot_load_deployment_id
```

No `Re-attached deployment ...` lines — `PER_DEPLOYMENT` never PATCHes.

The script asserts `config.deployment.hot_load_trainer_job is None` to
catch a regression where `PER_DEPLOYMENT` accidentally takes the
`PER_TRAINER` wiring.
