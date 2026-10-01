# Vendored tinker-cookbook modules for SDFT

This directory holds the few
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
modules that the SDFT recipe (`training/recipes/sdft`) needs in a
Fireworks-modified form. Everything else the recipe uses is imported from
upstream `tinker-cookbook`, installed with the optional `sdft` extra:

```bash
# from cookbook/training
uv pip install -e '.[sdft]'
```

Only SDFT code (this directory, `training/recipes/sdft/` and
`training/utils/distillation/sdft.py`) imports `tinker_cookbook`. Without the
extra, the rest of the cookbook works as before, and importing
`training.recipes.sdft` raises an `ImportError` with the install command.

## Source and license

- Upstream: [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook),
  Copyright 2025 Thinking Machines Lab, licensed under Apache 2.0 (see
  [`LICENSE`](LICENSE)).
- Copied from the Fireworks fork
  [fw-ai-external/tinker-cookbook](https://github.com/fw-ai-external/tinker-cookbook)
  at commit `b1223e5535edea2efb824bfe73a38d35e0067439` (branch `fireworks`).
- The `sdft` extra pins upstream `tinker-cookbook==0.5.7` from PyPI. The fork
  is based on a slightly newer upstream commit
  (`6be6b03dd6f4b69b829b5f4a7f25255514a92ed4`); the one API difference SDFT
  hits is handled in `supervised/train.py` (see below). `[tool.uv]
  override-dependencies` keeps this project's `transformers==5.10.4` over the
  `transformers<=5.5.4` that tinker-cookbook declares.

## What changed

Each file below differs from upstream in two ways: the fork's Fireworks
changes, and import rewrites made when vendoring. Each file also carries a
header comment summarizing its changes.

| File | Fireworks changes (from the fork) |
| --- | --- |
| `checkpoint_utils.py` | Saves checkpoints through `FiretitanTrainingClient`: records the server-canonical `step-N` state name and sampler snapshot name so checkpoints load across trainer jobs. Adds `extract_trainer_job_id`. |
| `rl/train.py` | Trains on `FiretitanTrainingClient` (created with `FiretitanServiceClient`) and pushes sampler weights to a Fireworks deployment with `DeploymentManager` / `WeightSyncer`. Training logprobs come from a separate `cross_entropy` forward pass. The KL reference model is a frozen `FiretitanTrainingClient`. |
| `rl/metrics.py` | `incorporate_kl_penalty` computes reference logprobs with a frozen `FiretitanTrainingClient` forward pass instead of a `tinker.SamplingClient`. |
| `supervised/train.py` | Creates the training client with `FiretitanServiceClient` from `fireworks_base_model`, supports cross-job resume, rejects `load_checkpoint_path`, and tolerates backends that do not return per-datum logprobs. |

Vendoring changes, applied on top of the fork:

- `supervised/train.py` uses tinker-cookbook 0.5.7's `NLLEvaluator`, which
  computes test-set NLL with a `cross_entropy` forward on the training client,
  instead of `SamplerNLLEvaluator`, which upstream added after 0.5.7.
- Imports of the vendored modules point at `training._vendor.tinker_cookbook_fw`
  instead of `tinker_cookbook`. All other `tinker_cookbook` imports are
  unchanged and resolve to the upstream package.
- A header comment at the top of each file notes its origin and changes.

The SDFT files outside this directory are also adapted from tinker-cookbook
and carry the same kind of header:

| File | Changes |
| --- | --- |
| `training/utils/distillation/sdft.py` | From the fork's `tinker_cookbook/distillation/sdft.py`. Fireworks changes: the teacher's top-K logprobs come from a top-K forward pass on a frozen `FiretitanTrainingClient`, and training and weight sync run on Firetitan clients and a Fireworks deployment. |
| `training/recipes/sdft/train.py` | Fireworks changes: adds `fireworks_*` and `teacher_*` CLI options and lists the `deepmath` dataset. |
| `training/recipes/sdft/sdft_test.py` | Fireworks changes: tests the Firetitan top-K path with `FiretitanTrainingClient` mocks. |
| `training/recipes/sdft/benchmark.py`, `run_continual_learning.py` | Same as upstream apart from import rewrites. |
| `training/recipes/sdft/datasets.py`, `eval.py` | Same as upstream. |
| `training/recipes/sdft/README.md` | Same as upstream apart from `python -m` paths and an attribution footer. |

Import rewrites in these files point at `training.utils.distillation.sdft`,
`training.recipes.sdft` and this package.

## Updating

Don't edit these files in place. To pick up fork changes:

1. Copy the four files above from the fork's `tinker_cookbook/` directory.
2. Reapply the import rewrites, the `NLLEvaluator` change (unless the pinned
   `tinker-cookbook` release has `SamplerNLLEvaluator`) and the header comments.
3. If a newer `tinker-cookbook` release matches the fork's upstream base, bump
   the `sdft` extra in `training/pyproject.toml` and run `uv lock`.
4. Update the commit hashes and versions in this README and in `__init__.py`.
5. With the `sdft` extra installed, run `pytest recipes/sdft/sdft_test.py`
   from `training/`.
