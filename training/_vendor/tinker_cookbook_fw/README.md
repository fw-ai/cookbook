# Vendored tinker-cookbook files for SDFT

This directory holds the
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
files that SDFT needs in a Fireworks-modified form, laid out as in
`tinker_cookbook/`. Everything else SDFT uses, including the unchanged SDFT
recipe files (`tinker_cookbook.recipes.sdft.datasets`, `.eval`), is imported
from upstream `tinker-cookbook`, installed with the optional `sdft` extra:

```bash
# from cookbook/training
uv pip install -e '.[sdft]'
```

Only this directory and `training/recipes/sdft_loop.py` import
`tinker_cookbook`. Without the extra, the rest of the cookbook works as before,
and importing either raises an `ImportError` with the install command.

To run SDFT on Fireworks serverless training (no trainer job or deployment to
provision), use the cookbook recipe, which runs `distillation/sdft_serverless.py`:

```bash
python -m training.recipes.sdft_loop   # edit its Config for model, dataset, LR, ...
```

The fork's original entry point, `recipes/sdft/train.py`, runs on a dedicated
trainer job plus an inference deployment.

## Source and license

- Upstream: [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook),
  Copyright 2025 Thinking Machines Lab, licensed under Apache 2.0 (see
  [`LICENSE`](LICENSE)).
- Copied from the Fireworks fork
  [fw-ai-external/tinker-cookbook](https://github.com/fw-ai-external/tinker-cookbook)
  at commit `b1223e5535edea2efb824bfe73a38d35e0067439` (branch `fireworks`).
- The `sdft` extra pins upstream `tinker-cookbook==0.5.7` from PyPI. The fork
  is based on a slightly newer upstream commit
  (`6be6b03dd6f4b69b829b5f4a7f25255514a92ed4`); the one API difference these
  files hit is handled in `supervised/train.py` (see below). `[tool.uv]
  override-dependencies` keeps this project's `transformers==5.10.4` over the
  `transformers<=5.5.4` that tinker-cookbook declares.

## What changed

Every file here except `distillation/sdft_serverless.py` is one the fork
changed. Each differs from upstream in two ways: the fork's Fireworks changes,
and import rewrites made when vendoring. Each file also carries a header
comment summarizing its changes.

| File | Fireworks changes (from the fork) |
| --- | --- |
| `distillation/sdft.py` | The teacher's top-K logprobs come from a top-K forward pass on a frozen `FiretitanTrainingClient`, and training and weight sync run on Firetitan clients and a Fireworks deployment. |
| `recipes/sdft/train.py` | Adds `fireworks_*` and `teacher_*` CLI options and lists the `deepmath` dataset. |
| `recipes/sdft/sdft_test.py` | Tests the Firetitan top-K path with `FiretitanTrainingClient` mocks. |
| `distillation/sdft_serverless.py` | **Not in the fork.** Added by Fireworks AI, branched from `distillation/sdft.py` (`Config` and `main()`, with the same structure and per-step body). It changes only what serverless needs: the client connects to the serverless pool; sampling uses `save_weights_for_sampler` snapshots instead of `WeightSyncer`; the teacher is a never-trained LoRA model (equal to the base model) on its own serverless session, because the pool is LoRA-only; checkpoints use plain `save_state` recorded in `checkpoints.jsonl`, so re-running on the same `log_path` resumes; and the tokenizer and renderer come from the cookbook. All SDFT helpers are imported from `sdft.py`. |
| `checkpoint_utils.py` | Saves checkpoints through `FiretitanTrainingClient`: records the server-canonical `step-N` state name and sampler snapshot name so checkpoints load across trainer jobs. Adds `extract_trainer_job_id`. |
| `rl/train.py` | Trains on `FiretitanTrainingClient` (created with `FiretitanServiceClient`) and pushes sampler weights to a Fireworks deployment with `DeploymentManager` / `WeightSyncer`. Training logprobs come from a separate `cross_entropy` forward pass. The KL reference model is a frozen `FiretitanTrainingClient`. |
| `rl/metrics.py` | `incorporate_kl_penalty` computes reference logprobs with a frozen `FiretitanTrainingClient` forward pass instead of a `tinker.SamplingClient`. |
| `supervised/train.py` | Creates the training client with `FiretitanServiceClient` from `fireworks_base_model`, supports cross-job resume, rejects `load_checkpoint_path`, and tolerates backends that do not return per-datum logprobs. Not used by `recipes/sdft/train.py`; upstream's SFT-stage runners (`run_continual_learning`, `benchmark`) used it. |

Vendoring changes, applied on top of the fork:

- Imports of the vendored files point at `training._vendor.tinker_cookbook_fw`
  instead of `tinker_cookbook`, as do the `python -m` examples in docstrings.
  All other `tinker_cookbook` imports are unchanged and resolve to the upstream
  package.
- `supervised/train.py` uses tinker-cookbook 0.5.7's `NLLEvaluator`, which
  computes test-set NLL with a `cross_entropy` forward on the training client,
  instead of `SamplerNLLEvaluator`, which upstream added after 0.5.7.
- A header comment at the top of each file notes its origin and changes.

## Updating

Don't edit these files in place. To pick up fork changes:

1. Copy the files above from the fork's `tinker_cookbook/` directory, keeping
   the same relative paths.
   Then carry any fork changes to `sdft.main()` over to its serverless branch,
   `sdft_serverless.main()`.
2. Reapply the import rewrites, the `NLLEvaluator` change (unless the pinned
   `tinker-cookbook` release has `SamplerNLLEvaluator`) and the header comments.
3. If a newer `tinker-cookbook` release matches the fork's upstream base, bump
   the `sdft` extra in `training/pyproject.toml` and run `uv lock`.
4. Update the commit hashes and versions in this README and in `__init__.py`.
5. With the `sdft` extra installed, run
   `pytest _vendor/tinker_cookbook_fw/recipes/sdft/sdft_test.py` from
   `training/`.
