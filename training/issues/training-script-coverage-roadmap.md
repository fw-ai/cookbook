# Training Script Coverage Roadmap

This document tracks the plan to raise unit coverage for the training
entrypoints under `training/recipes` and `training/examples`.

## Coverage Targets

Phase 1 targets:

- overall scoped script coverage `>= 70%`
- no recipe below `65%`
- no example training script below `60%`
- no scoped file with `NO DATA`

Current status:

- overall scoped coverage: met (`88.2%`)
- per-file floor for all scoped files: met (`>= 78.1%`)
- no `NO DATA` files in the scoped set: met
- CI ratchet: enabled (`overall >= 85%`, `per-file >= 75%`)

The goal is not line-count inflation. New tests should cover:

- config normalization and validation
- training-shape / `max_seq_len` resolution
- dataset loading, filtering, and datum conversion
- batching, flush, and accumulation logic
- example-script argument translation into recipe configs
- error handling on empty / invalid inputs

## Work Plan

1. Coverage measurement
   - keep `pytest-cov` in `training` dev dependencies
   - make the unit CI job print and upload scoped coverage
   - use the scoped coverage command from
     [training-script-coverage-baseline.md](./training-script-coverage-baseline.md)
     as the local baseline command
2. Extract test seams where needed
   - keep CLI `main()` wrappers thin
   - pull branchy logic into pure or mockable helpers only when it directly
     unlocks unit tests
3. Add focused unit suites
   - `train_frozen_lake.py`: start with `parse_args`, seed loading, and
     rollout-row conversion
   - `sft_loop.py`: dataset rendering/filtering and `max_seq_len` resolution
   - `dpo_loop.py` / `orpo_loop.py`: pair tokenization/filtering and
     batch-flush inputs
   - `rl_loop.py`: reward/filter helpers plus bootstrap decisions
   - `train_deepmath.py`, `train_sft.py`, `prepare_data.py`: CLI translation
     and dataset transforms
4. Ratchet in CI
   - start with reporting + artifact upload
   - enforce a hard fail-under threshold for the scoped scripts
