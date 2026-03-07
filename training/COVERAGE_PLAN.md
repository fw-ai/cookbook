# Training Script Coverage Plan

This document tracks unit coverage for the training entrypoints under `training/recipes`
and `training/examples`, along with the plan to raise that coverage meaningfully.

## Scope

Coverage is tracked for:

- `training/recipes/*.py`
- `training/examples/*/train_*.py`
- `training/examples/*/prepare_data.py`

Coverage is intentionally focused on training entrypoints, not assets, datasets, or test files.

## Baseline

Measured with:

```bash
cd training
pytest -q tests/unit tests/test_smoke_imports.py examples/frozen_lake/test_masking.py \
  --cov=. \
  --cov-report=term-missing \
  --cov-report=json:coverage.json
python tests/coverage_summary.py coverage.json
```

Current scoped baseline after the first coverage pass:

| File | Coverage |
| --- | ---: |
| `training/recipes/sft_loop.py` | 17.5% |
| `training/recipes/dpo_loop.py` | 14.0% |
| `training/recipes/orpo_loop.py` | 16.4% |
| `training/recipes/rl_loop.py` | 18.0% |
| `training/examples/frozen_lake/train_frozen_lake.py` | 29.3% |
| `training/examples/deepmath_rl/train_deepmath.py` | 28.6% |
| `training/examples/text2sql_sft/train_sft.py` | 90.7% |
| `training/examples/deepmath_rl/prepare_data.py` | 90.5% |

Current fast-unit coverage across the scoped training scripts is `28.8%` (`441/1531`).
The biggest remaining gap is still the recipe `main()` functions.

## Targets

Phase 1 targets:

- overall scoped script coverage `>= 70%`
- no recipe below `65%`
- no example training script below `60%`
- no scoped file with `NO DATA`

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
   - use the scoped coverage command above as the local baseline command
2. Extract test seams where needed
   - keep CLI `main()` wrappers thin
   - pull branchy logic into pure or mockable helpers only when it directly unlocks unit tests
3. Add focused unit suites
   - `train_frozen_lake.py`: start with `parse_args`, seed loading, and rollout-row conversion
   - `sft_loop.py`: dataset rendering/filtering and `max_seq_len` resolution
   - `dpo_loop.py` / `orpo_loop.py`: pair tokenization/filtering and batch-flush inputs
   - `rl_loop.py`: reward/filter helpers plus bootstrap decisions
   - `train_deepmath.py`, `train_sft.py`, `prepare_data.py`: CLI translation and dataset transforms
4. Ratchet in CI
   - start with reporting + artifact upload
   - once the first expansion pass lands, add a hard fail-under threshold for the scoped scripts
