# Training Script Coverage Baseline

This document tracks unit coverage for the training entrypoints under
`training/recipes` and `training/examples`.

## Scope

Coverage is tracked for:

- `training/recipes/*.py`
- `training/examples/*/train_*.py`
- `training/examples/*/prepare_data.py`

Coverage is intentionally focused on training entrypoints, not assets, datasets,
or test files.

## Measurement Command

```bash
cd training
pytest -q tests/unit tests/test_smoke_imports.py examples/frozen_lake/test_masking.py \
  --cov=. \
  --cov-report=term-missing \
  --cov-report=json:coverage.json
python tests/coverage_summary.py coverage.json
```

## Original Baseline

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

Initial fast-unit coverage across the scoped training scripts was `28.8%`
(`441/1531`).

## Current Progress

Latest measured coverage after the current unit-test expansion:

| File | Coverage |
| --- | ---: |
| `training/recipes/sft_loop.py` | 86.3% |
| `training/recipes/dpo_loop.py` | 81.7% |
| `training/recipes/orpo_loop.py` | 78.1% |
| `training/recipes/rl_loop.py` | 87.0% |
| `training/examples/frozen_lake/train_frozen_lake.py` | 81.4% |
| `training/examples/deepmath_rl/train_deepmath.py` | 91.0% |
| `training/examples/text2sql_sft/train_sft.py` | 90.7% |
| `training/examples/deepmath_rl/prepare_data.py` | 90.5% |

Current fast-unit coverage across the scoped training scripts is `88.2%`
(`1350/1531`).

Phase 1 coverage targets are now fully met for the scoped training scripts.

The CI ratchet now enforces:

- overall scoped coverage `>= 85%`
- per-file scoped coverage `>= 75%`
- no scoped file with `NO DATA`
