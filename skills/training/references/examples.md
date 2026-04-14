# Examples: where to look, what to copy

Worked examples live in `training/examples/`. This page lists them and points at the files you would copy; it does not describe what each task does (that's in the Fireworks docs).

| Path | Type | Copy when you need |
|------|------|-------|
| `examples/rl/deepmath/train_deepmath.py` | RL, single-turn | A verifier-based reward for math/code (parses `\boxed{}` and compares) |
| `examples/rl/frozen_lake/train_frozen_lake.py` | RL, multi-turn tool-use | Tool-call rollouts, action masking, visual validation |
| `examples/multihop_qa/train_multihop_qa_igpo.py` | IGPO, multi-turn | Per-turn IG rewards, local TF-IDF search env, turn-boundary detection |
| `examples/dpo/train_dpo.py` | DPO | Minimal preference-pair training setup |
| `examples/orpo/ifeval/train_ifeval_orpo.py` | ORPO | Instruction-following preference training without a reference model |
| `examples/sft/train_sft.py` | SFT | Minimum SFT config, text and VL variants |
| `examples/hotload_reattach/test_reattach.py` | Test | Deployment re-attach flow as an integration test template |

## Files you typically need to touch when forking

| What you're customizing | Edit |
|-------------------------|------|
| Reward function | Function passed as `reward_fn=` in the recipe's `Config` |
| Rollout processor (multi-turn tool-use) | `<example>/<name>_rollout.py` |
| Dataset preparation | `<example>/prepare_data.py` |
| Dataset format | The JSONL file itself; recipes expect `prompt` or `messages` |
| Model / shape / region | `Config.base_model`, `Config.infra.training_shape_id`, `Config.infra.region` |
| Loss algorithm (RL only) | `Config.policy_loss` -- see [`rl.md`](rl.md) |
| Hyperparameters | `Config.lr`, `Config.group_size`, `Config.weight_sync.weight_sync_interval`, etc. |

## Sibling files in each example

Most example directories contain:

- `train_<name>.py` -- main entry point (`python -m examples.<...>.train_<name>`)
- `<name>_rollout.py` or equivalent -- rollout processor (RL only)
- `prepare_data.py` -- one-off dataset prep (download, filter, format)
- `dataset.jsonl` -- bundled example data (when small enough)
- `run.sh` / `run_<model>.sh` -- convenience bash runners with pre-filled args
- `README.md` -- dataset-specific notes and hyperparameter tuning tips (when present)

## Assets and artifacts

Some examples include validation assets you should NOT commit or modify:

- `examples/rl/frozen_lake/validation_screenshots/` -- rollout rendering for debugging
- `examples/rl/frozen_lake/assets/` -- grid images
- `examples/rl/deepmath/deepmath_logs/` -- personal run logs (gitignored)

If you fork frozen_lake, keep `rendering.py` and `verify_rollout.py` around -- the rollout debugging workflow depends on them.
