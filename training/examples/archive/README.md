# Archived examples (deprecated)

Kept importable for existing notebooks and scripts; not maintained as
references and scheduled for removal. Each entry is covered by a current
example:

| Archived | Covered by |
| --- | --- |
| `countdown/` (serverless text GRPO) | `rl/deepmath/train_deepmath_serverless.py`; the Countdown case study in `case-studies/grpo_countdown/` still imports it |
| `single_turn_token_in/` | `recipes/rl_loop.py` (token-in single-turn rollout is built in) |
| `vanilla_sampler.py` | `RolloutSetup.sampler`, injected by both RL recipes |
| `eval_protocol_chat/` | `rl/frozen_lake/` (Eval Protocol rollout processor) |
| `launchers/*.sh` (per-model runs) | the entry point's own CLI, documented in its README |
| `manual_infra_tests.md` | its scripts were removed; re-attach and per-deployment weight sync are covered by SDK/e2e tests |

Do not add new code here.
