# RL examples

## Run this with your agent

Paste this into Claude Code, Cursor, Codex, or another coding agent. The
agent installs the Fireworks training skills if they are missing and shows
you one plan with its cost before it creates anything. The human-readable
guide is [Cookbook: Reinforcement Learning](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl).

```text
Run the Fireworks reinforcement learning (GRPO) cookbook for me, starting with the DeepMath example as a small smoke test:
https://github.com/fw-ai/cookbook/tree/main/training/examples/rl/deepmath

If the Fireworks training skills are not installed, install them first.
Claude Code:
claude plugin marketplace add fw-ai/cookbook
claude plugin install fireworks-training@fw-ai-cookbook
Cursor, Codex, or OpenCode. Set -a to cursor, codex, or opencode. Do not pass -a '*':
npx --yes skills add fw-ai/cookbook -g -s fireworks-training research configure debug -a cursor -y

If FIREWORKS_API_KEY is missing, walk me through https://app.fireworks.ai/settings/users/api-keys. Do not ask me to paste the key in chat.
Run only local and read-only steps first. Show me the plan and its cost, and wait for my approval before you create anything. When we are done, clean up only what this run created.
```

Each example is task code (data, prompt, reward, rollout) handed to a recipe.
Where both exist, `train_*.py` runs on a dedicated trainer + deployment
(`recipes/async_rl_loop.py`; `recipes/rl_loop.py` for the strictly on-policy
IGPO example) and `train_*_serverless.py` on the shared pool
(`recipes/experiment/async_rl_loop_serverless.py`) with the same rollout.

| Example | Modality / shape | Dedicated | Serverless |
| --- | --- | --- | --- |
| [`deepmath/`](./deepmath/) | text, single-turn, verifiable reward | `train_deepmath.py` | `train_deepmath_serverless.py` |
| [`visual_toolbench/`](./visual_toolbench/) | vision, multi-turn image tools, rubric judge | `train_visual_toolbench.py` | `train_visual_toolbench_serverless.py` |
| [`frozen_lake/`](./frozen_lake/) | text or image observations, multi-turn tool env (Eval Protocol) | `train_frozen_lake.py` | — |
| [`multihop_qa/`](./multihop_qa/) | multi-turn search, IGPO turn-level reward (`rl_loop` + `igpo.prepare_batch`) | `train_multihop_qa_igpo.py` | — |
| [`experiment/sao/`](./experiment/sao/) | experimental SAO/PPO with a value-head critic | `qwen3_4b_sao.py` | — |
| [`nemo_gym/`](./nemo_gym/) | multi-turn tool use via NeMo Gym resources/agent servers, token-exact proxy | `train.py` | — |

Sandboxed agentic RL lives in [`../harbor_rl/`](../harbor_rl/). Deprecated
examples are in [`../archive/`](../archive/).
