# RL examples

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

Sandboxed agentic RL lives in [`../harbor_rl/`](../harbor_rl/). Deprecated
examples are in [`../archive/`](../archive/).
