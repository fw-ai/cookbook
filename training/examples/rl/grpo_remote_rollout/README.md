# GRPO with RemoteRolloutProcessor

This example shows how to use the cookbook async RL recipe with:

- GRPO (`policy_loss="grpo"`)
- Fireworks Training SDK infrastructure from `training.recipes.async_rl_loop`
- Eval Protocol `RemoteRolloutProcessor`
- a multi-turn remote rollout server that signals completion through tracing

The example is intentionally small. Swap `remote_server/server.py` for your own
environment server once the wiring is clear.

## Files

- `prepare_data.py` writes a tiny arithmetic JSONL dataset.
- `remote_server/server.py` is a FastAPI `/init` server. It calls the model for
  multiple turns, logs `Status.rollout_finished()`, and emits per-turn prompt
  token ids in rollout extras.
- `rollout.py` builds one `EvaluationRow`, invokes `RemoteRolloutProcessor`,
  grades the hydrated result, and returns one `RolloutSample`.
- `convert.py` converts an evaluated `EvaluationRow` into the async recipe's
  token-native `RolloutSample`.
- `train.py` wires the dataset and rollout factory into
  `training.recipes.async_rl_loop.main`.

## How It Works

`RemoteRolloutProcessor` calls the remote server's `/init` endpoint with:

- the current hotloaded deployment model id
- a tracing `model_base_url`
- rollout metadata (`rollout_id`, `run_id`, `row_id`, etc.)

The remote server should not cache or rewrite model ids across training steps.
It uses the `model` and `model_base_url` from every `/init` request. The
Training SDK `WeightSyncer` hotloads newer weights into the same deployment, so
future rollouts automatically hit the newer policy behind that stable id.

## Run

Install dependencies from `requirements.txt` (includes `eval-protocol`). `run.sh`
only adds the cookbook repo root to `PYTHONPATH`.

From this directory:

```bash
python prepare_data.py
```

Start the remote server in one terminal:

```bash
export FIREWORKS_API_KEY=...
python -m training.examples.rl.grpo_remote_rollout.remote_server.server \
  --host 127.0.0.1 \
  --port 3000
```

Train in another terminal:

```bash
export FIREWORKS_API_KEY=...
export REMOTE_ROLLOUT_BASE_URL=http://127.0.0.1:3000

python -m training.examples.rl.grpo_remote_rollout.train \
  --dataset-path train.jsonl \
  --base-model accounts/fireworks/models/qwen3-1p5b-instruct \
  --tokenizer-model Qwen/Qwen2.5-1.5B-Instruct \
  --max-rows 4 \
  --completions-per-prompt 2 \
  --prompt-groups-per-step 1
```

Set `EP_MODEL_BASE_URL` if you need a non-default tracing gateway. The remote
server receives the tracing URL from `RemoteRolloutProcessor` and the real
inference base URL through completion params.

## Tracing Requirements

GRPO needs token ids, per-token inference logprobs, a loss mask, and a scalar
reward. Raw request/response traces are enough to show remote rollout
completion, but they are not enough for training unless the example can recover
token ids and logprobs.

The preferred path is:

- request `logprobs=True`
- run `RemoteRolloutProcessor(..., include_payloads=True)`
- log only `prompt_ids` per assistant turn in
  `execution_metadata.extra["token_turn_traces"]`
- reconstruct completion token ids/logprobs from tracing payloads
  (`assistant_turn_payloads`) during conversion to `RolloutSample`

Router replay matrices are optional. They are only needed for MoE training with
router replay enabled.
