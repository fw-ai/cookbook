# Eval Protocol Chat with Async RL

This example connects the cookbook `async_rl_loop` recipe to a single-turn Eval
Protocol chat rollout.

It uses:

- `training.recipes.async_rl_loop`
- Eval Protocol `RemoteRolloutProcessor`
- a small FastAPI chat rollout server that produces one assistant response per
  rollout
- cookbook renderers to pack the completed `EvaluationRow` into `RolloutRun`
- `policy_loss="reinforce"` for a simple RL objective

## Files

- `train.jsonl` is a tiny arithmetic dataset.
- `remote_server/server.py` is a FastAPI `/init` server. It calls the model
  through the tracing `model_base_url` and logs `Status.rollout_finished()`.
- `rollout.py` builds one `EvaluationRow`, invokes `RemoteRolloutProcessor`,
  grades the completed row, and returns one `RolloutRun`.
- `convert.py` uses the cookbook renderer to convert completed chat messages
  into token-native `RolloutSample` data for `async_rl_loop`.
- `train.py` wires the dataset and rollout factory into
  `training.recipes.async_rl_loop.main`.

## How It Works

`RemoteRolloutProcessor` calls the remote server's `/init` endpoint with:

- the current hotloaded deployment model id
- a tracing `model_base_url`
- rollout metadata (`rollout_id`, `run_id`, `row_id`, etc.)

The remote server uses the `model` and `model_base_url` from each request, then
performs one chat-completions call and logs `Status.rollout_finished()`. Future
rollouts hit the latest hotloaded policy behind the stable deployment id
managed by the async RL recipe.

After Eval Protocol hydrates the completed `EvaluationRow`, `convert.py` uses
the cookbook renderer to rebuild prompt token ids from the completed messages.
Completion token ids/logprobs come from tracing payload hydration. The final
`EvaluateResult.score` becomes the scalar reward.

## Run

Install dependencies from `requirements.txt` in addition to the cookbook
package dependencies.

Start the remote server in one terminal:

```bash
export FIREWORKS_API_KEY=...
python -m training.examples.rl.eval_protocol_chat.remote_server.server \
  --host 127.0.0.1 \
  --port 3000
```

Train in another terminal:

```bash
export FIREWORKS_API_KEY=...
export REMOTE_ROLLOUT_BASE_URL=http://127.0.0.1:3000

python -m training.examples.rl.eval_protocol_chat.train \
  --dataset-path train.jsonl \
  --base-model accounts/fireworks/models/qwen3-1p5b-instruct \
  --tokenizer-model Qwen/Qwen2.5-1.5B-Instruct \
  --max-rows 4 \
  --completions-per-prompt 2 \
  --prompt-groups-per-step 1
```

Set `EP_MODEL_BASE_URL` if you need a non-default tracing gateway.

## Chat Completions Flow

This flow is for single-turn Eval Protocol rollouts that use chat completions.
The remote server and `RemoteRolloutProcessor` work with OpenAI-style messages,
while `async_rl_loop` trains on token-native samples.

To bridge those surfaces, this example uses the cookbook renderer after the
rollout to rebuild prompt token ids from the completed Eval Protocol messages.
Completion token ids/logprobs come from tracing payload hydration, and the
Eval Protocol score becomes the reward.

This assumes the renderer matches the chat template used by the model serving
path. Production RL integrations should prefer token ids/logprobs emitted
directly by the generation path whenever possible.
