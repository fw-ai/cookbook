# Launch the complete Humanizer lesson

Use the notebook in your checkout. It needs `train_walkthrough.py`,
`evaluate_walkthrough.py`, `walkthrough/recorded.json`, and `pack/` beside it.
A downloaded notebook on its own cannot run the lesson.

From your **fireworks repository root**:

```bash
cd public-repos/cookbook/training
uv venv --python 3.12 .venv-humanizer
uv pip install --python .venv-humanizer/bin/python jupyterlab ipykernel matplotlib
.venv-humanizer/bin/python -m ipykernel install --user --name humanizer-lesson
cd case-studies/sft_humanizer
../../.venv-humanizer/bin/python -m jupyterlab humanizer_walkthrough.ipynb
```

For a standalone cookbook checkout, start in `cookbook/training` and skip the
first command. Select the **humanizer-lesson** kernel. Run All uses local data by
default. Setup downloads Python packages; the lesson's default cells make no model calls.

## How the live route works

Everything paid happens on Fireworks serverless, with one API key and no
infrastructure to set up:

1. **Train** on the serverless training pool. Your run saves a checkpoint.
2. **Promote** the checkpoint to a model in your account. That model id is the
   name you will call.
3. **Load it onto serverless.** Promotion makes the model READY but not yet
   callable (inference returns NOT_FOUND until this step). One command, about
   five minutes, needs `firectl` 1.8.15 or newer (`firectl upgrade`):

   ```bash
   firectl -a YOUR_ACCOUNT model load-lora YOUR_NEW_MODEL --serverless --wait
   ```

4. **Call it** on serverless inference, the same endpoint you use for the base
   model, with no deployment and no suffix.

The base model is `accounts/fireworks/models/glm-5p3-flash`. Your adapter will be
`accounts/YOUR_ACCOUNT/models/YOUR_NEW_MODEL`. Both are chat-completions model
names; the only difference in a request is the string.

## Move from exploration to training

Install the training package into the same environment, from `cookbook/training`:

```bash
uv pip install --python .venv-humanizer/bin/python -e .
```

Restart the notebook kernel and rerun the preparation cells. Supply
`FIREWORKS_API_KEY` in the environment of the Jupyter process; restart Jupyter if
you set it after launching. Do not put the key in a saved cell.

From the case-study directory, inspect the smoke and tutorial plans:

```bash
../../.venv-humanizer/bin/python train_walkthrough.py --preset smoke
../../.venv-humanizer/bin/python train_walkthrough.py --preset tutorial
```

Their default directories are distinct: `results/lesson-smoke` and
`results/lesson-tutorial`. Add `--execute` only when ready for paid training.
A nonempty directory is refused; use a new `--run-dir` for a deliberate new run.

A smoke without an output model id checks training mechanics and saves a
checkpoint you cannot call. To get a model you can call, name it before training:

```bash
../../.venv-humanizer/bin/python train_walkthrough.py --preset tutorial \
  --output-model-id YOUR_NEW_MODEL
# Review the plan, then add --execute to train and promote.
```

On success, `results/lesson-tutorial/output-model.json` records `model_id`,
`checkpoint`, and `job_id`, and `lesson-result.json` records the completed step
count. The full inference name is `accounts/YOUR_ACCOUNT/models/YOUR_NEW_MODEL`.

## Call the adapter

Nothing to deploy. Send one request to the promoted model id:

```bash
curl -sS https://api.fireworks.ai/inference/v1/chat/completions \
  -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "accounts/YOUR_ACCOUNT/models/YOUR_NEW_MODEL",
    "messages": [
      {"role": "system", "content": "You are an editor. Rewrite the text so it reads naturally for its audience without changing what it says. Keep every fact, number, name, condition, hedge and quotation. Return only the revised text."},
      {"role": "user", "content": "Audience: existing customers\n\nText:\nWe are excited to announce that, starting March 3, Standard-plan tickets may take up to 48 business hours."}
    ],
    "max_tokens": 600,
    "reasoning_effort": "low"
  }'
```

Use the same system prompt the adapter was trained with (the editor instruction
above). The adapter learned the behaviour under that prompt; without it you are
testing something else.

What the first replies tell you:

| Response | Meaning | What to do |
|---|---|---|
| `404` model not found | Not loaded onto serverless yet (the usual cause), or the id is wrong | Run the `load-lora --serverless --wait` step above; check `output-model.json` |
| Slow first reply | The adapter weights are being fetched and loaded for the first time | Normal. Measured on 2026-10-12: first call 80 s wall, 44 s to first token; second call 1.3 s |
| A rewrite comes back | The adapter is live | Go on to the evaluation |
| `401` / `403` | The key is not for this account | Use the account-scoped key that trained the model |

Run this probe once before evaluating. It is one small paid request, and it tells
you which row you are in before you spend on a whole evaluation.

## Evaluate under matched conditions

The evaluator compares base and adapter through the same serverless endpoint,
same prompt, same decoding, so the only difference is the weights. From the
case-study directory:

```bash
../../.venv-humanizer/bin/python evaluate_walkthrough.py \
  --adapter accounts/YOUR_ACCOUNT/models/YOUR_NEW_MODEL \
  --run-dir results/lesson-tutorial --split dev --max-inputs 5
# Add --execute to opt in to paid generation and model judging.
```

The evaluator first checks the judge against known-good and known-bad edits
(`calibrate_judge.py`) and stops if the judge gets one wrong, then generates and
judges five development passages for both arms. Each execution prints a fresh
directory such as `eval-dev-<UTC timestamp>-<unique suffix>` and keeps partial
evidence after a failure. A `404` from the adapter is an availability error, not
a quality result.

Inspect `samples.jsonl`, `summary.json`, and `summary.md` there. When you are
happy with the recipe, run the untouched test set once:

```bash
../../.venv-humanizer/bin/python evaluate_walkthrough.py \
  --adapter accounts/YOUR_ACCOUNT/models/YOUR_NEW_MODEL \
  --run-dir results/lesson-tutorial --split test --max-inputs 0 --idempotence --execute
```

Five development passages is a smoke of the measurement. The 99-passage test run
with the idempotence pass is the comparison; its summary is the one to keep.

## What it costs, and how to stop it

- Training bills per training token on the serverless pool: the two-epoch
  tutorial is on the order of a dollar; the smoke is cents.
- Inference bills per token on the base model and the adapter alike; the 99-
  passage test run with judging is a few dollars.
- Nothing bills while idle, but a loaded adapter holds a slot in a shared pool.
  The pool for a base model is capped (16 adapters for GLM 5.3 Flash as of
  2026-10-12), and a full pool answers `ResourceExhausted ... capacity ... is
  full` at load time. When you are done evaluating, free your slot:

  ```bash
  firectl -a YOUR_ACCOUNT model unload-lora YOUR_NEW_MODEL --serverless
  ```

  The model stays in your account and can be loaded again later. To remove it
  entirely, delete it from your account's model list.

## Recorded serving conditions

The October 8 evaluation used a dedicated multi-LoRA deployment. The October 12 seed-29 evaluation used the public inference endpoint. These are separate recorded runs; use the matching results directory when comparing their metrics.
