# HealthBench Professional on Harbor

This package adapts OpenAI's **HealthBench Professional** dataset to
[Harbor](https://github.com/harbor-framework/harbor) 0.19. It generates one
local Harbor task per public case, runs repeated model completions through the
same custom Fireworks agent, applies rubric judging, summarizes a job, and
exports reward-bearing trajectories for reinforcement learning.

This is an evaluation harness, not medical advice or a substitute for clinical
validation. Keep the benchmark examples and generated trajectories private.

## Reproducibility contract

The adapter pins the canonical 525-row `openai/healthbench-professional`
evaluation artifact:

- Hugging Face revision:
  `349962fd46dd02343a0d8a606491baf59154ea1a`
- `healthbench_professional_eval.jsonl` SHA-256:
  `d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2`
- Cases: 525
- Attempts per case: 8
- Sampling: temperature `1.0`, top-p `0.95`, and at most `16,384` output
  tokens
- Reasoning: no forced reasoning or reasoning-effort override
- Transport: one model request with a `1,800` second read timeout and no SDK
  retry; Harbor allows `2,100` seconds for the agent phase

Preparation fails if the downloaded row count, revision, or file digest does
not match this contract. The default full job is therefore **4,200 model
completions** (525 × 8) plus **9,080 independent rubric-judge calls**. Run a
small subset first and estimate both model and OpenAI judge cost before starting
the full job.

OpenAI has not published the exact runnable internal implementation used for
its reported HealthBench results. This adapter follows the public dataset
schema and documented scoring contract; the public
[`openai/simple-evals`](https://github.com/openai/simple-evals) project is the
reference implementation style. Do not claim parity with OpenAI's unpublished
internal harness without a separate parity study.

The score for a response is reconstructed from its signed rubric points and
then length-adjusted using the published HealthBench formula. Aggregates average
all repeated responses before clipping the final mean to the reported range.
Per-case values are rubric coverage signals, not holistic clinical-quality
ratings.

## Install

Requirements are Python 3.12+, Docker, `uv`, a Fireworks API key for the model,
and an OpenAI API key for rubric judging.

```bash
cd public-repos/cookbook/eval/healthbench_professional
uv sync --group dev

export FIREWORKS_API_KEY="..."
export OPENAI_API_KEY="..."
# Optional for Hugging Face authentication/caching:
export HF_TOKEN="..."
```

The environment Docker image installs the OpenAI client used by the verifier.
The keys are supplied at run time and are never baked into an image or generated
task.

## Prepare the pinned dataset

```bash
uv run healthbench-professional prepare
```

By default this writes 525 Harbor tasks under
`datasets/healthbench-professional/`. Generated tasks are ignored by Git. For a
quick local check:

```bash
uv run healthbench-professional prepare --limit 2 --overwrite
```

Each generated task contains a JSON instruction with the original message list,
the pinned task metadata, the common container definition, and verifier assets.
The source rubric and case metadata are available only to the verifier, not to
the model agent. The adapter intentionally excludes the physician reference
response and benchmark canary from every generated task.

## Run

Prepare and run a two-case smoke test before the full benchmark:

```bash
uv run healthbench-professional prepare --limit 2 --overwrite
uv run healthbench-professional run --limit 2 --job-name healthbench-smoke
```

Prepare all cases, then run all 525 with eight attempts each:

```bash
uv run healthbench-professional prepare --overwrite
uv run healthbench-professional run
```

`job.yaml` defaults to the public `accounts/fireworks/models/kimi-k2p6` model.
To target one of your deployments without changing the benchmark settings:

```bash
uv run healthbench-professional run \
  --model accounts/YOUR_ACCOUNT/deployments/YOUR_DEPLOYMENT
```

Use the `healthbench-professional run` wrapper for model and task-limit
overrides. It resolves those values into a complete config and validates that
config against Harbor before any model call. Passing `--model` or `--n-tasks`
directly alongside Harbor's `-c job.yaml` does not safely override this custom
config in Harbor 0.19.

To generate the exact resolved config without launching a run:

```bash
uv run healthbench-professional resolve-job \
  --model accounts/YOUR_ACCOUNT/deployments/YOUR_DEPLOYMENT \
  --output resolved-job.json
```

Do not compare jobs as apples-to-apples unless dataset revision, judge,
sampling, attempt count, agent version, and container image are all held fixed.
The longer request timeout is transport-only: it prevents valid long reasoning
responses from being cut off and does not change model sampling.

## View and summarize

Harbor stores local jobs under `jobs/`. Open its local trajectory viewer with:

```bash
uv run harbor view jobs
```

Create an aggregate summary from one completed job:

```bash
uv run healthbench-professional summarize jobs/JOB_NAME
```

The summary command validates completeness before reporting the aggregate. It
does not silently average partial or malformed trials.

## RL trajectory contract and export

The custom agent defaults to `require_token_trajectory=true`. Every model-call
step must record an ATIF trajectory at `/logs/agent/trajectory.json` with:

- `llm_call_count = 1`;
- the exact `prompt_token_ids` sent to the model;
- the exact `completion_token_ids` returned by the model;
- one generated-token log probability per completion token; and
- counts consistent with those arrays.

The verifier validates that trajectory **before** making a judge call. It also
requires the trajectory's original messages to exactly match the pinned case
and the scored answer bytes to exactly match the validated trajectory's visible
text. The run fails closed if that provenance binding, exact token IDs, or
aligned log probabilities are absent. The RL exporter also fails closed: it
never reconstructs tokens from text, substitutes a tokenizer, or retokenizes an
answer. That avoids silent token-boundary drift between rollout and training.

Benchmark scoring reads only the visible final answer and excludes reasoning.
The RL trajectory separately retains the exact raw prompt and completion token
IDs returned for the model call, together with aligned completion log
probabilities. Keeping those contracts separate prevents hidden reasoning from
changing the benchmark length adjustment while preserving the original rollout
for training.

After a completed job:

```bash
uv run healthbench-professional export-rl jobs/JOB_NAME \
  --output rl-output/healthbench-professional.jsonl
```

The export contains token-in/token-out trajectories and rewards suitable for a
downstream RL data loader. It does not launch training.

## Privacy and contamination

HealthBench maintainers ask that examples not be posted online because public
examples make benchmark contamination more likely. Accordingly:

- prepared tasks, Harbor jobs, trials, traces, and RL exports are local-only
  and ignored by Git;
- this recipe never uploads jobs or trajectories;
- do not commit prompts, rubrics, references, model answers, or judge traces;
- do not publish generated task bundles or Harbor viewer links; and
- share only minimal, manually reviewed excerpts when a private evaluation
  requires qualitative evidence.

API keys, private account identifiers, and deployment identifiers must never be
committed to this directory.

## Development checks

```bash
uv run pytest
```

The configuration tests validate the Harbor job contract, resolved model and
task-limit behavior, pinned sampling settings, token-trajectory requirement,
task verifier environment, and Docker dependency.
