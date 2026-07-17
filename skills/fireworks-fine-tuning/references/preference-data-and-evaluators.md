# Preference-pair generation and evaluator authoring

*Use this reference when the user has prompts but no DPO pairs, needs a reusable evaluator, or wants the agent to turn a plain-language success criterion into executable evaluation.*

Pilot generated preference pairs and evaluators inside its hosted workspace. The coding-agent flow can preserve that capability transparently: write artifacts in the user's workspace, show them for review, validate them locally, and create Fireworks resources only after approval.

## Route by the signal the user has

| User has | Recommended path |
|---|---|
| Ideal labeled answers | SFT. Do not manufacture preference pairs. |
| Human or model-ranked pairs | DPO or ORPO. Normalize to the managed preference schema. |
| Prompts only, plus a clear preference criterion | Generate pairs, review a sample, then run DPO or ORPO. |
| Prompts plus objective correctness | Managed RFT with a registered evaluator, or Training SDK RFT with an inline reward. |
| Open-ended quality criteria | Write and calibrate an LLM-judge rubric before training. |

Never silently turn prompts into preference data. Pair generation adds inference cost and embeds the generator or judge's bias into the training set.

## Generate preference pairs from prompts

### 1. Define the contract

Before inference, record:

- the source dataset and held-out split;
- the behavior that makes one answer better;
- the strong and weak generators, or the sampling settings that create meaningful variation;
- the judge or deterministic rule;
- tie, refusal, invalid-output, and low-confidence handling;
- sample count, token ceiling, estimated inference cost, and output path.

Keep the held-out evaluation split out of pair generation. Otherwise the downstream comparison leaks evaluation examples into training.

### 2. Plan + approve

Show the user the generation plan and inference cost before calling any model. Pair generation, judging, upload, and training are protected actions. A useful first pass is a small sample that proves the rubric separates outputs before generating the full set.

### 3. Generate two candidates

Use the Fireworks chat-completions API or Python SDK. Preserve for every row:

- source row ID or stable hash;
- prompt messages;
- both candidate texts;
- generator model and sampling settings;
- judge decision, score, confidence, and rubric version.

Do not include secrets, private file paths, or hidden chain-of-thought in the output.

### 4. Judge and normalize

Map the winner into the managed DPO schema:

```jsonl
{"input":{"messages":[{"role":"user","content":"..."}]},"preferred_output":[{"role":"assistant","content":"better"}],"non_preferred_output":[{"role":"assistant","content":"worse"}]}
```

Drop ties and invalid pairs by default. Do not randomly break ties. Report:

- attempted prompts;
- valid pairs;
- ties and invalids;
- preference rate by generator position;
- judge confidence distribution;
- a manually reviewed sample of wins and losses.

Randomize candidate order before judging and check for position bias. If one generator wins nearly every row, the pairs may be too easy to teach useful preferences.

### 5. Review, then upload

Show the user representative pairs and the aggregate quality report. Upload only after approval. Preserve the generated JSONL and provenance beside the run manifest. Then run the local DPO validator and `firectl dataset create`.

## Author an evaluator

Start with a short evaluator specification before code:

```markdown
Inputs:
Expected output:
Score range:
Exact and partial credit:
Invalid output behavior:
Edge cases:
Calibration examples:
```

Show the spec to the user and resolve ambiguity before implementing the evaluator.

### Managed RFT evaluator

Managed RFT uses a registered eval3 evaluator with an `entry_point`. Use Eval Protocol's current code-first flow and defer exact APIs to the live [evaluator docs](https://docs.fireworks.ai/fine-tuning/evaluators.md).

1. Write the Eval Protocol reward in the workspace.
2. Add deterministic unit examples for full credit, partial credit, zero, malformed output, and edge cases.
3. Run **offline-only** tests by importing and calling the reward function directly with local fixtures. Do not run Eval Protocol's registration-enabled `pytest` flow, authenticate, upload, or call Fireworks before approval.
4. Check that scores vary. All-identical scores make RFT a no-op.
5. Compute and record a SHA-256 hash of the reviewed evaluator source. Choose a stable evaluator display name when the current API supports one.
6. Show the evaluator code, dependencies, offline sample scores, source hash, planned name, and exact registration action.
7. Register only after approval. Eval Protocol's `pytest` flow may register both the evaluator and dataset, so treat invoking that flow as the protected registration action, not as a local test.
8. Preserve the full evaluator resource name, display name, source hash, and registration output in the run manifest.

Evaluator creation may be restricted by account role. Check the current docs and attempt no workaround. If `CreateEvaluatorV2` returns `403`, an admin authors or registers the evaluator once; the scoped agent can then launch managed RFT with that evaluator ID.

If registration returns ambiguously or the response is lost, do not run it again. Use the Eval Protocol output, account UI, or current evaluator API to locate a resource matching the planned display name and source hash. If it cannot be identified unambiguously, ask an admin or support to reconcile it before retrying.

### Training SDK inline reward

For custom rollouts, multi-turn trajectories, or reward logic that should remain in code, fork the cookbook `rl_loop` or `async_rl_loop` recipe and implement `reward_fn`. No registered evaluator is required.

The reward may read `ground_truth`, another reference field, tool outcomes, environment state, or a judge result. Declare the required fields in the plan and validate those fields locally. Do not require `ground_truth` when the selected reward does not use it.

Use `references/training-api.md` for recipe, rollout, deployment-shape, and numerics guidance.

### SFT or DPO candidate evaluator

An evaluator used to select a sweep winner does not need to be a managed RFT evaluator. It can be:

- deterministic exact match or structured validation;
- per-label accuracy and confusion summaries;
- a user-provided script;
- an LLM judge with a reviewed rubric.

Run the same evaluator against the base model and every candidate on the same held-out split. Record evaluator version, sample count, raw aggregate result, and failure count in the run manifest.

## Evaluator quality gate

Before using an evaluator to select or train a model:

1. It passes deterministic unit examples.
2. It produces nonconstant scores on representative samples.
3. The user has reviewed the contract and examples.
4. Its dependencies and credential needs are explicit.
5. Its inference cost is included in the plan.
6. It cannot mutate customer or platform resources.
7. Base and tuned models are evaluated on the same held-out data.

If these checks fail, stop and improve the evaluator. A sophisticated training loop cannot recover from a reward that does not measure the intended behavior.
