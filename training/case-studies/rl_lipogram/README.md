# Lipogram: learn to answer without “e”

## Get the data

Download one archive: [e-vasion-v2.zip](../bundles/e-vasion-v2.zip). A full repository checkout already includes it. The first notebook cell verifies its SHA-256 checksum and restores the data to the paths used below. For command-line scripts, run `python ../common/study_data.py e-vasion` from this folder first.

The archive preserves separate training, development, test and recorded-result files. It does not overwrite files you have changed. See [bundle setup](../bundles/README.md) for manual downloads and verification.

An experiment in useful English answers that avoid a common letter. The goal is
**complete, e-free AND useful answers without a system instruction**. A trained
adapter is probabilistic; this recipe does not guarantee perfect compliance.

The first recorded RL run is a negative result. Its old checker rewarded escaped
letters, and its prompted training metric did not transfer to served use. The saved
four-arm evaluation found 0/240 e-free sampled answers from the adapter alone and
4/240 with the instruction. Historical results remain unchanged; the revised
recipe below needs new experiments.

## Recommended experiment: reasoning + answer demonstrations → SFT → strict RL

Run these commands yourself from this directory, in the installed cookbook Python
environment. Generation, judging, training and served evaluation make paid calls.
All training uses serverless pools; no dedicated deployment is created.

Set the environment using your existing key (do not put keys in recorded commands):

```bash
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

### 1. Prepare and inspect verified demonstrations

Start with the existing train/test files, or build fresh data into a new directory
with `python build_prompts.py --out data/v2`. The prompt contract now covers all
final-answer text, including fences, quotations, Unicode look-alikes and escapes. The separate
reasoning trace is unrestricted: it can name forbidden words while planning. Avoid
requests that demand an incompatible exact string or literal code; inspect those
rejections rather than relaxing the checker.

```bash
python prepare_sft.py \
  --tasks data/lipogram_train.jsonl --held-out data/lipogram_test.jsonl \
  --teacher accounts/fireworks/models/glm-5p3-flash \
  --validation-groups 16 --validation-seed 1 \
  --effort max --max-tokens 16384 --mode reasoning \
  --limit 500 --attempts 4 --out data/sft-v2
```

The builder uses the current instruction even with older input files. It splits
whole prompt groups before generation, checks test overlap, then samples fresh, independent responses. Each retry uses the original request,
not a repair conversation that would be missing from the student input. Only completed answers passing both the mechanical rule and
quality judge are exported. Reasoning mode additionally requires a nonempty
`reasoning_content` returned with that exact answer in the same API response.
No earlier trace is paired with a later answer. Short useful answers are allowed.
The teacher sees the instruction; the exported student messages do not.

Max effort and 16,384 total output tokens are defaults, not a guaranteed sufficient
budget. Inspect truncation and missing-reasoning fields in the audit. If the API
does not return reasoning, reasoning mode rejects the sample rather than silently
training an answer-only dataset. `--mode answer-only` is an explicit alternative.

Outputs:

- `sft.jsonl`: accepted instruction-free pairs with native `reasoning_content`
  plus final-answer `content` (default reasoning mode).
- `sft_answer_only.jsonl`: the exact same accepted prompts and final answers,
  without reasoning, for a controlled comparison.
- `attempts.jsonl`: every trace/answer pair, acceptance, verdict, finish reason,
  response ID and API token usage.
- `validation.jsonl`: untouched validation prompts.
- `rl_train.jsonl`: training prompts after validation removal, for inspection.
- `manifest.json`: input hashes, generation settings and acceptance count.

Review accepted examples for substance and natural wording before training. Review
rejections for impossible tasks and systematic failure. A judge is a filter, not a
proof of correctness. Calibrate it on known useful, incorrect and evasive answers.
Use 300–500 accepted examples as an initial experiment, not a promised minimum.
The limit bounds prompts attempted, not accepted examples. Outputs require a new
directory; failures leave the audit trail, but this builder does not yet resume.

Include varied topics and response lengths, including prompts that tempt copying
forbidden words. Once “egg” has been used for debugging, treat it as a development
case, not evidence of unseen generalization. Keep a separate untouched final test.

### 2. Train SFT and measure it first

```bash
python train_sft.py --dataset data/sft-v2/sft.jsonl \
  --log-path runs/sft-v2 --output-model-id recorded-model-03
```

This uses the shared SFT recipe, rank 64 / alpha 128, one epoch, and loss on
both assistant reasoning and final-answer tokens. Only the final answer is
checked for forbidden letters. The native GLM renderer handles the thinking
boundary; do not insert hand-written tags into `content`.

For the answer-only comparison, use a separate log path and model ID:

```bash
python train_sft.py --dataset data/sft-v2/sft_answer_only.jsonl --mode answer-only \
  --log-path runs/sft-answer-only-v2 --output-model-id glm53-lipogram-answer-only-v2
```

Keep the epoch, optimizer, validation and inference settings identical when
comparing these variants; record training tokens because traces add training cost.

Save the portable **training checkpoint** reference printed by the recipe
(`<account>/<run-id>/step-N`); the sampler snapshot and promoted model ID are not
interchangeable with that reference.

The shared SFT recipe currently uses the GLM renderer's default effort (Max).
Evaluate both `--effort max` and `--effort low` in separate output directories;
do not assume results transfer between effort settings.

```bash
python evaluate_served.py --adapter accounts/YOUR_ACCOUNT/models/recorded-model-03 \
  --tasks data/sft-v2/validation.jsonl --effort max --out results/sft-v2-validation-max
```

### 3. Refine with strict RL

```bash
python lipogram_rl.py --dataset data/lipogram_train.jsonl \
  --init-from-state '<account>/<run-id>/step-N' \
  --eval-prompt-groups 16 --eval-seed 1 --eval-effort low \
  --reward-phase strict --drop-system-prompt-share 1 \
  --train-efforts low,max --max-sample-tokens 8192 \
  --steps 20 --run-dir runs/sft-rl-v2 \
  --output-model-id glm53-lipogram-sft-rl-v2
```

**Use exactly the same original task file, validation-group count and seed as
preparation.** Do not substitute `rl_train.jsonl`: it would carve a different
validation set. This preserves validation groups that neither SFT nor RL trained
on. Initializing from SFT loads weights with a fresh RL optimizer; `--resume-from`
continues an RL checkpoint with its optimizer.

Strict is now the default: violations earn zero, compliant useful answers earn
quality credit. Ramp remains an explicit comparison (`--reward-phase auto` or
`ramp`); partial credit falls with absolute violation count, so clean padding
cannot dilute it. Auto switches on instruction-free validation **joint success**.
The reward checks the final answer only, while GRPO updates the sampled reasoning
and answer tokens together. There is no reasoning-length penalty.
Truncated training samples earn no reward. Log `train/contributing_groups` and
`train/trained`: an attempted rollout with no reward spread is not a weight update.

### 4. Compare on the untouched final test

Run `evaluate_served.py` separately for SFT and SFT+RL, using the same test file,
effort and generation settings. Each run compares base, prompted base, adapter,
and prompted adapter, with greedy and sampled results. The primary comparison is
adapter **without** instruction versus prompted base.

Success means e-free, useful and complete. Judge failures are reported as unknown,
not zero quality; they do not count as successes in the conservative all-draw rate.
Reports include API-reported reasoning tokens (unknown if absent), generation
wall time excluding the judge, truncation, length, family/band breakdowns, paired bootstrap
intervals over prompts, and input/settings provenance. Use unique output directories.
No performance gain should be claimed from training reward alone.

## Offline checks

```bash
python -m pytest test_lipogram_rewards.py test_lipogram_training.py -q
```

The suite checks fences, escaped/look-alike letters, short answers, padding,
group leakage, SFT export, missing judges and paired success intervals. These are
local checks, including a cached-tokenizer GLM loss-mask/parser check; they do
not establish live model quality or service compatibility.

## Notebook dependencies

In addition to the cookbook training package, install the shared display dependencies from this study directory:

```bash
python -m pip install -r ../common/requirements-notebooks.txt
```

This installs Python packages only; it does not start training or inference.

## Reward scoring and resume semantics

New runs use one low-temperature judge call per distinct request/answer pair. Earlier recorded experiments averaged two calls; those saved results have not been recomputed. Repeating a temperature-zero call is not independent evidence. Identical completions within one scoring batch share a verdict. Temporary judge failures remain unscored; missing credentials or permanent HTTP configuration failures abort instead of silently dropping every group.

All visible text, including fenced code, is checked for prohibited characters. Truncated answers receive zero reward and do not count toward completed e-free or answered rates; their token and violation counts remain diagnostic.

`--eval-prompt-groups` counts distinct `(family, group_id)` groups, not prompt rows. Every prompt in each selected group is held out to prevent leakage. `dataset_split.json` records both selected group count and actual evaluation row count; sampling cost scales with rows times `eval_group_size`.

For `--reward-phase auto`, each DCP save writes a local `<checkpoint>.reward-state.json` containing the exact checkpoint reference and phase. Keep this file with the run artifacts and pass it with `--resume-reward-state` when using `--resume-from`. Missing or mismatched state stops before creating a training client. Explicit `ramp` or `strict` is an intentional override. Trainer state alone does not restore this Python-side curriculum.

The study-specific serverless loops remain separate from the maintained shared RL recipe. Consolidating them requires parity checks for reward filtering, grouped holdouts, curriculum state and checkpoint behavior; this change does not claim that refactor is complete.
