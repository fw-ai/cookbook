# Execution notes

Practical details that make a real run go smoothly. Each came from a live run.
Apply them while working; in replies, report what you checked and found
rather than restating these notes.

Environment and credential setup is in `sdk-setup.md` and `getting-started.md`.

## Model and data checks

- Model support for Managed SFT: read `supervisedLoraTunable` (LoRA) together
  with `useTrainingV2` from `client.models.get(...).model_dump()`. The
  full-parameter flag (`supervisedFullParameterTunable`) is hidden from
  non-superuser reads, so its absence does not mean full-parameter is
  unsupported; the create call decides. The older `tunable` field is retired
  and is absent or false.
- Account datasets cannot be downloaded through the API: downloads need a
  per-account flag that is off by default, and even then require
  `firectl signin` or the console, never an API key. Inspect account data
  through dataset metadata before launch and the job's render samples after
  create.

## Resource IDs

Create calls can hang. Give the client a timeout
(`Fireworks(account_id=..., timeout=120)`); if a create times out, `get` the
planned ID before retrying, and retry with the same ID only if it does not
exist. The render-samples URL can return 404 for the first minute after a job is
created; wait and fetch again.

Use a fresh job and output-model ID for every run (for example append a UTC
timestamp), and never reuse the ID of a deleted job. Job artifacts are stored
by job ID: a live run that reused a deleted job's ID failed immediately with
`INTERNAL` and exposed the old job's `metrics.jsonl`. Before reporting losses,
check that the metrics file was written after the job's `create_time`.

On busy shared accounts, list with a `filter` (for example
`base_model="accounts/fireworks/models/muse-glimmer-30b"`) and a small
`page_size`; unfiltered lists can fail with a message-too-large error.

## Resolved configuration

For Managed LoRA SFT, always pass `lora_rank` explicitly (8 or 16 for a first
run). On a base-model job an unset `lora_rank` is `0`, which requests
full-parameter tuning, not LoRA; it is rejected for models without a
full-parameter trainer, and otherwise produces a full model. After the job
finishes, confirm the output model's `kind` is `HF_PEFT_ADDON` with the expected
`peft_details.r`.

The plan gives concrete values. When the platform resolves a value at create
time (Managed learning rate, batch size, eval carve-out size), write "resolved
at create" rather than guessing, then read it back and report it:

- the job record for epochs, rank, dataset, and carve-out setting (the API does
  not report the carve-out size; derive it from steps and batch size and say
  it is derived);
- the first rows of `metrics.jsonl` (`metricsFileSignedUrl`) for `train/lr`,
  `train/step`, and `train/total_tokens`. A job record `learning_rate` of `0.0`
  means unset, not zero.

## Monitoring a Managed job

- Right after create, read a few rows of `render_samples.jsonl`
  (`job.model_dump()["renderSamplesSignedUrl"]`): decoded tokens, per-token loss
  weights, `train_on_what`, `image_count`, and `image_token_count`. Summarize
  across all rows (for example the min and max image tokens), not a few. If the
  rendering or masking is wrong, cancel and say why. Fetch signed URLs with
  `httpx` and `certifi`; python.org builds of Python on macOS often lack CA
  certificates for `urllib`.
- Managed SFT and DPO jobs draw runs from the serverless training pool, which
  has a per-account concurrent-run quota (documented default 8; accounts can
  differ, so quote the limit from the job's status message or a 429 rather than
  assuming the default). A
  `PENDING` job whose status starts "Waiting for a serverless training slot" is
  queued, not stuck. Serverless Training API sessions draw from the same
  quota, do not appear in job lists, and release their slot only when the run
  becomes terminal after the session expires (see the docs' serverless
  capacity section), so an open or recently closed session of yours can hold
  it. Quote `job.status.message`, explain the wait, and do not guess whose run
  holds the slot. Never cancel another run to jump the queue
  without approval.
- `training/examples/sft/muse_glimmer_starter.py watch` prints progress, the
  status message, a render-sample summary, final losses, and the resolved
  learning rate.
- To follow a background run, write its output straight to a log file and read
  the file. Piping through `grep` or `tee` buffers output, so a healthy run can
  look silent; use `python -u` and `grep --line-buffered` if you must filter.

## Reporting what still costs money

Name only documented facts. The serverless docs say checkpoint storage is
currently included at no charge. No public page states a storage price for
Managed SFT output models, so say the model stays on the account and that no
storage charge is documented, rather than calling it free. The promotion
covers training, not sampling: report in-session evaluation sampling as its own
cost line (estimated or unverified), never inside a `$0` total. On-demand
deployments bill per GPU-hour while replicas are running.

## Evaluation

Always read the job's `metrics.jsonl` before reporting. Train and eval loss are
available there even when the job record shows no metrics fields. If eval loss
rose, say so plainly and treat it as a warning, then look at learning rate,
steps, data size, duplicates, and masking. Judge success against the approved
held-out task metric and baseline, since loss and accuracy can move apart.
Use the same decoding settings (`max_tokens`, temperature, stop sequences) for
base and tuned. Before reporting a surprising number (for example a 0% base),
read a few raw outputs and count unterminated samples (`termination.is_clean`
false): those ran out of `max_tokens` and measure the budget, not the model. Keep per-example raw outputs and predictions for base and tuned on the same holdout so you can
report a paired comparison (for example an exact McNemar test) and intervals,
not only two means. The starter's serverless path writes
`predictions_base.jsonl`, `predictions_tuned.jsonl`, and a `paired` block in
`summary.json`.
