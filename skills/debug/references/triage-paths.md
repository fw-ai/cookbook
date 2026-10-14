# Debug triage paths

Systematic triage for Fireworks training and deployment issues. For the full
error catalog, read `../../fireworks-training/references/error-reference.md`.

## First step: classify the problem

When the user's message already makes the category clear (for example "stuck
at 12% for 40 minutes, job X"), state the category and start the read-only
checks in the same turn. Otherwise use **one AskQuestion** to pick a category,
then STOP. Plain language only.

| Category | User might say |
|---|---|
| **Job state** | stuck, not starting, RUNNING at 0%, no progress |
| **Error message** | failed, Internal error, RESOURCE_EXHAUSTED, 429, 412 |
| **Quality** | trained but worse, reward collapsed, no improvement |
| **Resume / checkpoint** | resume broken, checkpoint not found, AlreadyExists, adapter key mismatch |
| **Deploy / serving** | READY but wrong output, LoRA serves base, 404 on inference |

Record `entry_skill: debug` and the category in the run manifest.

**Telemetry:** one AskQuestion with `question_id: debug-q-category`. Map the
answer to `triage_category` (`job_state`, `error_message`, `quality`,
`resume_checkpoint`, `deploy_serving`). Append to `intake_responses[]` and emit
`debug_triage_answered`. See
`../../fireworks-training/references/telemetry.md`.

| Option ID | `triage_category` |
|---|---|
| `job_state` | `job_state` |
| `error_message` | `error_message` |
| `quality` | `quality` |
| `resume_checkpoint` | `resume_checkpoint` |
| `deploy_serving` | `deploy_serving` |

## Path: Job state

1. Identify resource family: SFT, DPO, or RFT job (or Training API session).
2. Read-only `get` through the SDK (`client.supervised_fine_tuning_jobs.get`;
   `pip install fireworks-ai` into a venv if it is missing) or
   `firectl ... get <id> -o json` — note state, timestamps,
   `job_progress` (percent, epoch), and `status.message`. Fetch metrics files
   as in `../../fireworks-training/references/execution-notes.md`.
3. Compare state vs progress (state alone lies — see error-reference). Then
   say which class the evidence points to, and why: **platform or capacity**
   (RUNNING with no progress and no error, queueing, a known incident) or
   **configuration** (status message names the dataset, schema, shape,
   quota, or parameter). If the record cannot tell them apart, say so and
   name the one read that would. `UNKNOWN / NOT PROVIDED` means the field was
   not observed; never restate it as "no error", "no message", or another
   confirmed absence. A `PENDING` Managed SFT or DPO job whose status starts
   "Waiting for a serverless training slot" is queued behind the account's
   concurrent-run quota. Open Serverless Training API sessions use the same
   quota, do not appear in job lists, and hold their slot until the session
   expires, so check the user's own recent sessions before suspecting anyone
   else. List jobs with a `filter` on the base model rather than paging the
   whole account.
4. If no progress past the approved timeout → report the numeric evidence,
   what remains unknown, and the smallest next read. Do not replace without
   cancel + user approval. Give the full escalation bundle only when the stall
   persists after that check, a retry also stalls, the user asks for it, or
   support handoff is the actual next action. Do not propose cancel-and-replace
   on the first ambiguous read.
5. After three failed hypotheses → escalate with evidence bundle (see below).

## Path: Error message

1. Capture exact status message and request/correlation IDs.
2. Read `../../fireworks-training/references/error-reference.md`.
3. Classify platform-side vs user-side vs unknown before acting.
4. De-mask: pull strongest progress signal for the method (W&B, trainer logs,
   `firectl get <id> -o json` progress fields — there is no `sftj export-metrics`).
5. If one bounded retry is justified, recommend it and hand off to Configure
   for cost disclosure and explicit approval. Debug does not run the retry.

## Path: Quality

Use steps 1–5 as an ordered evidence checklist. Run every available read
immediately and report the current diagnosis; do not ask for artifacts that the
job, manifest, or cookbook can locate. When a missing artifact truly blocks the
next distinction, request only that smallest item. Keep the diagnosis focused.

1. **Evaluation parity first.** Base and tuned on the same held-out split,
   same prompts and system prompt, same decoding settings (temperature, max
   tokens, stop sequences, reasoning effort). If you cannot see the eval
   setup, say in one line which of these you could not verify.
2. **Data.** Locate the train and holdout artifacts from the manifest, job
   config, or eval command. Compare chat template and trained turns, then run
   exact ID/content-hash overlap when both are available. If an artifact cannot
   be read, say so and give the remaining read-only steps: fetch the named
   job/eval config to locate the splits; compare format, chat template, system
   prompt, and trained turns; then compare normalized prompt IDs or content
   hashes and report overlap. Never invent a result.
3. **Loss curves.** Read the job's train and eval-loss history and say whether
   it shows overfitting (train falling, eval rising) or underfitting.
4. Read failures, not just aggregate metrics. For RFT, check evaluator
   discrimination (identical scores); for RL, check trainer/inference logprob
   alignment before blaming the platform.
5. Propose one concrete, evidence-conditioned next iteration. Its cost line is
   either a sourced amount or an explicit unknown with the exact resolution
   step/formula. When evidence is incomplete, choose one low-risk discriminating
   experiment (for example, base vs tuned on the same small fixed slice and
   decoding settings), not a menu of speculative fixes. Never omit the useful
   iteration merely because pricing or eligibility still needs Configure.
6. Hand off to **configure** only when user explicitly wants a new training run.

## Path: Resume / checkpoint

1. Read `../../fireworks-training/references/run-state-and-reporting.md`.
2. Reconcile planned IDs — `AlreadyExists` means query, never blind replace.
3. Training API: confirm cookbook commit and checkpoint name match.
4. Warm-start errors: check `HF_PEFT_ADDON` vs base model conflicts.
5. `Adapter key mismatch` or `loaded 0/N tensors`: the checkpoint's LoRA
   modules differ from the trainer. Copy the checkpoint's full saved
   target-module list onto `loraTargetModules` for a new trainer, then load
   the adapter. An explicit list replaces the default, so pass every saved
   name. See the error catalog. Do not retry the same trainer.

## Path: Deploy / serving

1. Read `../../fireworks-training/references/deploy-and-troubleshoot.md`.
2. Inspect deployment state, exact model path, and loaded addon state.
3. For multi-LoRA, inspect BF16 shape and `--enable-addons`.
4. `READY` is not serving proof. If a paid request is needed, hand off to
   Configure for cost disclosure and explicit approval before sending it.

## Three-strike rule

After **three** failed hypotheses on the same issue:

1. Stop guessing.
2. Build an escalation bundle: UTC timestamps, job/trainer/deployment IDs,
   model + shape, CLI/SDK versions, cookbook commit, retry history, what was
   ruled out, redacted identifiers.
3. Tell the user what evidence supports and what remains unknown.

## Handoffs

| Outcome | Route to |
|---|---|
| User wants to retry training with a fix | **configure** (explicit approval) |
| User needs to pick a cookbook entry first | **research** |
| Issue resolved (config fix, transient platform) | `debug_resolved` + report and stop |

## Attribution

For an existing run, read `skill_session_id` from its `run.md` and export that
value before recording Debug events. Only create a session for standalone
triage when neither the environment nor a run manifest provides one.

```bash
if [ -z "${FIREWORKS_SESSION_ID:-}" ]; then
  export FIREWORKS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
fi
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Never replace an existing run's session UUID.
