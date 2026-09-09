# Debug triage paths

Systematic triage for Fireworks training and deployment issues. For the full
error catalog, read `../../fireworks-training/references/error-reference.md`.

## First step: classify the problem

Use **one AskQuestion** to pick a category, then STOP. Plain language only.

| Category | User might say |
|---|---|
| **Job state** | stuck, not starting, RUNNING at 0%, no progress |
| **Error message** | failed, Internal error, RESOURCE_EXHAUSTED, 429, 412 |
| **Quality** | trained but worse, reward collapsed, no improvement |
| **Resume / checkpoint** | resume broken, checkpoint not found, AlreadyExists |
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

1. Identify resource family: `sftj`, `dpo-job`, or `rftj` (or Training API trainer).
2. `get <id> -o json` — note `State`, timestamps, last progress field.
3. Compare state vs progress (state alone lies — see error-reference).
4. If no progress past the approved timeout → gather evidence, do not replace
   without cancel + user approval.
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

1. Confirm base vs tuned evaluated on the **same** held-out split.
2. Check evaluator/reward discrimination (RFT saturation — identical scores).
3. For RL: check trainer/inference logprob alignment before blaming platform.
4. Read failures, not just aggregate metrics.
5. Hand off to **configure** only when user explicitly wants a new training run.

## Path: Resume / checkpoint

1. Read `../../fireworks-training/references/run-state-and-reporting.md`.
2. Reconcile planned IDs — `AlreadyExists` means query, never blind replace.
3. Training API: confirm cookbook commit and checkpoint name match.
4. Warm-start errors: check `HF_PEFT_ADDON` vs base model conflicts.

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
