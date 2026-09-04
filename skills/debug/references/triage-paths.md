# Debug triage paths

Systematic triage for Fireworks training and deployment issues. For the full
error catalog, read `../configure/references/error-reference.md`.

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

## Path: Job state

1. Identify resource family: `sftj`, `dpo-job`, or `rftj` (or Training API trainer).
2. `get <id> -o json` — note `State`, timestamps, last progress field.
3. Compare state vs progress (state alone lies — see error-reference).
4. If no progress past the approved timeout → gather evidence, do not replace
   without cancel + user approval.
5. After three failed hypotheses → escalate with evidence bundle (see below).

## Path: Error message

1. Capture exact status message and request/correlation IDs.
2. Read `../configure/references/error-reference.md` common issues table.
3. Classify platform-side vs user-side vs unknown before acting.
4. De-mask: pull strongest progress signal for the method (export-metrics, W&B).
5. One bounded retry only with user approval; no spend by default.

## Path: Quality

1. Confirm base vs tuned evaluated on the **same** held-out split.
2. Check evaluator/reward discrimination (RFT saturation — identical scores).
3. For RL: check trainer/inference logprob alignment before blaming platform.
4. Read failures, not just aggregate metrics.
5. Hand off to **configure** only when user explicitly wants a new training run.

## Path: Resume / checkpoint

1. Read `../configure/references/run-state-and-reporting.md` resume section.
2. Reconcile planned IDs — `AlreadyExists` means query, never blind replace.
3. Training API: confirm cookbook commit and checkpoint name match.
4. Warm-start errors: check `HF_PEFT_ADDON` vs base model conflicts.

## Path: Deploy / serving

1. Read `../configure/references/deploy-and-troubleshoot.md`.
2. `READY` is not serving proof — send one real request.
3. LoRA routing: verify exact model path and loaded addon state.
4. Multi-LoRA: confirm BF16 shape and `--enable-addons`.

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
| Issue resolved (config fix, transient platform) | report and stop |

## Attribution

```bash
export FIREWORKS_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```

Reuse the existing session UUID when debugging a run already in progress.
