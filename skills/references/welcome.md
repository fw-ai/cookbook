# Welcome (first turn)

Show this **once per chat** when the user opens with a vague or first-time
training message. All three skills (research, configure, debug) share this
entry. Read this file before any skill-specific workflow.

## When to show

Show welcome when **all** are true:

1. The user has **not** already picked an entry path in this thread (no prior
   welcome answer, no research handoff, no explicit skill choice).
2. The message does **not** unambiguously map to one skill (see skip rules).

## When to skip

Skip welcome and go straight to the matching skill when the user clearly signals:

| Signal | Skill |
|---|---|
| Train, deploy, resume, SFT/DPO/RFT on named model/data | **configure** |
| Stuck, failed, error, bad quality, job ID + symptom | **debug** |
| Chose **Research** / **Configure** / **Debug** from welcome already | that skill |
| Research handoff manifest exists in thread | **configure** |
| User says "skip intro" or names a cookbook path | **research** |

## Welcome message (required text)

Use this block verbatim (minor formatting ok; keep all three modes):

```text
**Fireworks Training**: three ways I can help:

1. **Research**: explore method, data, eval, and cookbook starting points
2. **Configure**: plan, run, and monitor a training job
3. **Debug**: triage a stuck or failed run

Just describe your goal in plain language, or pick below.
```

Then fire the **Entry** AskQuestion. **STOP** and wait.

Show the privacy notice from `telemetry-notice.md` **once** before the Entry
AskQuestion (same turn, after the welcome block).

Do not start research interview, configure preflight, or debug triage in the same
turn.

## Entry AskQuestion

Title: `Fireworks Training`

Prompt: `What would you like to do?`

| Option ID | Label | Route to |
|---|---|---|
| `research` | Explore: method, data, eval, and cookbook | **research** → skill banner + interview |
| `configure` | Train: plan or run fine-tuning | **configure** → skill banner + preflight |
| `debug` | Fix: triage a stuck or failed run | **debug** → skill banner + triage |

Legacy telemetry: `welcome_choice: discover` maps to `research`.

If the user replies in free text instead of picking an option, infer the route
and confirm with one sentence before continuing. Set `welcome_choice:
inferred_from_text` in telemetry.

## Telemetry

After the Entry answer (or when welcome is skipped), record in the run manifest
and emit `welcome_answered` when `FIREWORKS_API_KEY` is set. See
`telemetry-schema.md`.

| Field | When |
|---|---|
| `welcome_choice` | `research` \| `configure` \| `debug` \| `inferred_from_text` |
| `welcome_skipped` | `true` when skip rules applied (unambiguous intent) |

Append to `intake_responses[]` with `question_id: welcome-entry`, `user_choice`,
`response_source`, `answered_at_utc`. Set `task_summary` (≤200 chars, redacted)
from the opening message when known. See `telemetry-notice.md`.

Create `fireworks-training-runs/<run-id>/run.md` at welcome time if none exists
so journey fields have a home before research/configure starts.

If the user opts out of tracking, set `telemetry_opt_out: true` and skip server
events (see `telemetry-notice.md`).

## After the entry answer

| Choice | Next step |
|---|---|
| Research / explore | Open **research** `references/output-template.md` banner; run interview |
| Configure / train | Open **configure** with banner: `**Configure**: plan, run, and monitor Fireworks training.` |
| Debug / fix | Open **debug** with banner: `**Debug**: triage a stuck or failed training run.` |

Record `entry_skill: welcome` only in the run manifest when one is created at
welcome time; overwrite with the routed skill when that skill takes over.

## Attribution

At welcome time (first training skill touch in the session):

```bash
export FIREWORKS_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export FIREWORKS_CLIENT_SOURCE="fireworks-training-skill/2.2.0"
```
