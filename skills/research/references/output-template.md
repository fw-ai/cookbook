# Research response template

Every research turn must be scannable: **which skill is active** and **what
happens next**.

**First turn in chat:** use `../references/welcome.md` until the user picks
Research or describes a task.

## Skill banner (required after welcome)

```text
**Research** — exploring method, data, eval, and cookbook starting points.
```

## During interview (questions still open)

```text
**Research** — exploring method, data, eval, and cookbook starting points.

<AskQuestion — one question from interview-questions.md>

**Next:** readiness package after <N> more answer(s) / this answer.
```

Rules: one AskQuestion per turn; STOP and wait; no final recommendation yet.

## After interview (readiness package)

```text
**Research** — exploring method, data, eval, and cookbook starting points.

**Best match:** `<path or slug>` — <why, 2–3 sentences>

| | |
|---|---|
| Method | SFT / DPO / RFT / embedding |
| Cookbook entry | `training/...` (case study / example / recipe) |
| Dataset plan | local / bundled / HF candidates / labeling |
| Eval plan | metric + baseline + hook or gap |

**Runner-up (if any):** only when genuinely close.

<AskQuestion — handoff from interview-questions.md>

**Not ready to train?** Stay in research for README summary or labeling help.
```

Rules:

- Hand off through AskQuestion only.
- YAML readiness block goes in `run.md` — not customer-facing unless asked.
- Do not tell the user to paste a magic configure sentence.

## What research must never do

- Dump the full catalog unprompted.
- Recommend before completion gate (unless bypassed).
- Start configure planning or `firectl` in the same turn.
- Omit the skill banner.
