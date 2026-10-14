# Research response template

Every research turn must be scannable: **which skill is active** and **what
happens next**.

**First turn in chat:** use the mode routing in `../SKILL.md` until the user
picks Research or describes a task.

## Skill banner (required after welcome)

```text
**Research** — exploring method, data, eval, and cookbook starting points.
```

## When one blocking question remains

```text
**Research** — exploring method, data, eval, and cookbook starting points.

<your provisional recommendation and the defaults you inferred>

<one question — only the blocking one>
```

Rules: one decision-critical question by default. Group only tightly coupled
blockers when needed; state defaults for other gaps. Most turns need none.

## Recommendation (readiness package)

```text
**Research** — exploring method, data, eval, and cookbook starting points.

**Best match:** `<path or slug>` — <why, 2–3 sentences>

| | |
|---|---|
| Method | SFT / DPO / ORPO / RFT / IGPO / distillation / embedding |
| Cookbook entry | `training/...` (case study / example / recipe) |
| Dataset plan | local / bundled / HF candidates / labeling |
| Eval plan | metric on a named holdout + baseline + hook or gap; say if the metric needs paid sampling or a deployment |
| Pre-launch checks | render one row; for vision, validate image chunks and visual tokens against max sequence length |

**Runner-up (if any):** only when genuinely close.

Want me to plan this run? (skip if the user already said to go ahead)
```

Rules:

- Keep the initial recommendation to one screen by default: result, evidence,
  actionable data/eval plan, and next step. Offer deeper references instead of
  narrating every inspected file or runner-up.
- When the user says go, continue in configure the same turn.
- YAML readiness block goes in `run.md` — not customer-facing unless asked.
- Do not tell the user to paste a magic configure sentence.

## What research must never do

- Dump the full catalog unprompted.
- Run a multi-turn interview when defaults would do.
- Start configure planning before the user selects a starter or
  `plan_configure`. After either, present Configure's final plan in the same
  turn.
- Omit the skill banner.
