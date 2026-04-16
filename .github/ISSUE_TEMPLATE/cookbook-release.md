---
name: Cookbook release
about: Track a cookbook release from scope definition through sign-off
title: "[release] vX.Y.Z"
---

## Release summary

- Target version:
- Release owner:
- Reviewer:
- Internal test owner:
- Target commit:
- Planned release date:

## Scope

- 

## Release notes draft

- 

## Blocking checks

- [ ] Target scope and version are agreed
- [ ] Release commit is merged to `main`
- [ ] `Training CI` is green on the exact release commit
- [ ] Fireworks internal tests passed
- [ ] Link to Fireworks internal test run or sign-off is added below
- [ ] Clean package smoke test passed for `training/`
- [ ] No unresolved release blockers remain

## Fireworks internal test evidence

Link the dashboard, job, ticket, or written sign-off here.

-

## Smoke test evidence

Record the commands, environment, and result here.

```bash
cd training
python -m pip install --upgrade pip
python -m pip install -e '.[dev]' build
pytest -q tests/unit tests/test_smoke_imports.py
python -m build
```

## Rollback plan

- 

## Post-release verification

- [ ] GitHub release created
- [ ] Package artifact published, if applicable
- [ ] Install verification completed
- [ ] Follow-up issues filed, if needed
- [ ] Release issue closed
