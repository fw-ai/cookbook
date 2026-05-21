---
name: public-repo-promotion
description: Use when diagnosing Fireworks staged public repository promotion issues, including mismatches between `public-repos/python-sdk` or `public-repos/cookbook` and their public GitHub repositories, runtime drift gate failures, bootstrap conflicts, no-op promotes, and manual promote recovery. Assumes staging is the single source of truth and public repos should not receive direct writes.
---

# Public Repo Promotion

Fireworks promotes reviewed snapshots from the internal staging paths to public repositories:

- SDK staging: `public-repos/python-sdk` -> `fw-ai-external/python-sdk`
- Cookbook staging: `public-repos/cookbook` -> `fw-ai/cookbook`

Staging is the single source of truth. Public repos are mirrors. Direct public writes are policy violations unless they are immediately reverted or absorbed through staging.

## First Principles

1. Do not manually patch the public repo to resolve a staging mismatch. Make the fix in `public-repos/<target>/`, merge it internally, then promote.
2. Treat the runtime drift gate as a policy guard. It should block unabsorbed public-only changes that staging did not touch.
3. Bootstrap mismatches are one-time conflicts from content that existed on public before the promote pipeline owned it.
4. A no-op promote is success. If staging and public already match, no public PR should be required.
5. Use one public PR per target. SDK uses the rolling release branch; cookbook uses its rolling promote branch.

## Key Files

- `.github/workflows/public_repo_promote.yml` controls promotion from staging to public.
- `.github/workflows/staging_public_diverge_check.yml` checks staging PRs for public drift.
- `scripts/check_staging_public_diverge.py` classifies mismatches.
- `scripts/sync_public_repo.py` copies/replays staging into the public target and emits `changed=true|false`.
- `scripts/sdk_release_bump.py` appends the SDK release commit after SDK staging changes are replayed.

## Drift Classes

`check_staging_public_diverge.py` compares staging files with public `main`.

| Class | Meaning | Action |
|---|---|---|
| Forward drift | Public last-touch subject starts with `chore: promote ` or `release: `; staging has moved ahead. | Allow. Promote carries staging forward. |
| Backward drift | Public differs on a file that the current staging change did not touch, and public last-touch is not a promote/release commit. | Fail. Revert public or absorb through a staging PR. |
| Intentional override | Current staging change touches the diverging file. | Warn, then allow under the staging-as-SST policy. Reviewer must confirm this is intentional. |
| No drift | Bytes match. | Allow. |

## Resolving A Mismatch

Use this sequence when the promote workflow reports public drift.

1. Identify the target from the failing matrix job: `sdk` or `cookbook`.
2. Read the failing paths printed by `check_staging_public_diverge.py`.
3. Decide whether each path is old bootstrap content or a real direct public edit.
4. For old bootstrap content, create or use a staging PR that touches the path with the intended staging content.
5. For a real direct public edit, either revert it on public or port it into `public-repos/<target>/` in a staging PR.
6. Merge the staging PR.
7. Let the automatic promote run, or use manual recovery if the previous promote failed and no new push will retrigger it.

Do not bypass the gate by weakening token scopes, deleting the drift check, or pushing directly to public.

## Manual Recovery

Manual recovery means running `Public Repo Promote From Staging` from GitHub Actions after a previous promote failed.

Use it when:

- The staging change is already merged.
- The automatic promote failed.
- You need to rerun promotion without creating an unrelated staging change.

Inputs:

- `target`: `sdk` or `cookbook`.
- `staging_ref`: usually `main`, or the reviewed staging SHA.
- `source_history_base_ref`: optional base SHA/ref. Set this for bootstrap recovery so the workflow can compute which `public-repos/<target>/` files changed intentionally between the base and `staging_ref`.
- `dry_run`: start with `true` unless the intended result is clear.

If `source_history_base_ref` is empty, manual recovery runs in strict mode: every public mismatch is binding and can hard-fail. If it is set, changed paths since that base are treated as intentional overrides and produce warnings instead of hard failures.

## Common Failures

### Runtime drift gate blocks a workflow file

Symptom:

```text
Public sdk has 1 file(s) with out-of-band changes ...
  .github/workflows/ci.yml
    public last touched in ... chore: bootstrap ...
```

Interpretation: public has pre-promotion bootstrap history, while staging now owns the workflow. If the staging PR intentionally changed that workflow, promote should pass the path as modified so the gate warns rather than blocks.

Fix: merge the staging change and run promote with a valid history base. For manual recovery, provide `source_history_base_ref`.

### Cookbook says `No generated changes`, then label step fails

Symptom:

```text
No generated changes.
No open promote PR found ...
```

Interpretation: staging and public already match. No public PR exists because no public change is needed.

Fix: finalize steps must be gated on `steps.replay.outputs.changed == 'true'`. Do not make an empty public PR just to satisfy labeling.

### Public drift appears after direct public PR

Interpretation: someone changed the mirror directly.

Fix: under the no-direct-public-write policy, revert that public PR or absorb it into staging. Future public changes should originate from promotion only.

## Review Checklist

- Promotion still requires `PUBLIC_REPO_PROMOTE_TOKEN`; no token fallback to `github.token` or unrelated bot tokens.
- Runtime drift check runs before any public push.
- Push-modified or manually base-diffed paths can warn, but untouched backward drift still fails.
- `sync_public_repo.py` output `changed=false` skips SDK release finalization and cookbook labeling.
- Public workflow files live in staging when staging is the intended source of truth.
- PR text does not claim manual recovery works without a usable `source_history_base_ref` when bootstrap drift is expected.
