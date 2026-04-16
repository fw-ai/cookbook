# Cookbook Release Process

This document defines the release process for the Fireworks Cookbook repository.
It is intentionally lightweight and borrows two ideas from mature OSS projects:

- a simple semver-based stable release flow, similar to Tinker Cookbook
- an explicit release-blocker checklist, similar to PyTorch

For this repository, the most important addition is a hard gate on Fireworks
internal test coverage before a release can be cut.

## Scope

This process covers the release surface owned by this repository:

- the `fireworks-training-cookbook` Python package in `training/`
- GitHub tags and GitHub release notes for cookbook releases
- any README or docs updates required to explain the released behavior

## Release types

### Stable releases

- Format: `vX.Y.Z`
- Source branch: `main`
- Use for user-visible changes that should be supported and announced

### Patch releases

- Format: `vX.Y.Z`
- Scope is limited to bug fixes, documentation fixes, packaging fixes, and
  release blockers
- No new features unless they are required to fix a release-critical regression

### Future release types

Nightly or release-candidate flows can be added later, but they are out of
scope for the first cookbook release process. The immediate goal is a safe,
repeatable stable-release path.

## Roles

Each release should identify:

- **Release owner**: drives the checklist and owns the final go/no-go call
- **Reviewer**: validates scope, release notes, and rollback readiness
- **Internal test owner**: runs or confirms the Fireworks internal test suite

One person can cover more than one role for small releases, but the internal
test sign-off should be explicit.

## Hard release blockers

A cookbook release is blocked until all of the following are true:

1. The target scope and version are documented in a release issue.
2. The release commit is merged to `main`.
3. `Training CI` is green on the exact commit being released.
4. Fireworks internal tests pass for the release candidate commit.
5. A clean install/build smoke test passes for `training/`.
6. There are no unresolved release-blocking bugs for the chosen version.

### Fireworks internal tests

Fireworks internal tests are a required blocker for cookbook releases.

For now, these tests live outside this repository, so the gate is process-based
rather than fully automated. That means:

- do **not** create the final tag until the internal suite is green
- link the internal test run, dashboard, or owner sign-off in the release issue
- treat any failure in the internal suite as a release blocker

Longer-term, this should become an automated required status check that must
pass before any release workflow can publish artifacts.

## Release checklist

### 1. Open a release issue

Create a GitHub issue from the cookbook release template and fill in:

- target version
- release owner
- scope
- rollback plan
- link to the Fireworks internal test run once available

The release issue is the single place to track blockers and sign-off.

### 2. Confirm scope and choose the version

Use normal semver rules:

- **major**: breaking changes
- **minor**: backward-compatible features
- **patch**: fixes only

For the current cookbook, most releases will likely be minor or patch.

### 3. Prepare the release commit

At minimum:

- update `training/pyproject.toml` version if a package release is part of scope
- make any required README or docs updates
- draft release notes in the release issue

### 4. Merge to `main`

Release from a commit already merged to `main`. Avoid releasing from an
unmerged branch tip.

### 5. Validate repository-owned checks

Before tagging, confirm:

- `Training CI` is green on the exact commit
- any relevant manual checks requested in the release issue are complete

### 6. Run Fireworks internal tests

This is the additional cookbook-specific release gate.

The release owner should not proceed until:

- the internal test owner confirms the internal suite ran against the release
  candidate commit or equivalent build
- the result is linked in the release issue
- any failures are either fixed or explicitly triaged as non-blocking by the
  release owner and reviewer

If there is uncertainty, the release stays blocked.

### 7. Run a clean package smoke test

From `training/`, verify the package still builds and imports cleanly:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]' build
pytest -q tests/unit tests/test_smoke_imports.py
python -m build
```

If a release is intended for external users, also verify installability from a
fresh virtual environment before publishing.

### 8. Cut the release

Once all blockers are cleared:

1. create the annotated tag `vX.Y.Z`
2. push the tag
3. create the GitHub release
4. publish the package artifact if the release includes `training/`

For the first iteration of this process, the exact publish mechanics can remain
manual as long as the release issue records what was published and from which
commit.

### 9. Post-release verification

After publishing:

- verify the GitHub release notes are correct
- verify the package can be installed from the published source
- monitor for regressions or packaging issues

### 10. Announce and close

Close the release issue only after:

- links to the release artifacts are added
- any follow-up work is filed
- the rollout is considered healthy

## Rollback guidance

If a bad release escapes:

- stop further promotion immediately
- file a follow-up issue with the observed problem
- prefer a fast patch release over force-rewriting public history
- document the rollback or remediation steps in the release issue

## First follow-up automations

After the process is working manually, the next automations should be:

1. a tag-driven publish workflow for `fireworks-training-cookbook`
2. a nightly build flow if the team wants faster pre-release validation
3. a required external status check for Fireworks internal tests
4. generated release notes from merged PRs/issues
