# Cookbook release process

This document is the maintainer-facing release guide for the public cookbook.
It mirrors the operator runbook in Notion and describes the GitHub automation
that now backs the process.

## What counts as a cookbook release

Use this process for changes that affect runnable cookbook behavior, packaging,
or the supported backend path, including:

- `training/` package code
- recipe behavior
- packaging or versioning
- install or setup flow
- cookbook smoke-test behavior
- renderer, shape, image, service, or managed-service assumptions

Docs-only changes that do not change runnable cookbook behavior usually do not
need a versioned cookbook release.

## Release model

Every public release is a two-step flow:

1. create a candidate tag `vX.Y.Z-alpha.N`
2. cut the final stable tag `vX.Y.Z` only after blockers are green

The stable tag should point to the exact same validated commit as the candidate.
If the release commit changes after candidate validation, treat it as a new
candidate and rerun validation.

## Repo-owned release automation

### Candidate workflow

`.github/workflows/cookbook-release-candidate.yml` runs automatically on
candidate tags and:

- validates `vX.Y.Z-alpha.N` format
- confirms the tagged commit is reachable from `main`
- confirms `training/pyproject.toml` matches `X.Y.Z`
- runs the reusable `Training CI` workflow
- builds a source distribution and wheel for the exact candidate commit
- generates draft release notes from the git history since the previous stable tag
- uploads the candidate artifact plus a small release manifest

### Stable publish workflow

`.github/workflows/cookbook-release-publish.yml` runs automatically on stable
tags and:

- validates `vX.Y.Z` format
- confirms the tagged commit is reachable from `main`
- confirms `training/pyproject.toml` matches `X.Y.Z`
- confirms there is a candidate tag for the same `X.Y.Z` on the exact same SHA
- locates a successful candidate validation run for that SHA
- requires a matching cookbook release-record issue with explicit release metadata
- requires the release record to contain a recorded `go` decision
- reuses the exact candidate artifact
- publishes the GitHub release with the validated wheel, sdist, and candidate-generated notes

This workflow intentionally refuses to publish if the stable tag does not map
back to a previously validated candidate artifact.

## Release record

Open a release record issue from
`.github/ISSUE_TEMPLATE/cookbook-release.yml` before cutting the final tag.
Use it as the single release record for:

- target version
- candidate commit SHA
- candidate tag
- backend-coupled yes/no decision
- evidence links
- reviewer sign-off
- rollback plan
- final go or no-go readout

The issue template includes a machine-readable metadata block. Keep that block
updated when the candidate SHA or candidate tag changes, because the stable
publish workflow validates it directly.

## End-to-end release checklist

### 1. Decide whether a public release is needed

Confirm that the change affects runnable cookbook behavior, packaging, or the
supported backend path.

### 2. Choose the version

Use normal semver:

- major for breaking changes
- minor for backward-compatible features
- patch for fixes only

If packaging is in scope, update `training/pyproject.toml` so
`[project].version` matches the stable release version `X.Y.Z` before tagging.

### 3. Prepare the candidate commit

Before tagging:

- merge the intended release commit to `main`
- open the cookbook release record issue
- let the candidate workflow generate the first draft of the release notes
- record the exact candidate SHA in the issue

### 4. Create the candidate tag

Tag the exact release commit:

```bash
git checkout main
git pull origin main
git tag vX.Y.Z-alpha.N <candidate-sha>
git push origin vX.Y.Z-alpha.N
```

The candidate workflow will run automatically and produce the validated build
artifact for that SHA.

### 5. Attach required evidence

Record all required evidence in the release issue:

- candidate workflow run
- candidate draft release notes artifact
- `Training CI` result
- cookbook smoke-test result
- Fireworks internal test result or written sign-off
- control-plane smoke-test result when required
- managed-service validation when required
- reviewer sign-off
- rollback plan

Fireworks internal tests and backend validation are still external blockers.
The repo automation does not synthesize those signals yet, so they must be
attached to the release record manually.

### 6. Hold the go or no-go call

Do not create the stable tag until the release owner records an explicit go
decision in the release issue and updates the metadata block accordingly.

### 7. Create the stable tag

Create the stable tag on the exact same SHA as the validated candidate:

```bash
git tag vX.Y.Z <candidate-sha>
git push origin vX.Y.Z
```

The stable publish workflow will fail if:

- the tag version does not match `training/pyproject.toml`
- the tagged SHA is not on `main`
- no candidate tag for `vX.Y.Z-alpha.N` points to that SHA
- no successful candidate workflow run exists for the same artifact
- no matching release-record issue contains the expected metadata and `go` state

### 8. Verify after publication

After the workflow publishes the GitHub release:

- confirm the generated release notes look correct
- confirm the wheel and sdist are attached
- confirm installability if the release depends on packaging changes
- monitor for regressions and open follow-up issues if needed

## Decision matrix

| Change type | Candidate tag required | Training CI | Cookbook smoke tests | Fireworks internal tests | Control-plane smoke test | Managed-service validation |
| --- | --- | --- | --- | --- | --- | --- |
| Docs-only, no runnable behavior change | Usually no public release | No release-specific run | No | No | No | No |
| Recipe, package, install, or runnable UX change | Yes | Yes | Yes | Yes | Usually no | Usually no |
| Renderer, shape, image, service, or backend-coupled behavior change | Yes | Yes | Yes | Yes | Yes | Yes |
| Unclear scope | Yes | Yes | Yes | Yes | Default to yes until waived | Default to yes until waived |

## Common mistakes to avoid

- cutting the stable tag before Fireworks internal tests are green
- validating one commit and releasing another
- skipping backend validation for renderer or shape changes
- tagging a commit that is not on `main`
- forgetting to bump `training/pyproject.toml` before tagging
