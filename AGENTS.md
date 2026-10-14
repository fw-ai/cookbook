# Fireworks Training Cookbook

Training recipes, examples, and agent skills for the
[Fireworks Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction),
built on the [Fireworks Python SDK](https://github.com/fw-ai-external/python-sdk)
(`fireworks.training.sdk`).

## Where to start

- **Running or planning training for a user:** read
  [`skills/GETTING-STARTED.md`](skills/GETTING-STARTED.md), then the skill that
  fits the request:
  - [`skills/research/`](skills/research/SKILL.md): the goal is clear but the
    method, data, or evaluation is not.
  - [`skills/configure/`](skills/configure/SKILL.md): plan, launch, monitor,
    evaluate, deploy, and tear down a run.
  - [`skills/debug/`](skills/debug/SKILL.md): a run is stuck, failed, or
    produced a poor model.
  - [`skills/fireworks-training/`](skills/fireworks-training/SKILL.md) holds the
    shared references the three skills load. `skills/discover/` is a redirect
    for older installs.
- **Installing these skills:** follow the Install section of
  [`skills/GETTING-STARTED.md`](skills/GETTING-STARTED.md#install).
- **A first fine-tune:** [`training/examples/muse_glimmer/`](training/examples/muse_glimmer/README.md)
  runs a small SFT job on preloaded example data.
- **Changing recipe or example code:** [`training/README.md`](training/README.md)
  explains the layout and where each kind of change belongs.

## Layout

- `training/`: the installable package (`fireworks-training-cookbook`).
  `recipes/` holds the training loops, `utils/` the shared pieces, `examples/`
  task code that calls a recipe, `case-studies/` end-to-end notebooks, and
  `tests/` the unit and smoke tests.
- `skills/`: agent skills, packaged by `.claude-plugin/` and `.codex-plugin/`
  (the plugin version lives in their `plugin.json`).
- `llms.txt`: canonical doc links for LLM readers.
- `advisor/`, `eval/`, `partners/`, and `archived/` are not part of the
  Training API cookbook. Don't extend them or route users to them.

## Setup and checks

Python 3.11 or newer.

```bash
cd training && uv pip install -e ".[dev]"   # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest tests/unit                            # from training/
```

After editing `skills/`, run from the cookbook root:

```bash
python skills/validate_skills.py
python skills/test_entry_points.py
python skills/test_research_configure_handoff.py
python -m unittest skills.configure.tests.test_cost_estimation_contract -v
```

## Rules for changes

- **Keep skills in sync with the protocol.** Any change to the Training API
  SDK, Tinker protocol, trainer or deployment payloads, checkpoint semantics,
  hotload flow, optimizer-step semantics, or the recipe/SDK compatibility
  contract must update the relevant docs in `skills/`. Delete or rewrite stale
  guidance rather than leaving it next to the new behavior.
- **Check claims against the source.** Facts in skills about pricing,
  promotions, defaults, quotas, or model availability should match the
  Fireworks docs or the control plane. Link the doc page when one exists.
- **Region placement.** Config classes keep an explicit `region` field for
  compatibility with existing runners, but cookbook code must not add
  `deployment_region`, hard-code default regions, infer region from
  accelerator or shape, or copy a trainer region into a hot-load deployment.
  Leave unset values unset so the backend trainer/deployment gateway can choose
  defaults and enforce colocation.
- **Skill size.** `skills/configure/SKILL.md` has a 240-line limit enforced by
  `validate_skills.py`. Put detail in a reference file and link it.

## Running jobs on a real account

Training jobs, deployments, and sampling cost money on the account behind
`FIREWORKS_API_KEY`. Show the user the plan and cost and get approval before
creating anything, never print or log the key, and only delete resources your
session created. The configure skill covers the full flow.
