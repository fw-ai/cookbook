# Cookbook Case-Study Workflow

**Audience:** AI coding agents and humans working on Fireworks Training Cookbook
case studies.

**Goal:** Use this repo as a local WIP sandbox, then promote curated changes through
Fireworks staging when ready. Agents should read this file at the start of any
cookbook or case-study task.

---

## Workspace map

This Cursor workspace spans four repos. Each has a distinct role — do not conflate them.

| Path | Repo | Role | When to touch it |
| --- | --- | --- | --- |
| `/Users/sinan/cookbook` | `fw-ai/cookbook` | **WIP sandbox** — experiments, scratch notebooks, large local data, on a `case-studies/*` branch | Always start here for case-study work |
| `/Users/sinan/fireworks/public-repos/cookbook` | Private staging snapshot | **Promotion target** for reviewed cookbook changes | When a case study is ready to ship |
| `/Users/sinan/fireworks/public-repos/python-sdk` | Private staging snapshot | SDK / Training API protocol changes | When recipes need SDK seam changes |
| `/Users/sinan/docs` | Public docs (Mintlify) | User-facing documentation | When shipping needs docs updates |
| `/Users/sinan/marketing` | Marketing site | Blog posts, landing pages | When shipping needs marketing content |
| `/Users/sinan/fireworks` (rest) | Platform mono-repo | Control plane, serving, infra | Only when the case study exposes a platform bug or needs internal tooling |

### Promotion pipeline (canonical)

```
cookbook (case-studies/* branch)   WIP: develop freely, keep heavy artifacts local
        │
        ▼  curate & copy reviewed files
fireworks/public-repos/cookbook    staging PR in fw-ai/fireworks
        │
        ▼  merge + automation (public_repo_promote.yml)
fw-ai/cookbook (public)            published cookbook
```

**Hard rule:** Do not open PRs directly to `fw-ai/cookbook` or
`fw-ai-external/python-sdk`. Author in staging; let promotion workflows publish
after merge. See `fireworks/public-repos/CLAUDE.md`.

---

## Repo layout

All cookbook work happens in a single checkout at `/Users/sinan/cookbook`
(remote `git@github.com:fw-ai/cookbook.git`). Case-study WIP lives on
`case-studies/*` branches; `main` tracks published cookbook state.

**Branch naming:** `case-studies/<short-name>` (e.g. `case-studies/embedding-support-search`,
`case-studies/agentic-rl-text2sql`).

**Create a new case-study branch:**

```bash
cd /Users/sinan/cookbook
git checkout -b case-studies/<name>
```

---

## Where agents should look first

When the user asks about cookbook training, case studies, or fine-tuning recipes:

1. **Read this file** (`WORKFLOW.md`) for process.
2. **Read `CLAUDE.md`** for cookbook-specific agent rules (region placement, skills).
3. **Read `skills/fireworks-training/SKILL.md`** for Training API routing and lifecycle.
4. **Read `training/README.md`** for recipe catalog and install instructions.
5. **Read the case-study folder README** (if it exists) before editing that case study.

### Primary directories (WIP repo)

```
cookbook/
├── WORKFLOW.md              ← this file (process)
├── CLAUDE.md                ← agent rules for this repo
├── training/
│   ├── recipes/             ← shared training loops (GRPO, SFT, DPO, embedding, …)
│   ├── case-studies/        ← end-to-end notebooks (one folder per case study)
│   ├── utils/               ← shared config, losses, metrics, service helpers
│   ├── examples/            ← smaller worked examples
│   └── tests/               ← unit + e2e tests
├── skills/fireworks-training/   ← agent skill + progressive references
└── eval/                    ← benchmark adapters (HealthBench, etc.)
```

Only `training/` is relevant for most work. Ignore `archived/` unless explicitly asked.

### Staging mirror (promotion target)

```
fireworks/public-repos/cookbook/training/
├── recipes/
├── case-studies/            ← fewer case studies than WIP; this is what ships
├── utils/
└── skills/
```

WIP currently has more case studies than staging. That is expected — staging catches up
when case studies are promoted.

---

## Development workflow

### Phase 1 — Explore in WIP (`cookbook`)

Do all messy work here:

- Iterate on notebooks (`*.ipynb`)
- Add supporting Python modules (`*_tools.py`, `*_reward.py`, evaluators)
- Download datasets and build local indexes (ChromaDB, SQLite, JSONL)
- Run inference, eval, and training cells against real Fireworks APIs
- Keep scratch copies locally (`copy.ipynb`, `copy 2.ipynb`) — never promote these

**Credentials:** `.env` at repo root with `FIREWORKS_API_KEY` (and
`FIREWORKS_ACCOUNT_ID` when provisioning). Never commit secrets.

**Install (from `training/README.md`):**

```bash
cd /Users/sinan/cookbook/training
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre -e .
```

Case-study-only work may need just `pip install fireworks-ai eval-protocol` — check
the case-study README.

### Phase 2 — Curate for promotion

Before copying to staging, clean up the case-study folder:

| Include | Exclude |
| --- | --- |
| Final notebook(s) with clear names | `* copy*.ipynb`, scratch notebooks |
| `README.md` describing the problem, setup, cost | Large datasets (JSONL, CSV trees, SQLite, ChromaDB blobs) |
| Small Python helpers (`sql_tools.py`, `sql_reward.py`) | `*.log`, `tunnel.log`, `wandb/`, `__pycache__/` |
| `prepare_data.py` or `make_*.py` to regenerate data | Model weights, safetensors, eval artifact dirs |
| `.gitignore` entries for local-only paths | IDE workspace files (`.code-workspace`) |
| Remote evaluator package if needed (with `requirements.txt`) | `ep_eval/`, `ep_remote_*/` runtime outputs |

Follow patterns in `training/case-studies/.gitignore`:

- `*.jsonl` ignored except `*.sample.jsonl`
- `*.log`, `*.sqlite`, `data/`, `__pycache__/`, `.ipynb_checkpoints/` ignored

Add a **sample** dataset slice when the notebook needs an example input.

### Phase 3 — Copy to staging

Copy curated files into the matching path under staging:

```bash
# Example: promote agentic_rl_text2sql
STAGING=/Users/sinan/fireworks/public-repos/cookbook/training/case-studies/agentic_rl_text2sql
WIP=/Users/sinan/cookbook/training/case-studies/agentic_rl_text2sql

mkdir -p "$STAGING"
rsync -av --exclude='*.log' --exclude='dev_databases/' --exclude='chromadb_text2sql/' \
  --exclude='*copy*' --exclude='*.jsonl' --exclude='dev.json' \
  --exclude='ep_eval/' --exclude='ep_remote_*/' --exclude='.DS_Store' \
  "$WIP/" "$STAGING/"
```

Also update `training/README.md` case-study table in **both** WIP and staging if
adding a new case study.

If the change touches shared recipes or utils, copy those paths too:

```bash
# Shared code example
rsync -av /Users/sinan/cookbook/training/utils/<file>.py \
  /Users/sinan/fireworks/public-repos/cookbook/training/utils/
```

### Phase 4 — Staging PR in `fireworks`

1. Work on a branch in `/Users/sinan/fireworks` (e.g. `cookbook/agentic-rl-text2sql`).
2. Stage only `public-repos/cookbook/` changes (and `public-repos/python-sdk/` if SDK changed).
3. Run checks:

```bash
cd /Users/sinan/fireworks/public-repos/cookbook
uv run pytest
uv run ruff check .
```

4. Open PR in `fw-ai/fireworks`. CI workflows: `cookbook_unit_tests.yml`,
   `cookbook-skills-validate.yml`, `public_cookbook_ci.yml`.
5. After merge, automation promotes to public `fw-ai/cookbook`.

### Phase 5 — Docs and marketing (optional)

Ship user-facing content in parallel when the case study is public-ready:

| Change | Repo | Path |
| --- | --- | --- |
| Training API / fine-tuning docs | `docs` | `fine-tuning/`, `guides/` |
| API reference changes | `docs` | Regenerate via `make` from adjacent `fireworks` |
| Blog post / landing page | `marketing` | `apps/web/`, `local_drafts/` |

Docs and marketing have their own PR workflows — they do not go through
`public-repos/cookbook` staging.

---

## SDK vs cookbook boundary

| Layer | Location | Owns |
| --- | --- | --- |
| **Recipes / case studies** | `training/recipes/`, `training/case-studies/` | Training loops, rewards, datasets, rendering, metrics |
| **Cookbook utils** | `training/utils/` | Thin helpers, `build_service_client(...)` mapping |
| **SDK** | `public-repos/python-sdk/` | Trainer/deployment provisioning, weight sync, hot-load, checkpoints |

**Recipes must not** construct trainer jobs, deployments, or REST clients directly.
They go through `build_service_client(...)` then `service.create_*`.

If a case study needs a new SDK seam, change **both** staging repos in the same
fireworks PR and update `skills/fireworks-training/` docs.

**Region placement:** Do not hard-code default regions in cookbook code. Leave
`region` unset so the backend selects defaults. See `CLAUDE.md`.

---

## Agent do / don't checklist

### Do

- Start in `cookbook` (on a `case-studies/*` branch) for all case-study WIP.
- Read the case-study README and notebook before editing.
- Keep heavy artifacts local; provide regeneration scripts instead.
- Promote through `fireworks/public-repos/cookbook/`.
- Update `training/README.md` case-study table when adding a new study.
- Update `skills/fireworks-training/` when protocol or recipe contracts change.
- Run `ruff check` and relevant tests before suggesting a staging PR is ready.
- Ask the user before running mutating `firectl` / provisioning commands that cost money.

### Don't

- Don't PR directly to `fw-ai/cookbook` or `fw-ai-external/python-sdk`.
- Don't commit datasets, model weights, ChromaDB indexes, or eval run outputs.
- Don't promote scratch notebooks (`* copy*.ipynb`).
- Don't put infra provisioning logic in case-study notebooks — use SDK seams.
- Don't hard-code regions, deployment regions, or infer region from accelerator type.
- Don't edit `archived/` unless explicitly asked.
- Don't assume the `case-studies/*` branch and staging are in sync — WIP is ahead by design.

---

## Promotion readiness checklist

Use this before copying a case study to staging:

- [ ] One (or two) clearly named final notebooks; scratch copies removed or gitignored
- [ ] `README.md` with: problem statement, prerequisites, install, estimated cost, teardown
- [ ] `prepare_data.py` or documented download steps for required data
- [ ] `.gitignore` covers all local-only artifacts
- [ ] No secrets, API keys, or account-specific IDs hard-coded in notebooks
- [ ] Notebook runs end-to-end on a clean env with documented deps
- [ ] Entry added to `training/README.md` case-study table
- [ ] Shared recipe/utils changes tested (`uv run pytest` in staging copy)
- [ ] Skills docs updated if Training API behavior changed
- [ ] User confirmed which cells provision GPU (cost money) vs are eval-only

---

## Current WIP inventory

Case studies in `cookbook/training/case-studies/` (WIP):

| Folder | Status hint |
| --- | --- |
| `sft_prompt_router`, `sft_cord_receipts`, `dpo_style`, `reasoning_rl`, `embedding_support_search` | Already in staging |
| `sft_judge_alignment` | WIP — has README, ready for promotion review |
| `agentic_rl_text2sql` | Active WIP — inference + RL loop notebooks |
| `countdown_rl`, `agentic_rl_tau2`, `coding_agent_rl_simple` | WIP case studies |
| `coding_eval_braintrust`, `judge_calibration` | WIP / repro |
| `multilora_fleet`, `humansound_rl`, `sft_prompt_router_serverless` | WIP / specialized |

Staging (`fireworks/public-repos/cookbook/training/case-studies/`) currently has 6
case studies plus `partners/voyage-ai/` under `public-repos/cookbook/partners/`.

---

## Related references

| Doc | Path |
| --- | --- |
| Cookbook agent rules | `cookbook/CLAUDE.md` |
| Training skill (routing) | `cookbook/skills/fireworks-training/SKILL.md` |
| Staging authoring rules | `fireworks/public-repos/CLAUDE.md` |
| E2E test guide | `fireworks/public-repos/TRAINING_E2E_TEST_GUIDE.md` |
| Public repo promotion | `fireworks/public-repos/README.md` |
| Recipe catalog + install | `cookbook/training/README.md` |
| Case-study gitignore patterns | `cookbook/training/case-studies/.gitignore` |

---

## Quick decision tree (for agents)

```
User asks about a training case study?
  ├─ Yes → Read WORKFLOW.md (this file) + case-study README
  │         Work in cookbook/training/case-studies/<name>/
  │         Need to ship? → Copy to fireworks/public-repos/cookbook/ → staging PR
  │
  ├─ SDK / protocol / provisioning bug?
  │         → fireworks/public-repos/python-sdk/ (+ cookbook if recipe-facing)
  │
  ├─ Docs update for users?
  │         → /Users/sinan/docs/
  │
  └─ Blog / marketing?
            → /Users/sinan/marketing/
```
