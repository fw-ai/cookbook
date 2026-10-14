# Finding cookbook code

Installed skills contain only Markdown. Every `training/...` path in these
skills (recipes, examples, case studies, starter scripts) is relative to the
root of the [fw-ai/cookbook](https://github.com/fw-ai/cookbook) repository, not
to the skill folder or the user's project.

Resolve an existing cookbook root once per session:

```bash
if [ -d training/recipes ]; then
  COOKBOOK_DIR="$PWD"                      # the workspace is the cookbook
elif [ -n "$FIREWORKS_COOKBOOK_DIR" ] && \
     [ -d "$FIREWORKS_COOKBOOK_DIR/training/recipes" ]; then
  COOKBOOK_DIR="$FIREWORKS_COOKBOOK_DIR"
elif [ -d "$HOME/.cache/fireworks-cookbook/training/recipes" ]; then
  COOKBOOK_DIR="$HOME/.cache/fireworks-cookbook"
fi
[ -n "$COOKBOOK_DIR" ] && git -C "$COOKBOOK_DIR" rev-parse --short HEAD
```

For inspection, prefer single-file reads from
`https://raw.githubusercontent.com/fw-ai/cookbook/main/<path>` when no checkout
exists; do not clone merely to inspect or guess content from a path name. When
the user asks to execute cookbook code and a checkout is required, disclose the
target directory and shallow-clone then, recording the commit.

Install dependencies only when execution requires them, in an isolated virtual
environment after stating what will be created:

```bash
cd "$COOKBOOK_DIR/training"
python3 -m venv .venv && .venv/bin/pip install -e .   # needs Python 3.11+ (check python3 --version, else python3.12); uv works too
.venv/bin/python -m examples.sft.muse_glimmer_starter plan
```

Any Python 3.11 or newer works (3.14 was tested); `uv` and 3.12 are optional.

Copy a recipe or example into the user's project only when they want to modify
it; record the source path and cookbook commit next to the copy.

Before launching or evaluating a run, read
[`execution-notes.md`](execution-notes.md): signed-URL fetches, render-sample checks, resolved-config read-back, queueing, and fair base-vs-tuned
evaluation.
