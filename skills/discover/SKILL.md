---
name: discover
description: >-
  Deprecated redirect — use research instead. Renamed in v2.2.0 to match Tinker
  cookbook conventions. For exploring method, data, eval, and cookbook entries
  use research; for training execution use configure; for failures use debug.
---

# discover (redirect)

**Renamed to `research` in v2.2.0.** Load **research** and follow that skill's
instructions.

| Your goal | Skill |
|---|---|
| Explore task, data, eval, cookbook | **research** |
| Plan, run, monitor, deploy training | **configure** |
| Debug a failed or stuck run | **debug** |

Install `fireworks-training-skill/2.2.0` skills:

```bash
npx --yes skills add fw-ai/cookbook -g -s research -s configure -s debug -a cursor -y
```

Do not follow workflow steps in this file. This is a **redirect** stub for v2.1.0
installs that still reference `discover`.
