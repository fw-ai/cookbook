# Renderers

Cookbook-local renderer implementations (`gemma4`, `glm5`, `minimax_m2`,
`nemotron`). Each module subclasses
`tinker_cookbook.renderers.base.Renderer` and registers itself via
`register_renderer(...)`, so it's reachable through
`get_renderer(name, tokenizer)`.

This folder also hosts the **verifier** under
[`verifier/`](./verifier/) — the validation half of any renderer
change. Always pair adding/editing a renderer with a verifier run.

## Adding a new renderer

→ [`cookbook/skills/renderer/SKILL.md`](../../skills/renderer/SKILL.md)
covers the contract (4 methods), the training-mechanics invariants,
common shape decisions (stop signal, thinking modes, tool calls), the
implementation flow, and the gotchas.

## Verifying a renderer

→ [`cookbook/skills/verifier/SKILL.md`](../../skills/verifier/SKILL.md)
covers the workflow: pick renderer / tokenizer / model, edit the
inspection rules, run interactively or batch, inspect the audit table.
The verifier highlights every token by provenance and flags
attribute combinations the rule file marks as worth a closer look:

![Verifier — token stream](./verifier/images/token_stream.png)

For agents triaging a renderer-shaped problem, the dev skill router
([`cookbook/skills/dev/SKILL.md`](../../skills/dev/SKILL.md)) routes
both questions ("how do I add a renderer?" / "why does my model emit
unexpected tokens?") to those two skills.
