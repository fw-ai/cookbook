# Renderer Compatibility Matrix for RL Rollouts

This audit captures the RL-relevant behavior of every renderer in scope for
the renderer-backed rollout helpers.  Two cookbook renderers (`gemma4` and
`kimi_k25` from Tinker upstream) form the AC-10 canary matrix; the others
(`glm5`, `minimax_m2`, `nemotron`, `qwen3`) are out of the canary matrix
but may still be wired into rollouts via `build_renderer(...)`.

This document supersedes ad-hoc reading of renderer modules during rollout
debugging.  It does not change the AC-10 canary set (still `gemma4` +
`kimi_k25` only); broader renderer support is a follow-up if a concrete
consumer arrives.

| Renderer                   | Source                | `has_extension_property`                     | `get_stop_sequences()` shape | Tool calling   | Multimodal chunks | Parse-failure mode | Notes |
|----------------------------|-----------------------|----------------------------------------------|-------------------------------|----------------|-------------------|---------------------|-------|
| `gemma4`                   | cookbook              | `True` (always)                              | `list[int]`                   | yes            | text-only here    | `(message, False)` on truncation / unparseable | Canary; full multi-turn supported. |
| `kimi_k25`                 | tinker-cookbook       | inherits `KimiK2Renderer` base default       | `list[int]`                   | yes            | text-only here    | `(message, False)` on truncation / unparseable | Canary; full multi-turn supported when extension is preserved by the configured mode. |
| `qwen3`                    | tinker-cookbook       | `not strip_thinking_from_history`            | `list[int]`                   | yes            | text-only         | `(message, False)` on truncation / unparseable | **Default `strip_thinking_from_history=True` => `False`** — multi-turn rollouts must reconfigure or hit the AC-3 guard. |
| `glm5`                     | cookbook              | `not self.strip_thinking_from_history`       | `list[int]`                   | yes            | text-only         | `(message, False)` | Same Qwen3-style extension caveat. |
| `minimax_m2`               | cookbook              | `not self.strip_thinking_from_history`       | `list[int]`                   | yes (custom format with `_format_tool_calls`) | text-only | `(message, False)` | Same extension caveat. |
| `nemotron`                 | cookbook              | inherits `Qwen3Renderer`                     | `list[int]`                   | yes            | text-only         | `(message, False)` | Same extension caveat. |

Key points for RL rollouts:

* **Stop-sequence shape is `list[int]`** for every in-scope renderer today;
  the helpers and SDK primitive still preserve `list[str] | list[int]`
  end-to-end so future renderers are unconstrained.
* **Tool calling is parsed renderer-side** in every renderer that supports
  it.  The cookbook tool example (`multi_turn_tool/rollout.py`) extracts
  `tool_calls` from the parsed assistant message and routes them to the
  user-supplied env.  Renderer-side tool *execution* is rejected.
* **Extension-property hazard**: every renderer except `gemma4` exposes a
  mode in which `has_extension_property=False`.  Multi-turn flatten loops
  trip the AC-3 guard before the second sampling call when this happens.
  Users either reconfigure the renderer (`strip_thinking_from_history=False`)
  or stay on a single-turn loop.
* **Multimodal chunks**: every renderer produces `EncodedTextChunk` only in
  text-only mode.  The `model_input_to_token_ids` adapter raises
  `MultimodalRenderingNotSupported` if any other chunk type appears.
  Multimodal RL is out of scope for this iteration.
* **Parse-failure contract**: every renderer's `parse_response` returns
  `(message, parse_success)`.  Truncated outputs (`finish_reason="length"`)
  with no stop token observed typically yield `parse_success=False`.  The
  framework deliberately does not bake a parse-failure-policy enum; users
  branch on `parse_success` inside their `reward_fn` (DROP via `return None`
  or zero-reward via `return 0.0`).

The above informs the AC-10 parametrized tests (`tests/unit/test_rl_renderer_behavior.py`):
the canary suite covers `gemma4` and `kimi_k25` for prompt+stops round-trip,
parse_response success/failure, tool-call round-trip (kimi_k25 only — gemma4
also supports tool calling but coverage is concentrated on the upstream
canary), and `has_extension_property` across modes.
